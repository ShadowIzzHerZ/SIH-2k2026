"""Shared dataset building, training and evaluation for the drift experiments.

Every experiment uses the same drive-level split (prep.py -> split.json), the
same test windows/chains, and the same report shape, so rows in results.md
are directly comparable.
"""
from __future__ import annotations

import copy
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.calibration import calibrate, estimate_gravity_vector, leveling_rotation
from src.chain_eval import ChainTrainDataset, build_chains, chained_drift
from src.data.windowing import IOVNBDWindowDataset, build_windows, calibrate_sequence
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import drift_metric
from src.train import chain_loss, compute_loss, forward_pass

ROOT = Path(__file__).resolve().parent
DT = 0.1
WS = 50
SLOW = 6.0
LOSS_WEIGHTS = {"speed": 1.0, "heading": 10.0, "heading_end": 20.0, "chain": 1.0, "drift": 2.0}
MODEL_KW = dict(input_channels=6, cnn_channels=[32, 64, 64], cnn_kernel_size=5, gru_hidden=128,
                gru_layers=2, dropout=0.2, output_dim=2)
torch.set_num_threads(int(__import__("os").environ.get("NTHREADS", 4)))


def load_sequences():
    seqs = pickle.load(open(ROOT / "cache" / "sequences.pkl", "rb"))
    split = json.load(open(ROOT / "split.json"))["sequence_split"]
    return seqs, split


# ---------------------------------------------------------------- calibration
def _smooth(x, k=10):
    # edge-normalised moving average (plain mode="same" pulls the ends toward 0,
    # which turns into a huge fake dv/dt at both ends of every drive)
    return np.convolve(x, np.ones(k), mode="same") / np.convolve(np.ones_like(x), np.ones(k), mode="same")


def calibrate_fwdfix(seq, min_speed=3.0):
    """Leveling from the whole drive's mean accel (gravity dominates it), then
    the forward axis from GPS: in the leveled frame, horizontal accel rotated
    by psi should equal [dv/dt, v*yaw_rate] (longitudinal, centripetal).
    Closed form: psi = angle(sum(conj(a) * target)) with both as complex
    numbers. The repo's estimate_yaw_misalignment compares phone-frame accel
    angles with world compass headings, which mixes in the mean compass
    heading of the drive, so its 'forward' axis is close to arbitrary."""
    seq = copy.deepcopy(seq)
    g = estimate_gravity_vector(seq.accel.astype(np.float64))
    R = leveling_rotation(g)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        a = seq.accel.astype(np.float64) @ R.T
        w = seq.gyro.astype(np.float64) @ R.T
    v = np.asarray(seq.speed_gt, dtype=np.float64)
    dvdt = np.gradient(_smooth(v), DT)
    lat = _smooth(v) * _smooth(w[:, 2])
    ac = _smooth(a[:, 0]) + 1j * _smooth(a[:, 1])
    tc = dvdt + 1j * lat
    ok = (v > min_speed) & np.isfinite(tc) & np.isfinite(ac)
    psi = float(np.angle(np.sum(np.conj(ac[ok]) * tc[ok]))) if ok.sum() > 100 else 0.0
    av, gv = calibrate(seq.accel.astype(np.float64), seq.gyro.astype(np.float64), R, psi)
    seq.accel = av.astype(np.float32)
    seq.gyro = gv.astype(np.float32)
    fwd = av[ok, 0]
    r = float(np.corrcoef(_smooth(av[:, 0])[ok], dvdt[ok])[0, 1]) if ok.sum() > 100 else float("nan")
    seq.calib_info = {"psi_deg": float(np.degrees(psi)), "r_long": r}
    return seq


def calibrate_orig(seq):
    seq = copy.deepcopy(seq)
    return calibrate_sequence(seq)


# ---------------------------------------------------------------- datasets
def build_splits(calib="orig", stride=10, seq_filter=None):
    """Returns dict of IOVNBDWindowDataset: train, val, test, test_iovnbd, test_comma."""
    seqs, split = load_sequences()
    cal = {"orig": calibrate_orig, "fwdfix": calibrate_fwdfix}[calib] if isinstance(calib, str) else calib
    wins = {"train": [], "val": [], "test_iovnbd": [], "test_comma": []}
    info, cal_seqs = {}, {}
    for k, s in seqs.items():
        sp = split[k]
        if seq_filter is not None and not seq_filter(k, sp):
            continue
        cs = cal(s)
        info[k] = getattr(cs, "calib_info", None)
        cal_seqs[k] = cs
        ws = build_windows(cs, WS, stride, DT)
        for w in ws:
            w.src = k
        if sp == "test":
            wins["test_iovnbd" if k.startswith("iovnbd") else "test_comma"].extend(ws)
        else:
            wins[sp].extend(ws)
    out = {n: IOVNBDWindowDataset(w) for n, w in wins.items()}
    out["test"] = IOVNBDWindowDataset(wins["test_iovnbd"] + wins["test_comma"])
    out["_calib_info"] = info
    out["_seqs"] = cal_seqs
    return out


# ---------------------------------------------------------------- model
def new_model(**over):
    kw = dict(MODEL_KW)
    kw.update(over)
    return BiasCorrectionNet(**kw)


class ZeroNet(torch.nn.Module):
    """Physics only: no learned correction."""

    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], 2)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def window_drifts(model, ds, fwd=forward_pass):
    model.eval()
    out = []
    for batch in DataLoader(ds, batch_size=256, shuffle=False):
        _, pos_pred, _, pos_gt = fwd(model, batch, DT, "cpu")
        out.extend(drift_metric(pos_pred, pos_gt).numpy().tolist())
    return np.array(out)


def _stats(d):
    d = np.asarray(d)
    if len(d) == 0:
        return None
    return {"n": int(len(d)), "median": float(np.median(d)), "mean": float(d.mean()),
            "p90": float(np.percentile(d, 90)), "under10": float((d < 10).mean() * 100)}


def evaluate(model, splits, chain_fn=chained_drift, fwd=forward_pass, sets=("test_iovnbd", "test_comma")):
    """Window 5 s and chained 10/30/60 s drift, per dataset, all/slow/fast."""
    res = {}
    for name in sets:
        ds = splits[name]
        if len(ds) == 0:
            continue
        d = window_drifts(model, ds, fwd)
        ms = np.array([w.speed_gt.mean() for w in ds.windows])
        r = {"5s": {"all": _stats(d), "slow": _stats(d[ms < SLOW]), "fast": _stats(d[ms >= SLOW])}}
        for n in (2, 6, 12):
            row = {}
            for rn, rng in (("all", None), ("slow", (0.0, SLOW)), ("fast", (SLOW, 1e9))):
                c = chain_fn(model, ds, "cpu", DT, window_size=WS, n_chunks=n, max_chains=None, speed_range=rng)
                row[rn] = ({"n": c["n_chains"], "median": c["median_drift_pct"], "mean": c["mean_drift_pct"],
                            "p90": c["p90_drift_pct"], "under10": c["pass_rate_pct"]} if c else None)
            r[f"{n * 5}s"] = row
        res[name] = r
    return res


def headline(res):
    """Compact row: median drift (%) for the cells that matter."""
    def g(ds, k, sp="all"):
        try:
            return res[ds][k][sp]["median"]
        except (KeyError, TypeError):
            return float("nan")
    return {
        "io_5s": g("test_iovnbd", "5s"), "io_30s": g("test_iovnbd", "30s"), "io_60s": g("test_iovnbd", "60s"),
        "io_30s_slow": g("test_iovnbd", "30s", "slow"), "io_30s_fast": g("test_iovnbd", "30s", "fast"),
        "comma_5s": g("test_comma", "5s"), "comma_30s": g("test_comma", "30s"),
    }


def fmt_headline(h):
    return " | ".join(f"{k} {v:5.1f}" for k, v in h.items())


# ---------------------------------------------------------------- training
def val_score(model, splits, chain_fn=chained_drift, fwd=forward_pass):
    d = window_drifts(model, splits["val"], fwd)
    c = chain_fn(model, splits["val"], "cpu", DT, window_size=WS, n_chunks=6, max_chains=600)
    return float(np.median(d)) + c["median_drift_pct"], float(np.median(d)), c["median_drift_pct"]


def train(model, splits, name, stage1_epochs=6, stage2_epochs=6, lr=1e-3, lr2=3e-4, chunks=3,
          max_chains=12000, weights=None, loss_fn=None, chain_loss_fn=None, chain_fn=chained_drift,
          fwd=forward_pass, train_ds=None, seed=42, log=print):
    """v3 recipe on a CPU budget: stage 1 single-window, stage 2 chained-chunk
    fine-tune (K=3). Keeps the epoch with the best val score (window median +
    chained 30 s median). Same budget for every experiment."""
    torch.manual_seed(seed)
    weights = dict(weights or LOSS_WEIGHTS)
    loss_fn = loss_fn or (lambda b: compute_loss(*fwd(model, b, DT, "cpu"), weights))
    chain_loss_fn = chain_loss_fn or (lambda b: chain_loss(model, b, DT, "cpu", weights))
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    best = (float("inf"), None)
    hist = []
    tr = train_ds or splits["train"]
    stages = [("single", stage1_epochs, lr, DataLoader(tr, batch_size=64, shuffle=True))]
    if stage2_epochs:
        chains = build_chains(tr, WS, chunks, chain_stride_chunks=2, max_chains=max_chains, seed=seed)
        stages.append(("chain", stage2_epochs, lr2,
                       DataLoader(ChainTrainDataset(tr, chains), batch_size=21, shuffle=True)))
    ep = 0
    for stage, n_ep, stage_lr, loader in stages:
        if n_ep == 0:
            continue
        if stage == "chain" and best[1] is not None:
            model.load_state_dict(best[1])
        opt = torch.optim.AdamW(model.parameters(), lr=stage_lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
        for _ in range(n_ep):
            t0 = time.time()
            model.train()
            tot, nb = 0.0, 0
            for b in loader:
                opt.zero_grad()
                loss, _ = chain_loss_fn(b) if stage == "chain" else loss_fn(b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss)
                nb += 1
            sched.step()
            model.eval()
            score, wmed, cmed = val_score(model, splits, chain_fn, fwd)
            hist.append({"epoch": ep, "stage": stage, "loss": tot / max(nb, 1), "val_win_median": wmed,
                         "val_chain30_median": cmed, "score": score, "sec": time.time() - t0})
            log(f"[{name}] ep {ep} {stage} loss {tot / max(nb, 1):.2f} | val 5s med {wmed:.2f} "
                f"30s med {cmed:.2f} | score {score:.2f} | {time.time() - t0:.0f}s")
            if score < best[0]:
                best = (score, copy.deepcopy(model.state_dict()))
                torch.save(best[1], ckpt / f"{name}.pt")
            ep += 1
    model.load_state_dict(best[1])
    model.eval()
    return model, hist


def save_result(name, desc, res, hist=None, extra=None):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rec = {"name": name, "desc": desc, "headline": headline(res), "full": res, "history": hist, "extra": extra}
    json.dump(rec, open(out / f"{name}.json", "w"), indent=1)
    return rec


# ---------------------------------------------------------------- pluggable chained eval
@torch.no_grad()
def chained_drift_x(model, dataset, device, dt, window_size=50, n_chunks=6, max_chains=800,
                    speed_range=None, oracle=None, step_fn=None, batch_size=128, bias_fn=None, **_):
    """Same chains and metric as src.chain_eval.chained_drift, with hooks:
    oracle='speed' uses true speed, oracle='heading' uses the true direction of
    travel (error decomposition); step_fn(raw, c, v, th, imu) -> (speed, heading)
    replaces the integrator for an idea that changes the dynamics."""
    chains = build_chains(dataset, window_size, n_chunks, max_chains=max_chains, speed_range=speed_range)
    if not chains:
        return {}
    model.eval()
    drifts = []
    for b0 in range(0, len(chains), batch_size):
        cb = chains[b0:b0 + batch_size]
        items = [[dataset[i] for i in ids] for ids in cb]
        B = len(cb)
        v = torch.tensor([it[0]["v0"] for it in items], dtype=torch.float32)
        th = torch.tensor([it[0]["theta0"] for it in items], dtype=torch.float32)
        bias = None
        if bias_fn is not None:  # (B, 6) per-chain IMU bias from the GPS before the blackout
            bias = torch.tensor(np.stack([bias_fn(dataset.windows[ids[0]]) for ids in cb]), dtype=torch.float32)
        p = torch.zeros(B, 2)
        gt_end = torch.zeros(B, 2)
        dist = torch.zeros(B)
        for j in range(n_chunks):
            imu = torch.stack([it[j]["imu"] for it in items])
            raw = torch.stack([it[j]["imu_raw"] for it in items])
            pos_gt = torch.stack([it[j]["pos_gt"] for it in items])
            sgt = torch.stack([it[j]["speed_gt"] for it in items])
            if bias is not None:
                raw = raw - bias.unsqueeze(1)
                imu = raw
            c = model(imu)
            if step_fn is not None:
                speed, heading = step_fn(raw, c, v, th, imu)
            else:
                acc = raw[..., 0] + c[..., 0]
                vs, vv = [], v
                for k in range(acc.shape[1]):
                    vv = (vv + acc[:, k] * dt).clamp(0.0, 50.0)
                    vs.append(vv)
                speed = torch.stack(vs, dim=1)
                heading = th.unsqueeze(1) + torch.cumsum((raw[..., 5] + c[..., 1]) * dt, dim=1)
            if oracle == "speed":
                speed = sgt
            if oracle == "heading":
                d = torch.diff(torch.cat([torch.zeros(B, 1, 2), pos_gt], 1), dim=1)
                u = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
                step = speed.unsqueeze(-1) * u * dt
            else:
                step = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1) * dt
            p = p + step.sum(dim=1)
            v, th = speed[:, -1], heading[:, -1]
            if oracle == "heading":  # keep the chain's heading on truth too
                th = torch.atan2(u[:, -1, 1], u[:, -1, 0])
            gt_end = gt_end + pos_gt[:, -1, :]
            dist = dist + torch.linalg.norm(pos_gt[:, 1:] - pos_gt[:, :-1], dim=-1).sum(dim=1)
        drifts.extend((100.0 * torch.linalg.norm(p - gt_end, dim=-1) / dist.clamp(min=1e-6)).tolist())
    d = np.array(drifts)
    return {"n_chains": int(len(d)), "mean_drift_pct": float(d.mean()), "median_drift_pct": float(np.median(d)),
            "p90_drift_pct": float(np.percentile(d, 90)), "pass_rate_pct": float((d < 10.0).mean() * 100)}


def decompose(model, ds, n_chunks=6, speed_range=None):
    """Median chained drift: as is / with true speed / with true heading."""
    out = {}
    for o in (None, "speed", "heading"):
        r = chained_drift_x(model, ds, "cpu", DT, n_chunks=n_chunks, max_chains=None, speed_range=speed_range, oracle=o)
        out[o or "model"] = r.get("median_drift_pct", float("nan"))
    return out
