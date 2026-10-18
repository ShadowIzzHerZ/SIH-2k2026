"""Off-by-one fix for prediction vs ground truth.

pos_gt[k] (windowing.py) is the position k steps after the window's first
sample, so pos_gt[0] = 0. src.models.strapdown_ins integrates with an
inclusive cumsum, so pos_pred[k] is the position after k+1 steps and
speed_pred[0] already includes one step of acceleration. A perfect
constant-velocity prediction therefore scores ~2% drift over a 5 s window,
and the speed/heading losses compare samples one step apart. Chained
evaluation also dropped the step from each window's last sample to the next
window's first sample (in prediction and truth alike).

Here everything is aligned:
  speed[k]   = v0 + sum_{i<k} a_i dt          (speed[0] = v0)
  heading[k] = theta0 + sum_{i<k} w_i dt       (heading[0] = theta0)
  pos[k]     = sum_{i<k} speed_i dt [cos, sin](heading_i)   (pos[0] = 0)
A 5 s window compares the T-1 steps inside it. A chain integrates all T
steps of every chunk, so chunk j ends exactly where chunk j+1 starts, and
the truth includes the step between windows ("gt_next", from the drive's
own GPS track).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.drift import lib
from src.chain_eval import build_chains
from src.data.io_vnbd_loader import latlon_to_local_xy
from src.models.strapdown_ins import drift_metric
from src.train import compute_loss


def _excl(x):
    return torch.cumsum(x, 1) - x


def integrate(raw, c, v0, th0, dt=lib.DT):
    """Returns speed, heading, pos (B,T[,2]) aligned with the truth, plus the
    state at the next window's first sample and the full T-step displacement."""
    acc = raw[..., 0] + c[..., 0]
    w = raw[..., 5] + c[..., 1]
    speed = torch.clamp(v0.unsqueeze(1) + _excl(acc * dt), min=0.0)
    heading = th0.unsqueeze(1) + _excl(w * dt)
    step = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], -1) * dt
    pos = torch.cumsum(step, 1) - step
    v_next = torch.clamp(speed[:, -1] + acc[:, -1] * dt, min=0.0)
    th_next = heading[:, -1] + w[:, -1] * dt
    return speed, heading, pos, v_next, th_next, pos[:, -1] + step[:, -1]


def make_fwd(set_v0=False):
    def fwd(model, batch, dt, device):
        if set_v0:
            model.cur_v0 = batch["v0"]
        c = model(batch["imu"])
        speed, _, pos, *_ = integrate(batch["imu_raw"], c, batch["v0"], batch["theta0"], dt)
        return speed, pos, batch["speed_gt"], batch["pos_gt"]
    return fwd


fwd_aligned = make_fwd(False)


def make_loss(model, set_v0=False, aux_w=0.0):
    fwd = make_fwd(set_v0)

    def fn(b):
        loss, m = compute_loss(*fwd(model, b, lib.DT, "cpu"), lib.LOSS_WEIGHTS)
        if aux_w:
            loss = loss + aux_w * torch.nn.functional.smooth_l1_loss(model.aux, b["speed_gt"])
        return loss, m
    return fn


class NextStep(torch.utils.data.Dataset):
    """Adds gt_next: the true step from the window's last sample to the next
    window's first sample (zeros if the GPS is missing there)."""

    def __init__(self, base, seqs):
        self.base, self.windows = base, base.windows
        xy = {}
        self.nxt = np.zeros((len(self.windows), 2), dtype=np.float32)
        for i, w in enumerate(self.windows):
            if w.src not in xy:
                s = seqs[w.src]
                xy[w.src] = latlon_to_local_xy(s.lat, s.lon)
            p = xy[w.src]
            e = w.start + len(w.speed_gt)
            if e < len(p):
                d = p[e] - p[e - 1]
                if np.isfinite(d).all():
                    self.nxt[i] = d

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = dict(self.base[i])
        it["gt_next"] = torch.from_numpy(self.nxt[i])
        return it


class ChainTrain(torch.utils.data.Dataset):
    """src.chain_eval.ChainTrainDataset plus gt_next."""

    def __init__(self, dataset, chains):
        self.dataset, self.chains = dataset, chains

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, i):
        items = [self.dataset[j] for j in self.chains[i]]
        out = {k: torch.stack([it[k] for it in items]) for k in ("imu", "imu_raw", "speed_gt", "pos_gt", "gt_next")}
        out["v0"], out["theta0"] = items[0]["v0"], items[0]["theta0"]
        return out


def _gt_disp(pgt, nxt):
    return pgt[:, -1] + nxt, torch.linalg.norm(pgt[:, 1:] - pgt[:, :-1], dim=-1).sum(1) + torch.linalg.norm(nxt, dim=-1)


def make_chain_loss(model, end_w=None, aux_w=0.0, set_v0=False):
    def fn(b):
        imu, raw, sgt, pgt, nxt = b["imu"], b["imu_raw"], b["speed_gt"], b["pos_gt"], b["gt_next"]
        v, th = b["v0"], b["theta0"]
        B, K = imu.shape[:2]
        ew = end_w or [1.0] * K
        p_cum, gt_cum, dist = torch.zeros(B, 2), torch.zeros(B, 2), torch.zeros(B)
        total = 0.0
        for j in range(K):
            if set_v0:
                model.cur_v0 = v
            c = model(imu[:, j])
            speed, _, pos, v, th, disp = integrate(raw[:, j], c, v, th)
            lj, _ = compute_loss(speed, pos, sgt[:, j], pgt[:, j], lib.LOSS_WEIGHTS)
            gd, dd = _gt_disp(pgt[:, j], nxt[:, j])
            p_cum, gt_cum, dist = p_cum + disp, gt_cum + gd, dist + dd
            cd = 100.0 * torch.linalg.norm(p_cum - gt_cum, dim=-1) / dist.clamp(min=1e-6)
            total = total + lj + ew[j] * cd.mean()
            if aux_w:
                total = total + aux_w * torch.nn.functional.smooth_l1_loss(model.aux, sgt[:, j])
        return total / K, {}
    return fn


@torch.no_grad()
def chained_drift(model, dataset, device, dt, window_size=50, n_chunks=6, max_chains=800,
                  speed_range=None, batch_size=128, **_):
    """Same chains as src.chain_eval.chained_drift, aligned integration, and
    the truth includes the steps between windows."""
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
        p, gt, dist = torch.zeros(B, 2), torch.zeros(B, 2), torch.zeros(B)
        for j in range(n_chunks):
            imu = torch.stack([it[j]["imu"] for it in items])
            raw = torch.stack([it[j]["imu_raw"] for it in items])
            pgt = torch.stack([it[j]["pos_gt"] for it in items])
            nxt = torch.stack([it[j]["gt_next"] for it in items])
            if hasattr(model, "cur_v0"):
                model.cur_v0 = v
            c = model(imu)
            _, _, _, v, th, disp = integrate(raw, c, v, th, dt)
            gd, dd = _gt_disp(pgt, nxt)
            p, gt, dist = p + disp, gt + gd, dist + dd
        drifts.extend((100.0 * torch.linalg.norm(p - gt, dim=-1) / dist.clamp(min=1e-6)).tolist())
    d = np.array(drifts)
    return {"n_chains": int(len(d)), "mean_drift_pct": float(d.mean()), "median_drift_pct": float(np.median(d)),
            "p90_drift_pct": float(np.percentile(d, 90)), "pass_rate_pct": float((d < 10.0).mean() * 100)}


def build_splits(calib="fwdfix", extra=False):
    sp = lib.build_splits(calib, extra=extra)
    for k in ("train", "val", "test", "test_iovnbd", "test_comma"):
        sp[k] = NextStep(sp[k], sp["_seqs"])
    return sp


def evaluate(model, splits, set_v0=False):
    return lib.evaluate(model, splits, chain_fn=chained_drift, fwd=make_fwd(set_v0))


# lib.train builds its chain dataset with this name; aligned runs need gt_next in it.
def use_aligned_chains():
    lib.ChainTrainDataset = ChainTrain
