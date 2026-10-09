"""
Train BiasCorrectionNet end-to-end against real trajectory drift.

Pipeline per window:
    1. network reads calibrated IMU window -> [delta_v, delta_theta] residuals
    2. physics baseline: forward_accel = accel[:,0], yaw_rate = gyro[:,2]
    3. corrected rates = physics + residuals
    4. integrate corrected rates (strapdown_ins.py) -> predicted speed,
       heading, and 2D trajectory for the window
    5. loss = per-step speed error + end-to-end position drift vs GT

Run:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.chain_eval import ChainTrainDataset, build_chains, chained_drift
from src.data.windowing import load_combined_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import dead_reckon_position, drift_metric, integrate_heading, integrate_speed


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_pass(model, batch, dt: float, device):
    imu = batch["imu"].to(device)               # (B, T, 6 or 12) — network input, possibly
                                                 # extended/normalized (see IOVNBDWindowDataset)
    imu_raw = batch["imu_raw"].to(device)        # (B, T, 6) — always real accel/gyro units,
                                                 # used for the physics integration below
    speed_gt = batch["speed_gt"].to(device)     # (B, T)
    pos_gt = batch["pos_gt"].to(device)         # (B, T, 2)
    v0 = batch["v0"].to(device)                 # (B,) true starting speed
    theta0 = batch["theta0"].to(device)         # (B,) true starting heading

    corrections = model(imu)                    # (B, T, 2) -> [delta_v, delta_theta]
    delta_v, delta_theta = corrections[..., 0], corrections[..., 1]

    forward_accel = imu_raw[..., 0] + delta_v    # residual added to raw forward accel
    yaw_rate = imu_raw[..., 5] + delta_theta      # residual added to raw yaw rate (gz)

    # Start from the window's real initial state, not zero — a window is a
    # random slice mid-drive, so the vehicle is essentially never stopped
    # and facing the arbitrary "heading=0" reference right at the slice
    # boundary. Integrating from zero here was the actual bug behind the
    # drift plateau; see windowing.py's Window docstring for the full story.
    speed_pred = integrate_speed(forward_accel, dt, v0=v0)
    heading_pred = integrate_heading(yaw_rate, dt, theta0=theta0)
    pos_pred = dead_reckon_position(speed_pred, heading_pred, dt)

    return speed_pred, pos_pred, speed_gt, pos_gt


def _heading_from_pos(pos: torch.Tensor) -> torch.Tensor:
    """Direction of travel (rad, 0=+x/east — same convention as
    integrate_heading) implied by each consecutive step of a position
    trajectory. Works on both pos_pred and pos_gt: for pos_pred specifically,
    this recovers heading_pred exactly wherever speed_pred > 0 (dead_reckon_
    position's own vx=speed*cos(heading)/vy=speed*sin(heading) construction
    guarantees atan2(vy,vx)=heading), so no separate heading_pred plumbing
    through forward_pass/its 5 other callers is needed."""
    d = pos[:, 1:, :] - pos[:, :-1, :]
    return torch.atan2(d[..., 1], d[..., 0])


def compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, weights: dict):
    speed_loss = nn.functional.mse_loss(speed_pred, speed_gt)
    # drift loss: normalize final position error by distance travelled so it
    # matches the PS's own metric shape (%, not raw metres) and doesn't get
    # swamped by long high-speed windows.
    drift_pct = drift_metric(pos_pred, pos_gt)
    drift_loss = drift_pct.mean()
    metrics = {"speed_loss": speed_loss.item(), "drift_pct": drift_pct.mean().item()}
    total = weights["speed"] * speed_loss + weights["drift"] * drift_loss

    # Real gap found and fixed this session: configs/default.yaml has
    # declared loss_weights.heading=0.5 since before this file's current
    # form, but nothing ever read it — drift_loss (end-to-end position
    # error) was the only signal ever telling the network its *heading*
    # predictions specifically were wrong, an indirect, compounding-prone
    # route to what should be a direct one. Ground truth here is the real
    # direction of travel between consecutive true GPS positions (pos_gt
    # always comes from real lat/lon — see windowing.py — unlike heading_gt,
    # which many IO-VNBD files don't have at all), masked to steps with
    # enough real movement to trust: a near-zero step's implied direction is
    # dominated by GPS noise, not signal, the same low-speed hazard
    # fusion.py's docstring already flags and diagnose_label_noise.py
    # measured directly this session (heading disagreement ~4x worse below
    # 3 m/s than at 6-10 m/s) — supervising heading off a noisy label in
    # exactly the regime that needs it most would fight the fix, not help it.
    heading_weight = weights.get("heading", 0.0)
    if heading_weight > 0:
        min_step_m = weights.get("heading_min_step_m", 0.15)
        step_dist = torch.linalg.norm(pos_gt[:, 1:, :] - pos_gt[:, :-1, :], dim=-1)
        valid = step_dist >= min_step_m
        if valid.any():
            heading_pred = _heading_from_pos(pos_pred)
            heading_gt = _heading_from_pos(pos_gt)
            diff = heading_pred - heading_gt
            circular_err = torch.atan2(torch.sin(diff), torch.cos(diff))  # wrap to (-pi, pi]
            heading_loss = (circular_err[valid] ** 2).mean()
        else:
            heading_loss = torch.zeros((), device=pos_pred.device, dtype=pos_pred.dtype)
        total = total + heading_weight * heading_loss
        metrics["heading_loss"] = heading_loss.item()

    # Final-heading term. The per-step term above covers every step, but the
    # failure that matters for the live app is the heading a window ENDS on:
    # fusion.py / FusionEngine.kt start each 5 s blackout chunk from the
    # previous chunk's final heading. The cleanlabels checkpoint (heading
    # loss off) put ~2 rad/s of yaw "correction" into the last 10 samples of
    # a window: the per-window drift number barely noticed (it only reads the
    # final POSITION), but chained over 30 s the estimate spun in circles
    # (~90-100% drift on all 64 comma2k19 segments vs 17% median for the old
    # checkpoint). Direction of travel over the last `heading_end_steps`
    # steps, predicted vs real, wrapped to (-pi, pi]; masked to windows that
    # really moved over that span (same noise argument as above).
    end_weight = weights.get("heading_end", 0.0)
    if end_weight > 0:
        k = int(weights.get("heading_end_steps", 10))
        d_pred = pos_pred[:, -1, :] - pos_pred[:, -1 - k, :]
        d_gt = pos_gt[:, -1, :] - pos_gt[:, -1 - k, :]
        valid = torch.linalg.norm(d_gt, dim=-1) >= weights.get("heading_end_min_m", 1.0)
        if valid.any():
            diff = torch.atan2(d_pred[..., 1], d_pred[..., 0]) - torch.atan2(d_gt[..., 1], d_gt[..., 0])
            end_loss = (torch.atan2(torch.sin(diff), torch.cos(diff))[valid] ** 2).mean()
        else:
            end_loss = torch.zeros((), device=pos_pred.device, dtype=pos_pred.dtype)
        total = total + end_weight * end_loss
        metrics["heading_end_loss"] = end_loss.item()

    return total, metrics


def chain_loss(model, batch, dt: float, device, weights: dict):
    """Chained-chunk training loss over K back-to-back windows.

    The live fusion engine starts every 5 s blackout chunk from the PREVIOUS
    chunk's own final speed/heading/position, but single-window training
    always starts from the truth, so nothing ever taught the network to
    recover from its own end-state error (the cleanlabels yaw blow-up and the
    ~35-43% chained drift of chainfix both come from that). Here chunk 0
    starts from the true state, chunk j>0 from chunk j-1's predicted
    end state WITH gradients, so an early heading error is punished by every
    later chunk it throws off. Per chunk: the usual compute_loss (speed +
    in-chunk drift + heading terms), plus `chain` x the cumulative drift %
    (error of the summed chunk displacements vs the summed true ones over the
    distance travelled so far, the same definition chain_eval.chained_drift
    and the PS metric use), averaged over chunks."""
    imu = batch["imu"].to(device)            # (B, K, T, C)
    raw = batch["imu_raw"].to(device)
    speed_gt = batch["speed_gt"].to(device)  # (B, K, T)
    pos_gt = batch["pos_gt"].to(device)      # (B, K, T, 2)
    v = batch["v0"].to(device)
    th = batch["theta0"].to(device)
    B, K = imu.shape[0], imu.shape[1]
    p_cum = torch.zeros(B, 2, device=device)
    gt_cum = torch.zeros(B, 2, device=device)
    dist = torch.zeros(B, device=device)
    chain_w = weights.get("chain", 1.0)
    total, metrics = 0.0, {}
    last_chain_drift = None
    for j in range(K):
        c = model(imu[:, j])
        acc = raw[:, j, :, 0] + c[..., 0]
        yaw = raw[:, j, :, 5] + c[..., 1]
        speed = integrate_speed(acc, dt, v0=v)
        heading = integrate_heading(yaw, dt, theta0=th)
        pos = dead_reckon_position(speed, heading, dt)
        loss_j, m = compute_loss(speed, pos, speed_gt[:, j], pos_gt[:, j], weights)
        p_cum = p_cum + pos[:, -1]
        gt_cum = gt_cum + pos_gt[:, j, -1]
        dist = dist + torch.linalg.norm(pos_gt[:, j, 1:] - pos_gt[:, j, :-1], dim=-1).sum(dim=1)
        cd = 100.0 * torch.linalg.norm(p_cum - gt_cum, dim=-1) / dist.clamp(min=1e-6)
        total = total + loss_j + chain_w * cd.mean()
        for k, val in m.items():
            metrics[k] = metrics.get(k, 0.0) + val / K
        v, th = speed[:, -1], heading[:, -1]
        last_chain_drift = cd
    metrics["chain_drift_pct"] = last_chain_drift.mean().item()   # cumulative over all K chunks
    return total / K, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--resume", default=None,
        help="Warm-start from an existing checkpoint (e.g. checkpoints/best.pt) instead "
             "of random init. The loaded weights' val drift is measured once up front so "
             "best-checkpoint tracking / early stopping stay honest about whether this run "
             "actually beats what it started from. The previous best.pt and train_history.json "
             "are backed up (*_prev) before anything gets overwritten.",
    )
    parser.add_argument(
        "--comma2k19_dir", default=None,
        help="Mix comma2k19 windows (see src/data/comma2k19_loader.py) into train/val/test "
             "alongside IO-VNBD, e.g. --comma2k19_dir data/comma2k19_demo/data. Opt-in — "
             "omitted (the default) trains on IO-VNBD only, unchanged from before.",
    )
    parser.add_argument(
        "--decimeter_dir", default=None,
        help="Mix Google Smartphone Decimeter Challenge windows (see "
             "src/data/decimeter_loader.py) into train/val/test alongside "
             "IO-VNBD (and comma2k19, if given), e.g. --decimeter_dir "
             "DECIMETER/sdc2023/sdc2023/train. Opt-in — omitted (the "
             "default) leaves training unchanged.",
    )
    parser.add_argument(
        "--ppc_dir", default=None,
        help="Mix PPC-Dataset windows (see src/data/ppc_loader.py) into "
             "train/val/test alongside IO-VNBD (and comma2k19/decimeter, if "
             "given), e.g. --ppc_dir data/PPC/PPC-Dataset. Real urban-Japan "
             "driving, added specifically to help IO-VNBD's urban/low-speed "
             "drift number (see diagnose_drift.py) — not highway-flavored "
             "like comma2k19/decimeter. Opt-in — omitted (the default) "
             "leaves training unchanged.",
    )
    parser.add_argument(
        "--pvs_dir", default=None,
        help="Mix PVS-Dataset windows (see src/data/pvs_loader.py) into "
             "train/val/test alongside IO-VNBD (and whatever else is given), "
             "e.g. --pvs_dir data/PVS/PVS-Dataset. Real mixed-speed Brazilian "
             "road driving, added for the same reason as --ppc_dir — helping "
             "the urban/low-speed number, not highway-flavored. Opt-in — "
             "omitted (the default) leaves training unchanged.",
    )
    parser.add_argument(
        "--own_recordings_dir", default=None,
        help="Mix real phone recordings (see DevRecorder / "
             "data/own_recordings/README.md) into training, e.g. "
             "--own_recordings_dir data/own_recordings. Added to *train only* "
             "(never val/test) — these are ad hoc supplementary clips, not a "
             "benchmark, unlike --comma2k19_dir. Opt-in — omitted (the "
             "default) trains on IO-VNBD (+ comma2k19 if given) only, "
             "unchanged from before.",
    )
    parser.add_argument(
        "--extra_features", action="store_true",
        help="Append windowing.py's engineer_features() 6 extra channels (accel/gyro "
             "magnitude, jerk, local smoothing+roughness) to the raw 6, z-score-normalized "
             "per-channel using train-split statistics (see IOVNBDWindowDataset). First "
             "attempt at this (no normalization, low first-cycle patience) was inconclusive "
             "— see engineer_features()'s docstring. Automatically bumps "
             "model.input_channels to 12 regardless of the config value.",
    )
    parser.add_argument(
        "--heading_weight", type=float, default=None,
        help="Override configs/default.yaml's loss_weights.heading (0.5) for this run — "
             "see compute_loss's doc for what this term does. 0.5 turned out, empirically, "
             "to hurt both the full model and the urban-regime model on a real run (both "
             "landed worse than their own no-heading-loss baselines) — it was never tuned "
             "before this session since the weight was previously declared but dead code "
             "(see compute_loss's doc), so 0.5 has no empirical basis. Pass a smaller value "
             "(e.g. 0.05) to test a lighter touch, or 0.0 to disable it entirely without "
             "editing the config. Omitted (the default) uses whatever the config says.",
    )
    parser.add_argument(
        "--heading_end_weight", type=float, default=None,
        help="Override configs/default.yaml's loss_weights.heading_end for this run (the "
             "final-heading term in compute_loss; 0 disables it).",
    )
    parser.add_argument(
        "--chain_train_chunks", type=int, default=0,
        help="Train on K back-to-back 5 s windows per sample (see chain_loss) instead of one, "
             "carrying the model's own end state from chunk to chunk. 0 = off (the old "
             "single-window objective). 3 is the setting used for the chaintrain run.",
    )
    parser.add_argument(
        "--chain_train_max", type=int, default=40000,
        help="Cap on training chains per epoch when --chain_train_chunks is on (seeded random subset).",
    )
    parser.add_argument(
        "--chain_chunks", type=int, default=6,
        help="Chained-blackout validation length in 5 s chunks (6 = a 30 s blackout, the "
             "regime fusion.py/the app actually runs; see src/chain_eval.py). Checkpoint "
             "selection uses val window drift + this chained median drift. 0 disables it "
             "and selects on window drift only, as before.",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="Save/load under checkpoints/best_<run_name>.pt and "
             "results/train_history_<run_name>.json instead of the plain best.pt / "
             "train_history.json — use for any experimental run (different "
             "input_channels, architecture, etc.) so it can't silently overwrite the "
             "real best.pt with a checkpoint of an incompatible shape. Omitted (the "
             "default) behaves exactly as before.",
    )
    parser.add_argument(
        "--speed_regime", choices=["urban", "highway"], default=None,
        help="Train a regime-specific model instead of one shared model across both — "
             "see diagnose_drift.py's confirmed finding that mixing highway-flavored data "
             "(comma2k19, decimeter) measurably hurts IO-VNBD's low-speed drift number, "
             "evidence of a real shared-capacity tradeoff in one small model, not a bug. "
             "'urban' keeps train/val windows with mean speed_gt below --urban_max_speed_mps; "
             "'highway' keeps windows at or above it. The test split is left untouched (not "
             "filtered) so this checkpoint's eval_report stays directly comparable against "
             "the existing full-test numbers already in docs/understanding.md. Omitted (the "
             "default) trains on every window, unchanged from before.",
    )
    parser.add_argument(
        "--urban_max_speed_mps", type=float, default=6.0,
        help="Threshold for --speed_regime, in m/s (a window's OWN mean speed_gt, not a "
             "per-sample cutoff). Only read when --speed_regime is set.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Real gap found and fixed this session: nothing here ever seeded torch's own "
             "RNG (model weight init, dropout, DataLoader shuffling) — only the data *split* "
             "was seeded (load_combined_dataset_splits' own separate seed=0). Every run, even "
             "with identical data/config, took a different stochastic path, so before/after "
             "comparisons across runs (this session included several) carry an unknown amount "
             "of pure random-init noise on top of whatever real effect was being measured. "
             "Fixed value now (was previously unset/nondeterministic) so two runs with the "
             "same flags are actually comparable. Change only if you deliberately want a "
             "different draw (e.g. to sanity-check how much a result varies by seed alone).",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.extra_features:
        print("--extra_features: using 12-channel input (6 raw + 6 engineered), "
              "z-score normalized from train-split stats")

    cfg = yaml.safe_load(open(args.config))
    if args.heading_weight is not None:
        cfg["train"]["loss_weights"]["heading"] = args.heading_weight
        print(f"--heading_weight: overriding loss_weights.heading -> {args.heading_weight}")
    if args.heading_end_weight is not None:
        cfg["train"]["loss_weights"]["heading_end"] = args.heading_end_weight
        print(f"--heading_end_weight: overriding loss_weights.heading_end -> {args.heading_end_weight}")
    device = pick_device(cfg["train"]["device"])
    print(f"device: {device}")

    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir,
        decimeter_dir=args.decimeter_dir,
        ppc_dir=args.ppc_dir,
        pvs_dir=args.pvs_dir,
        own_recordings_dir=args.own_recordings_dir,
        data_root=cfg["data"]["root"],
        variant=cfg["data"]["variant"],
        column_map=cfg["data"]["column_map"],
        sample_rate_hz=cfg["data"]["sample_rate_hz"],
        window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        file_prefix=cfg["data"].get("file_prefix", ""),
        extra_features=args.extra_features,
    )

    # Chained validation must score the FULL val set (a regime filter keeps
    # only some windows, which fragments chains), restricted to chains whose
    # own mean speed is in this model's regime.
    import copy
    val_full = copy.copy(splits["val"])
    chain_speed_range = None
    if args.speed_regime == "urban":
        chain_speed_range = (0.0, args.urban_max_speed_mps)
    elif args.speed_regime == "highway":
        chain_speed_range = (args.urban_max_speed_mps, float("inf"))

    if args.speed_regime:
        def keep(w):
            mean_speed = float(w.speed_gt.mean())
            return (mean_speed < args.urban_max_speed_mps) if args.speed_regime == "urban" \
                else (mean_speed >= args.urban_max_speed_mps)

        # train/val only — test stays the full, untouched split (see the
        # arg's own doc) so this checkpoint's eval_report is directly
        # comparable against every other checkpoint's numbers, not scored
        # against an easier/harder subset of its own choosing.
        for split_name in ("train", "val"):
            before = len(splits[split_name].windows)
            splits[split_name].windows = [w for w in splits[split_name].windows if keep(w)]
            after = len(splits[split_name].windows)
            print(f"[speed_regime={args.speed_regime}] {split_name}: {before} -> {after} windows "
                  f"(threshold {args.urban_max_speed_mps} m/s)")

    train_loader = DataLoader(splits["train"], batch_size=cfg["train"]["batch_size"], shuffle=True)
    if args.chain_train_chunks > 0:
        train_chains = build_chains(splits["train"], cfg["data"]["window_size"], args.chain_train_chunks,
                                    chain_stride_chunks=2, max_chains=args.chain_train_max, seed=args.seed)
        print(f"chained-chunk training: {len(train_chains)} chains x {args.chain_train_chunks} chunks "
              f"({args.chain_train_chunks * 5} s each)")
        train_loader = DataLoader(ChainTrainDataset(splits["train"], train_chains),
                                  batch_size=max(8, cfg["train"]["batch_size"] // args.chain_train_chunks), shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=cfg["train"]["batch_size"], shuffle=False)

    input_channels = 12 if args.extra_features else cfg["model"]["input_channels"]
    model = BiasCorrectionNet(
        input_channels=input_channels,
        cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"],
        gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"],
        dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    ).to(device)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = f"_{args.run_name}" if args.run_name else ""
    best_name = f"best{suffix}.pt"
    history_path = results_dir / f"train_history{suffix}.json"
    dt = 1.0 / cfg["data"]["sample_rate_hz"]

    def chain_metrics(m):
        if args.chain_chunks <= 0:
            return {}
        m.eval()
        r = chained_drift(m, val_full, device, dt, window_size=cfg["data"]["window_size"],
                          n_chunks=args.chain_chunks, speed_range=chain_speed_range)
        return {"chain_median": r["median_drift_pct"], "chain_mean": r["mean_drift_pct"],
                "chain_pass": r["pass_rate_pct"], "chain_n": r["n_chains"]} if r else {}

    def selection_score(window_drift, chain):
        # window drift alone is blind to chained heading failures (see
        # chain_eval.py); add the chained median so a checkpoint can't win
        # on one while failing the other.
        return window_drift + chain.get("chain_median", 0.0)

    epoch_offset = 0
    if args.resume:
        print(f"resuming from {args.resume}")
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
        except (RuntimeError, OSError) as e:
            # RuntimeError: most likely the checkpoint was trained with a
            # different model.input_channels (an architecture/feature
            # change) and its layer shapes no longer match. OSError: path
            # doesn't exist / unreadable. Either way, an unattended loop
            # (train_until_target.sh, the Colab loop cell) shouldn't crash
            # over it — fall back to a cold start instead.
            print(f"could not load {args.resume} ({e}); "
                  f"starting from random init instead (likely an input_channels/architecture change or missing checkpoint).")
            args.resume = None  # so the loop below knows this run is effectively a cold start

    if args.resume:
        # Back up whatever the previous run left behind before this run
        # overwrites best.pt / train_history.json — resuming should never
        # silently destroy the checkpoint/history it started from.
        best_path = ckpt_dir / best_name
        if best_path.exists():
            shutil.copyfile(best_path, ckpt_dir / f"best{suffix}_prev.pt")
        prev_history = []
        if history_path.exists():
            prev_history = json.load(open(history_path))
            shutil.copyfile(history_path, results_dir / f"train_history{suffix}_prev.json")
            epoch_offset = (prev_history[-1]["epoch"] + 1) if prev_history else 0

        # Measure the loaded weights' actual val drift before training so
        # "best" tracking below is honest about whether this run improves on
        # what it started from, instead of resetting to inf and overwriting
        # best.pt with something worse the moment val drift dips even once.
        model.eval()
        # Not hardcoded to {"speed_loss", "drift_pct"} — compute_loss can
        # return extra keys (e.g. "heading_loss", only when loss_weights.
        # heading > 0, see its own doc) that a fixed-key dict would KeyError
        # on the first accumulation. .get(k, 0.0) makes this robust to
        # whatever metrics compute_loss actually returns.
        baseline: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
                _, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
                for k, v in metrics.items():
                    baseline[k] = baseline.get(k, 0.0) + v
        for k in baseline:
            baseline[k] /= max(1, len(val_loader))
        base_chain = chain_metrics(model)
        baseline.update(base_chain)
        print(f"resumed weights baseline val drift: {baseline['drift_pct']:.2f}%"
              + (f" | chained {args.chain_chunks * 5}s median {base_chain['chain_median']:.2f}% "
                 f"(n={base_chain['chain_n']})" if base_chain else ""))
    else:
        prev_history = []
        baseline = None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    # Flat LR the whole run was a real gap — a run showed steadily shrinking
    # per-epoch improvement (-1.4% -> -0.6% -> -0.4% -> -0.2%...) consistent
    # with the fixed step size overshooting near a minimum rather than the
    # model having genuinely stopped learning. Cosine decay lets it keep
    # taking finer steps as training progresses instead of asking one LR to
    # work well for both the beginning and the end of the run. On a resume,
    # this restarts the cosine cycle from the configured peak LR (a "warm
    # restart") rather than continuing the decayed tail of the previous run
    # — deliberately, since a fully-decayed LR has nowhere left to explore.
    #
    # Bug found after 7 identical-result warm-restart cycles in a row
    # (train_until_target_comma.sh, 2026-09-05): T_max was always
    # cfg["train"]["epochs"] (60), but early_stop_patience=8 was cutting
    # every *resumed* cycle off after only 8-16 real epochs — nowhere near
    # 60 — so LR barely moved off its 1.0e-3 peak (e.g. 1.00e-03 -> 9.67e-04
    # over 8 epochs) before the cycle ended. Every restart was therefore
    # retracing almost the same high-LR trajectory for a similar short
    # duration and landing in the same place — that's *why* the cycles kept
    # matching, not evidence the model was maxed out.
    #
    # First attempt used T_max = patience*3 (24), reasoning it'd decay
    # nicely *if* a cycle ran long. It didn't help: a resumed cycle that
    # never finds a new best always stops at exactly `patience` epochs
    # (no improvement to reset the counter), so the guaranteed worst-case
    # window is `early_stop_patience` epochs, not something a longer T_max
    # can lean on. Sized to the guaranteed window instead, so LR reaches a
    # genuinely low, fine-tuning-scale value even in that worst case
    # (T_max=10, patience=8 -> LR ~2e-4 by the 8th epoch instead of ~9.7e-4).
    # Cold start keeps the full-length cycle (unchanged; that run already
    # reached the 63.26% best cleanly over 57 real epochs, finding enough
    # new bests along the way to actually use a long cycle).
    cycle_len = cfg["train"]["epochs"] if not args.resume else max(cfg["train"]["early_stop_patience"] + 2, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cycle_len)

    # Seed "best" from the loaded checkpoint's real val drift (not inf) so
    # this run only overwrites best.pt when it actually beats what it
    # started from.
    best_val_drift = baseline["drift_pct"] if baseline is not None else float("inf")
    best_score = selection_score(baseline["drift_pct"], baseline) if baseline is not None else float("inf")
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    history = []

    for i in range(cfg["train"]["epochs"]):
        epoch = epoch_offset + i
        model.train()
        t0 = time.time()
        # Not hardcoded — see baseline's identical doc above (compute_loss
        # can return extra keys like "heading_loss" that a fixed-key dict
        # would KeyError on).
        train_metrics: dict[str, float] = {}
        for batch in train_loader:
            opt.zero_grad()
            if args.chain_train_chunks > 0:
                loss, metrics = chain_loss(model, batch, dt, device, cfg["train"]["loss_weights"])
            else:
                speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
                loss, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0.0) + v
        for k in train_metrics:
            train_metrics[k] /= max(1, len(train_loader))

        model.eval()
        val_metrics: dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
                _, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        val_chain = chain_metrics(model)
        val_metrics.update(val_chain)
        score = selection_score(val_metrics["drift_pct"], val_chain)
        dt_epoch = time.time() - t0
        extra = ""
        if "heading_end_loss" in val_metrics:
            extra += f" | end-heading loss {val_metrics['heading_end_loss']:.3f}"
        if val_chain:
            extra += f" | chained {args.chain_chunks * 5}s median {val_chain['chain_median']:.2f}% pass {val_chain['chain_pass']:.0f}%"
        print(f"epoch {epoch:03d} | train drift {train_metrics['drift_pct']:.2f}% "
              f"| val drift {val_metrics['drift_pct']:.2f}%{extra} | score {score:.2f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e} | {dt_epoch:.1f}s")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score})
        json.dump(prev_history + history, open(history_path, "w"), indent=2)
        scheduler.step()

        if score < best_score:
            best_score = score
            best_val_drift = val_metrics["drift_pct"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_dir / best_name)
            print(f"  -> new best score {best_score:.2f} (val drift {best_val_drift:.2f}%), saved checkpoints/{best_name}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no score improvement for {patience} epochs)")
                break

    print(f"best val drift this run: {best_val_drift:.2f}% -> checkpoints/{best_name} "
          f"(previous best backed up at checkpoints/best{suffix}_prev.pt)" if args.resume else
          f"best val drift: {best_val_drift:.2f}% -> checkpoints/{best_name}")


if __name__ == "__main__":
    main()
