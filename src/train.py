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


def compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, weights: dict):
    speed_loss = nn.functional.mse_loss(speed_pred, speed_gt)
    # drift loss: normalize final position error by distance travelled so it
    # matches the PS's own metric shape (%, not raw metres) and doesn't get
    # swamped by long high-speed windows.
    drift_pct = drift_metric(pos_pred, pos_gt)
    drift_loss = drift_pct.mean()
    total = weights["speed"] * speed_loss + weights["drift"] * drift_loss
    return total, {"speed_loss": speed_loss.item(), "drift_pct": drift_pct.mean().item()}


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
        "--run_name", default=None,
        help="Save/load under checkpoints/best_<run_name>.pt and "
             "results/train_history_<run_name>.json instead of the plain best.pt / "
             "train_history.json — use for any experimental run (different "
             "input_channels, architecture, etc.) so it can't silently overwrite the "
             "real best.pt with a checkpoint of an incompatible shape. Omitted (the "
             "default) behaves exactly as before.",
    )
    args = parser.parse_args()
    if args.extra_features:
        print("--extra_features: using 12-channel input (6 raw + 6 engineered), "
              "z-score normalized from train-split stats")

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    print(f"device: {device}")

    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir,
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

    train_loader = DataLoader(splits["train"], batch_size=cfg["train"]["batch_size"], shuffle=True)
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
        baseline = {"speed_loss": 0.0, "drift_pct": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
                _, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
                for k, v in metrics.items():
                    baseline[k] += v
        for k in baseline:
            baseline[k] /= max(1, len(val_loader))
        print(f"resumed weights baseline val drift: {baseline['drift_pct']:.2f}%")
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
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    history = []

    for i in range(cfg["train"]["epochs"]):
        epoch = epoch_offset + i
        model.train()
        t0 = time.time()
        train_metrics = {"speed_loss": 0.0, "drift_pct": 0.0}
        for batch in train_loader:
            opt.zero_grad()
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            loss, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in metrics.items():
                train_metrics[k] += v
        for k in train_metrics:
            train_metrics[k] /= max(1, len(train_loader))

        model.eval()
        val_metrics = {"speed_loss": 0.0, "drift_pct": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
                _, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
                for k, v in metrics.items():
                    val_metrics[k] += v
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        dt_epoch = time.time() - t0
        print(f"epoch {epoch:03d} | train drift {train_metrics['drift_pct']:.2f}% "
              f"| val drift {val_metrics['drift_pct']:.2f}% | lr {scheduler.get_last_lr()[0]:.2e} | {dt_epoch:.1f}s")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        json.dump(prev_history + history, open(history_path, "w"), indent=2)
        scheduler.step()

        if val_metrics["drift_pct"] < best_val_drift:
            best_val_drift = val_metrics["drift_pct"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_dir / best_name)
            print(f"  -> new best val drift {best_val_drift:.2f}%, saved checkpoints/{best_name}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    print(f"best val drift this run: {best_val_drift:.2f}% -> checkpoints/{best_name} "
          f"(previous best backed up at checkpoints/best{suffix}_prev.pt)" if args.resume else
          f"best val drift: {best_val_drift:.2f}% -> checkpoints/{best_name}")


if __name__ == "__main__":
    main()
