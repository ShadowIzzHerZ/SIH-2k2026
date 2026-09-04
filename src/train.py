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

from src.data.windowing import load_dataset_splits
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
    imu = batch["imu"].to(device)              # (B, T, 6)
    speed_gt = batch["speed_gt"].to(device)     # (B, T)
    pos_gt = batch["pos_gt"].to(device)         # (B, T, 2)
    v0 = batch["v0"].to(device)                 # (B,) true starting speed
    theta0 = batch["theta0"].to(device)         # (B,) true starting heading

    corrections = model(imu)                    # (B, T, 2) -> [delta_v, delta_theta]
    delta_v, delta_theta = corrections[..., 0], corrections[..., 1]

    forward_accel = imu[..., 0] + delta_v        # residual added to raw forward accel
    yaw_rate = imu[..., 5] + delta_theta          # residual added to raw yaw rate (gz)

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
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    print(f"device: {device}")

    splits = load_dataset_splits(
        data_root=cfg["data"]["root"],
        variant=cfg["data"]["variant"],
        column_map=cfg["data"]["column_map"],
        sample_rate_hz=cfg["data"]["sample_rate_hz"],
        window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        file_prefix=cfg["data"].get("file_prefix", ""),
    )

    train_loader = DataLoader(splits["train"], batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"],
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
    history_path = results_dir / "train_history.json"
    dt = 1.0 / cfg["data"]["sample_rate_hz"]

    epoch_offset = 0
    if args.resume:
        print(f"resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

        # Back up whatever the previous run left behind before this run
        # overwrites best.pt / train_history.json — resuming should never
        # silently destroy the checkpoint/history it started from.
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            shutil.copyfile(best_path, ckpt_dir / "best_prev.pt")
        prev_history = []
        if history_path.exists():
            prev_history = json.load(open(history_path))
            shutil.copyfile(history_path, results_dir / "train_history_prev.json")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

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
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  -> new best val drift {best_val_drift:.2f}%, saved checkpoints/best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    print(f"best val drift this run: {best_val_drift:.2f}% -> checkpoints/best.pt "
          f"(previous best backed up at checkpoints/best_prev.pt)" if args.resume else
          f"best val drift: {best_val_drift:.2f}% -> checkpoints/best.pt")


if __name__ == "__main__":
    main()
