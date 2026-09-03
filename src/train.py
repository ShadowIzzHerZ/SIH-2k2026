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

    corrections = model(imu)                    # (B, T, 2) -> [delta_v, delta_theta]
    delta_v, delta_theta = corrections[..., 0], corrections[..., 1]

    forward_accel = imu[..., 0] + delta_v        # residual added to raw forward accel
    yaw_rate = imu[..., 5] + delta_theta          # residual added to raw yaw rate (gz)

    speed_pred = integrate_speed(forward_accel, dt)
    heading_pred = integrate_heading(yaw_rate, dt)
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    dt = 1.0 / cfg["data"]["sample_rate_hz"]

    best_val_drift = float("inf")
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    history = []

    for epoch in range(cfg["train"]["epochs"]):
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
              f"| val drift {val_metrics['drift_pct']:.2f}% | {dt_epoch:.1f}s")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["drift_pct"] < best_val_drift:
            best_val_drift = val_metrics["drift_pct"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    json.dump(history, open("results/train_history.json", "w"), indent=2)
    print(f"best val drift: {best_val_drift:.2f}% -> checkpoints/best.pt")


if __name__ == "__main__":
    main()
