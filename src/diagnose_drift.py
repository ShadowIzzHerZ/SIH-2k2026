"""
Per-window drift diagnostics — find out *what* is actually driving the
60%+ mean/high worst-case (809%) test drift, instead of guessing at another
architecture/data change blind.

For every test window: predicted drift-%, distance travelled, max/mean
|yaw rate|, and starting speed. Prints correlations against drift-% and the
worst/best windows' stats, and dumps the full per-window table to
results/drift_diagnostics.csv for further slicing.

Run:
    python -m src.diagnose_drift --config configs/default.yaml --checkpoint checkpoints/best.pt
    python -m src.diagnose_drift --config configs/default.yaml --checkpoint checkpoints/best.pt --comma2k19_dir data/comma2k19_demo/data
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.windowing import load_combined_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import drift_metric
from src.train import forward_pass, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--comma2k19_dir", default=None)
    parser.add_argument("--out_csv", default="results/drift_diagnostics.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])

    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir,
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

    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"],
        cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"],
        gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"],
        dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dt = 1.0 / cfg["data"]["sample_rate_hz"]
    batch_size = cfg["train"]["batch_size"]
    loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch in loader:
            imu = batch["imu_raw"]  # (B, T, 6) — always real accel/gyro units regardless of --extra_features
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            drift_pct = drift_metric(pos_pred, pos_gt).cpu().numpy()

            step_dist = torch.linalg.norm(pos_gt[:, 1:, :] - pos_gt[:, :-1, :], dim=-1).cpu().numpy()
            distance_m = step_dist.sum(axis=1)

            gyro_z = imu[..., 5].numpy()  # (B, T) yaw rate, rad/s
            max_yaw_rate = np.abs(gyro_z).max(axis=1)
            mean_yaw_rate = np.abs(gyro_z).mean(axis=1)
            v0 = batch["v0"].numpy()
            mean_speed = speed_gt.cpu().numpy().mean(axis=1)

            for i in range(len(drift_pct)):
                rows.append({
                    "drift_pct": float(drift_pct[i]),
                    "distance_m": float(distance_m[i]),
                    "max_yaw_rate": float(max_yaw_rate[i]),
                    "mean_yaw_rate": float(mean_yaw_rate[i]),
                    "v0": float(v0[i]),
                    "mean_speed": float(mean_speed[i]),
                })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    drift = np.array([r["drift_pct"] for r in rows])
    distance = np.array([r["distance_m"] for r in rows])
    max_yaw = np.array([r["max_yaw_rate"] for r in rows])
    mean_yaw = np.array([r["mean_yaw_rate"] for r in rows])
    v0 = np.array([r["v0"] for r in rows])
    mean_speed = np.array([r["mean_speed"] for r in rows])

    print(f"n_windows: {len(rows)}")
    print(f"drift_pct: mean={drift.mean():.2f}% median={np.median(drift):.2f}% "
          f"p90={np.percentile(drift, 90):.2f}% max={drift.max():.2f}%")

    def corr(name, x):
        r = np.corrcoef(drift, x)[0, 1]
        print(f"  corr(drift_pct, {name}) = {r:.3f}")

    print("\n--- correlations with drift_pct ---")
    corr("distance_m", distance)
    corr("max_yaw_rate", max_yaw)
    corr("mean_yaw_rate", mean_yaw)
    corr("v0", v0)
    corr("mean_speed", mean_speed)

    # how much of the mean does the tail account for?
    order = np.argsort(-drift)
    top1pct_n = max(1, len(drift) // 100)
    top1pct_share = drift[order[:top1pct_n]].sum() / drift.sum()
    print(f"\ntop 1% of windows ({top1pct_n}) account for "
          f"{top1pct_share * 100:.1f}% of the summed drift_pct")

    print(f"\nmean drift EXCLUDING top 1% worst windows: "
          f"{np.delete(drift, order[:top1pct_n]).mean():.2f}%")
    print(f"mean drift EXCLUDING top 5% worst windows: "
          f"{np.delete(drift, order[:max(1, len(drift)//20)]).mean():.2f}%")

    print("\n--- worst 15 windows ---")
    for idx in order[:15]:
        r = rows[idx]
        print(f"  drift={r['drift_pct']:8.1f}%  distance={r['distance_m']:7.2f}m  "
              f"max_yaw={r['max_yaw_rate']:.2f} rad/s  v0={r['v0']:.2f} m/s")

    print("\n--- best 10 windows ---")
    for idx in order[-10:]:
        r = rows[idx]
        print(f"  drift={r['drift_pct']:8.2f}%  distance={r['distance_m']:7.2f}m  "
              f"max_yaw={r['max_yaw_rate']:.2f} rad/s  v0={r['v0']:.2f} m/s")

    print(f"\nfull per-window table saved to {args.out_csv}")


if __name__ == "__main__":
    main()
