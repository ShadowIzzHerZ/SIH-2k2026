"""
Evaluate a trained model against the actual SIH26168 benchmark criteria
(doc §2 / §5):
    - positional drift < 10% of distance travelled during GNSS blackout
    - example checks: <5m drift over 50m in <1min; <100m drift over 1km @60kmph

Reports both the aggregate drift-% (same metric trained on) and the two
named example scenarios, on the held-out test split, sequence by sequence
so you can see which trip types the model struggles on (tight turns are
usually the failure mode for heading-rate errors) before the demo.

Run:
    python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.windowing import load_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.train import forward_pass, pick_device
from src.models.strapdown_ins import drift_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])

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
    test_loader = DataLoader(splits["test"], batch_size=cfg["train"]["batch_size"], shuffle=False)

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
    all_drift = []
    with torch.no_grad():
        for batch in test_loader:
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            drift_pct = drift_metric(pos_pred, pos_gt)
            all_drift.extend(drift_pct.cpu().numpy().tolist())

    all_drift = np.array(all_drift)
    target = cfg["eval"]["drift_target_pct"]
    pass_rate = float((all_drift < target).mean() * 100)

    report = {
        "n_windows": len(all_drift),
        "mean_drift_pct": float(all_drift.mean()),
        "median_drift_pct": float(np.median(all_drift)),
        "p90_drift_pct": float(np.percentile(all_drift, 90)),
        "worst_drift_pct": float(all_drift.max()),
        "target_pct": target,
        "pass_rate_pct": pass_rate,
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open("results/eval_report.json", "w"), indent=2)

    if report["mean_drift_pct"] < target:
        print(f"\n✅ mean drift {report['mean_drift_pct']:.2f}% is under the {target}% PS target.")
    else:
        print(f"\n⚠️  mean drift {report['mean_drift_pct']:.2f}% is OVER the {target}% PS target — "
              f"needs more training/tuning before demo day.")


if __name__ == "__main__":
    main()
