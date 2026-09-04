"""
Cross-dataset generalization check: run the IO-VNBD-trained model on
comma2k19 (a second, independently-collected real dataset — different
device, different country/roads, different driving scenario) with zero
retraining, and report the same drift metric the PS grades on.

This is not a substitute for training on comma2k19 — it answers a
different, useful question first: does the model generalize at all, or is
it overfit to IO-VNBD's specific quirks (mounting, sensor noise profile,
etc.)? See src/data/comma2k19_loader.py's module docstring for the
dataset's own caveats (demo split only, highway-only, EON device not a
literal phone).

Run (after downloading the demo split — see comma2k19_loader.py):
    python -m src.evaluate_comma2k19 --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.comma2k19_loader import load_all_segments
from src.data.windowing import IOVNBDWindowDataset, build_windows, calibrate_sequence
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import drift_metric
from src.train import forward_pass, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_dir", default="data/comma2k19_demo/data")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    dt = 1.0 / cfg["data"]["sample_rate_hz"]

    paths = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No parquet files under {args.data_dir} — download the comma2k19 demo split first "
            "(see src/data/comma2k19_loader.py's module docstring)."
        )
    seqs = load_all_segments(paths, target_hz=cfg["data"]["sample_rate_hz"])

    all_windows = []
    for seq in seqs:
        calibrate_sequence(seq)
        all_windows.extend(build_windows(
            seq, window_size=cfg["data"]["window_size"], stride=cfg["data"]["window_stride"], dt=dt,
        ))
    print(f"comma2k19: {len(seqs)} segments -> {len(all_windows)} windows")

    loader = DataLoader(IOVNBDWindowDataset(all_windows), batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"], cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"], gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"], dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    drifts = []
    with torch.no_grad():
        for batch in loader:
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            drifts.extend(drift_metric(pos_pred, pos_gt).cpu().numpy().tolist())
    drifts = np.array(drifts)

    report = {
        "n_windows": len(drifts),
        "n_segments": len(seqs),
        "mean_drift_pct": float(drifts.mean()),
        "median_drift_pct": float(np.median(drifts)),
        "p90_drift_pct": float(np.percentile(drifts, 90)),
        "worst_drift_pct": float(drifts.max()),
        "target_pct": cfg["eval"]["drift_target_pct"],
        "pass_rate_pct": float(100 * (drifts < cfg["eval"]["drift_target_pct"]).mean()),
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open("results/comma2k19_eval.json", "w"), indent=2)
    print("\nsaved to results/comma2k19_eval.json")


if __name__ == "__main__":
    main()
