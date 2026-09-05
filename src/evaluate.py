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
    # with comma2k19 mixed in (matches how it was trained, if --comma2k19_dir was used):
    python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt --comma2k19_dir data/comma2k19_demo/data
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.windowing import load_combined_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.train import forward_pass, pick_device
from src.models.strapdown_ins import drift_metric


def eval_drift(model, dataset, dt, device, batch_size, target_pct=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_drift = []
    with torch.no_grad():
        for batch in loader:
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            all_drift.extend(drift_metric(pos_pred, pos_gt).cpu().numpy().tolist())
    all_drift = np.array(all_drift)
    report = {
        "n_windows": len(all_drift),
        "mean_drift_pct": float(all_drift.mean()),
        "median_drift_pct": float(np.median(all_drift)),
        "p90_drift_pct": float(np.percentile(all_drift, 90)),
        "worst_drift_pct": float(all_drift.max()) if len(all_drift) else float("nan"),
    }
    if target_pct is not None:
        report["target_pct"] = target_pct
        report["pass_rate_pct"] = float((all_drift < target_pct).mean() * 100)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument(
        "--comma2k19_dir", default=None,
        help="Also break out IO-VNBD-only vs comma2k19-only test drift separately, not just "
             "the combined number — pass the same value used for src.train's --comma2k19_dir.",
    )
    parser.add_argument(
        "--extra_features", action="store_true",
        help="Must match whatever the checkpoint was trained with — see src.train's "
             "--extra_features. Rebuilds the same 12-channel normalized input.",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="Save reports as results/eval_report_<run_name>.json / "
             "eval_report_breakdown_<run_name>.json instead of the plain names. Evaluating "
             "any checkpoint other than checkpoints/best.pt without this silently "
             "overwrites the production checkpoint's own saved report with whatever "
             "checkpoint you just happened to pass — found this happen for real "
             "(eval_report.json briefly held the engineered-features run's numbers "
             "instead of best.pt's). Omitted (the default) behaves exactly as before.",
    )
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
        extra_features=args.extra_features,
    )

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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dt = 1.0 / cfg["data"]["sample_rate_hz"]
    target = cfg["eval"]["drift_target_pct"]
    batch_size = cfg["train"]["batch_size"]
    suffix = f"_{args.run_name}" if args.run_name else ""

    report = eval_drift(model, splits["test"], dt, device, batch_size, target_pct=target)

    print(json.dumps(report, indent=2))
    json.dump(report, open(f"results/eval_report{suffix}.json", "w"), indent=2)

    if report["mean_drift_pct"] < target:
        print(f"\n✅ mean drift {report['mean_drift_pct']:.2f}% is under the {target}% PS target.")
    else:
        print(f"\n⚠️  mean drift {report['mean_drift_pct']:.2f}% is OVER the {target}% PS target — "
              f"needs more training/tuning before demo day.")

    if "iovnbd_test_only" in splits and "comma2k19_test_only" in splits:
        iov = eval_drift(model, splits["iovnbd_test_only"], dt, device, batch_size, target_pct=target)
        comma = eval_drift(model, splits["comma2k19_test_only"], dt, device, batch_size, target_pct=target)
        breakdown = {"iovnbd_test_only": iov, "comma2k19_test_only": comma}
        print("\n--- breakdown by dataset (same checkpoint) ---")
        print(json.dumps(breakdown, indent=2))
        json.dump(breakdown, open(f"results/eval_report_breakdown{suffix}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
