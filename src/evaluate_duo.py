"""
Score the duo (urban model + highway model, switched by speed) against a
single model, on the same held-out test windows and the same chained 30 s
blackouts. Per-window routing uses the window's true starting speed (v0 >=
switch_mps -> highway model); chained routing uses the model's own estimated
speed at each chunk, as fusion.run_fusion does.

Run:
    python -m src.evaluate_duo --urban checkpoints/best_duo_urban.pt --highway checkpoints/best_duo_highway.pt \
        --single checkpoints/best_chainfix.pt --comma2k19_dir data/comma2k19_demo/data --run_name duo
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.chain_eval import chained_drift
from src.data.windowing import load_combined_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import drift_metric
from src.train import forward_pass, pick_device


def load_model(path, cfg, device):
    m = cfg["model"]
    net = BiasCorrectionNet(input_channels=m["input_channels"], cnn_channels=m["cnn_channels"],
                            cnn_kernel_size=m["cnn_kernel_size"], gru_hidden=m["gru_hidden"],
                            gru_layers=m["gru_layers"], dropout=m["dropout"], output_dim=m["output_dim"]).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


@torch.no_grad()
def window_drifts(model, dataset, dt, device, batch_size):
    out = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        _, pos_pred, _, pos_gt = forward_pass(model, batch, dt, device)
        out.extend(drift_metric(pos_pred, pos_gt).cpu().numpy().tolist())
    return np.array(out)


def summarize(d, target):
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "pass_pct": float((d < target).mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--urban", required=True)
    ap.add_argument("--highway", required=True)
    ap.add_argument("--single", default=None, help="single-model baseline checkpoint to compare against")
    ap.add_argument("--comma2k19_dir", default=None)
    ap.add_argument("--switch_mps", type=float, default=6.0)
    ap.add_argument("--run_name", default="duo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    dt = 1.0 / cfg["data"]["sample_rate_hz"]
    target = cfg["eval"]["drift_target_pct"]
    bs = cfg["train"]["batch_size"]
    ws = cfg["data"]["window_size"]
    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir, data_root=cfg["data"]["root"], variant=cfg["data"]["variant"],
        column_map=cfg["data"]["column_map"], sample_rate_hz=cfg["data"]["sample_rate_hz"], window_size=ws,
        window_stride=cfg["data"]["window_stride"], train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"], file_prefix=cfg["data"].get("file_prefix", ""))

    urban, highway = load_model(args.urban, cfg, device), load_model(args.highway, cfg, device)
    single = load_model(args.single, cfg, device) if args.single else None

    report = {}
    for name in ("iovnbd_test_only", "comma2k19_test_only"):
        if name not in splits:
            continue
        ds = splits[name]
        v0 = np.array([w.v0 for w in ds.windows])
        du, dh = window_drifts(urban, ds, dt, device, bs), window_drifts(highway, ds, dt, device, bs)
        duo = np.where(v0 >= args.switch_mps, dh, du)
        entry = {"window": {"duo": summarize(duo, target), "urban_only": summarize(du, target), "highway_only": summarize(dh, target)},
                 "chained_30s": {"duo": chained_drift(urban, ds, device, dt, window_size=ws, max_chains=None, model_hi=highway, switch_mps=args.switch_mps)}}
        if single is not None:
            entry["window"]["single"] = summarize(window_drifts(single, ds, dt, device, bs), target)
            entry["chained_30s"]["single"] = chained_drift(single, ds, device, dt, window_size=ws, max_chains=None)
        report[name] = entry

    print(json.dumps(report, indent=2))
    json.dump(report, open(f"results/eval_{args.run_name}.json", "w"), indent=2)
    print("\n=== summary (median drift %, share under 10%) ===")
    for name, e in report.items():
        for kind in ("window", "chained_30s"):
            for k, v in e[kind].items():
                med = v.get("median", v.get("median_drift_pct")); pas = v.get("pass_pct", v.get("pass_rate_pct"))
                print(f"{name:22s} {kind:12s} {k:13s} median {med:6.1f}%   under10 {pas:5.1f}%")


if __name__ == "__main__":
    main()
