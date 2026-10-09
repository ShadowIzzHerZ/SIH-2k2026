"""
One honest results table for a checkpoint, on the held-out TEST split only.

Everything is labelled by what it actually is:
  - dataset:   IO-VNBD (mixed urban/suburban/highway UK/FR/NG phone drives) and
               comma2k19 (US highway). Not "urban" and "highway".
  - regime:    windows split by their own mean ground-truth speed, below 6 m/s
               ("slow") or at/above ("fast"). Chains by the chain's mean speed.
  - metric:    "window"  = one 5 s window, started from the true heading/speed
                          (what train.py optimises, no chaining).
               "chained" = 10/30/60 s blackouts built from back-to-back 5 s
                          chunks, each started from the model's OWN previous
                          end state (what the app does), same drift-% definition.
  - drift %:   final position error / true distance travelled; the PS target is < 10%.

Run:
    python -m src.report_final --checkpoint checkpoints/best_v3.pt --compare checkpoints/best.pt checkpoints/best_chainfix.pt \
        --comma2k19_dir data/comma2k19_demo/data --out results/final_report_v3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from src.chain_eval import chained_drift
from src.data.windowing import load_combined_dataset_splits
from src.evaluate_duo import load_model, window_drifts
from src.train import pick_device

SLOW_FAST_MPS = 6.0


def stats(d, target):
    d = np.asarray(d)
    if len(d) == 0:
        return None
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)), "under_target_pct": float((d < target).mean() * 100)}


def evaluate(model, splits, cfg, device, target):
    dt = 1.0 / cfg["data"]["sample_rate_hz"]
    ws = cfg["data"]["window_size"]
    bs = cfg["train"]["batch_size"]
    out = {"window": {}, "chained": {}}
    for name, label in (("test", "all test"), ("iovnbd_test_only", "IO-VNBD"), ("comma2k19_test_only", "comma2k19")):
        if name not in splits:
            continue
        ds = splits[name]
        d = window_drifts(model, ds, dt, device, bs)
        ms = np.array([w.speed_gt.mean() for w in ds.windows])
        out["window"][label] = {"all speeds": stats(d, target), f"slow (<{SLOW_FAST_MPS:g} m/s)": stats(d[ms < SLOW_FAST_MPS], target),
                                f"fast (>={SLOW_FAST_MPS:g} m/s)": stats(d[ms >= SLOW_FAST_MPS], target)}
        for n_chunks in (2, 6, 12):
            key = f"{n_chunks * 5} s"
            row = {}
            for rname, rng in (("all speeds", None), (f"slow (<{SLOW_FAST_MPS:g} m/s)", (0.0, SLOW_FAST_MPS)),
                               (f"fast (>={SLOW_FAST_MPS:g} m/s)", (SLOW_FAST_MPS, 1e9))):
                r = chained_drift(model, ds, device, dt, window_size=ws, n_chunks=n_chunks, max_chains=None, speed_range=rng)
                row[rname] = ({"n": r["n_chains"], "mean": r["mean_drift_pct"], "median": r["median_drift_pct"], "p90": r["p90_drift_pct"],
                               "under_target_pct": r["pass_rate_pct"]} if r else None)
            out["chained"].setdefault(label, {})[key] = row
    return out


def fmt(s):
    return "      n/a      " if s is None else f"{s['median']:5.1f} / {s['mean']:6.1f} / {s['under_target_pct']:4.1f}%"


def print_table(name, res):
    print(f"\n=== {name} ===  (median % / mean % / share under 10%)  n in brackets")
    print(f"{'WINDOW (5 s, no chaining)':32s} {'all speeds':>28s} {'slow <6 m/s':>28s} {'fast >=6 m/s':>28s}")
    for label, row in res["window"].items():
        cells = list(row.values())
        print(f"{label:32s} " + " ".join(f"{fmt(c):>22s}[{c['n'] if c else 0:>6d}]" for c in cells))
    for dur in ("10 s", "30 s", "60 s"):
        print(f"{'CHAINED ' + dur + ' blackout':32s} {'all speeds':>28s} {'slow <6 m/s':>28s} {'fast >=6 m/s':>28s}")
        for label, rows in res["chained"].items():
            cells = list(rows[dur].values())
            print(f"{label:32s} " + " ".join(f"{fmt(c):>22s}[{c['n'] if c else 0:>6d}]" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="other checkpoints scored on the SAME test set")
    ap.add_argument("--comma2k19_dir", default=None)
    ap.add_argument("--out", default="results/final_report")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    target = cfg["eval"]["drift_target_pct"]
    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir, data_root=cfg["data"]["root"], variant=cfg["data"]["variant"],
        column_map=cfg["data"]["column_map"], sample_rate_hz=cfg["data"]["sample_rate_hz"], window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"], train_split=cfg["data"]["train_split"], val_split=cfg["data"]["val_split"],
        file_prefix=cfg["data"].get("file_prefix", ""))

    report = {"target_pct": target, "test_windows": len(splits["test"]), "models": {}}
    for ck in [args.checkpoint] + args.compare:
        res = evaluate(load_model(ck, cfg, device), splits, cfg, device, target)
        report["models"][ck] = res
        print_table(ck, res)
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    print(f"\nsaved {args.out}.json")


if __name__ == "__main__":
    main()
