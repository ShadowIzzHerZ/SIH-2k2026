"""
report_final.py, scored only on IO-VNBD test recordings the model never saw.

Why this exists: IO-VNBD ships the same drive in more than one folder
(e.g. "Categorised IOVNB Dataset/Vw (Driver E)/Vw08/S-Vw8.csv" and
"Uncategorised IOVNB Dataset/S-Dataset/S-Vw8.csv" hold identical samples,
only the float formatting differs). load_dataset_splits() used to split by file
path, so with the default seed 17 of the 23 test files were copies of a
train/val recording (same file name, same samples). Every IO-VNBD test
number from report_final.py / evaluate.py up to that change was partly
scored on training drives. comma2k19 is split by segment and is not affected.

load_dataset_splits() now splits by recording (windowing.split_recordings),
so a checkpoint trained after that change has no such overlap. Every
checkpoint trained before it (best.pt, best_v3chain.pt and the rest) was
trained on the old path split, so for those pass --split path: the test
split is still built the new way, and every test window whose recording
was in the OLD train/val split is dropped.

This script drops every IO-VNBD test window whose recording (file name,
case-insensitive) also appears in the train or val split, then runs the
exact same window + chained scoring as report_final.py. With the old path
split and the default seed only one distinct recording survived (Vta30,
both copies), because the other unseen test files are too short for
recover_yaw_axis() and are skipped. Treat its IO-VNBD rows as one-drive
evidence, not a full test set.

Run (checkpoints trained before the recording split):
    python -m src.report_unseen --split path --checkpoint checkpoints/best_v3chain.pt --compare checkpoints/best.pt \
        --comma2k19_dir data/comma2k19_demo/data --out results/final_report_unseen
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import yaml

from src.data.windowing import IOVNBDWindowDataset, load_combined_dataset_splits, recording_key, split_recordings
from src.evaluate_duo import load_model
from src.report_final import evaluate, print_table
from src.train import pick_device


def seen_recordings(d: dict, seed: int = 0, split: str = "recording") -> set[str]:
    """Recordings in the train+val split. split="recording" is what
    load_dataset_splits() does now; split="path" reproduces the old per-path
    shuffle exactly, for checkpoints trained before the change."""
    root = os.path.join(d["root"], d["variant"])
    paths = sorted(glob.glob(os.path.join(root, "**", f"{d.get('file_prefix', '')}*.csv"), recursive=True))
    if split == "recording":
        parts = split_recordings(paths, d["train_split"], d["val_split"], seed)
        return {recording_key(p) for p in parts["train"] + parts["val"]}
    paths = np.array(paths)
    np.random.default_rng(seed).shuffle(paths)
    n_seen = int(len(paths) * d["train_split"]) + int(len(paths) * d["val_split"])
    return {recording_key(p) for p in paths[:n_seen]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--comma2k19_dir", default=None)
    ap.add_argument("--out", default="results/final_report_unseen")
    ap.add_argument("--split", choices=["recording", "path"], default="recording",
                    help="How the checkpoints' train/val split was made: 'path' for anything trained "
                         "before the recording-level split (all checkpoints shipped so far).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    device = pick_device(cfg["train"]["device"])
    target = cfg["eval"]["drift_target_pct"]
    splits = load_combined_dataset_splits(
        comma2k19_dir=args.comma2k19_dir, data_root=d["root"], variant=d["variant"], column_map=d["column_map"],
        sample_rate_hz=d["sample_rate_hz"], window_size=d["window_size"], window_stride=d["window_stride"],
        train_split=d["train_split"], val_split=d["val_split"], file_prefix=d.get("file_prefix", ""))

    seen = seen_recordings(d, split=args.split)
    test = splits["iovnbd_test_only"]
    keep = [w for w in test.windows if recording_key(w.src) not in seen]
    unseen = sorted({recording_key(w.src) for w in keep})
    print(f"IO-VNBD test windows {len(test.windows)} -> unseen {len(keep)} from {unseen}")
    splits["iovnbd_test_only"] = IOVNBDWindowDataset(keep, test.extra_features, test.norm_mean, test.norm_std)
    splits["test"] = splits["iovnbd_test_only"]

    report = {"target_pct": target, "note": "IO-VNBD rows use only test recordings with no copy in train/val",
              "train_split_kind": args.split,
              "unseen_iovnbd_recordings": unseen, "models": {}}
    for ck in [args.checkpoint] + args.compare:
        res = evaluate(load_model(ck, cfg, device), splits, cfg, device, target)
        res["window"].pop("all test", None)
        res["chained"].pop("all test", None)
        report["models"][ck] = res
        print_table(ck, res)
    json.dump(report, open(f"{args.out}.json", "w"), indent=2)
    print(f"\nsaved {args.out}.json")


if __name__ == "__main__":
    main()
