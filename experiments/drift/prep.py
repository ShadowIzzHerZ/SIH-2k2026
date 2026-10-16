"""Drive-level split + cached sequences for the drift experiments.

Why this exists: IO-VNBD ships every phone drive twice (Categorised/ and
Uncategorised/), and the project's file-path split put 17 of 23 test files'
twins in train/val. Here each drive is loaded ONCE (the Uncategorised copy),
drives that are parts of one session (Vw14a/b/c, S3a/b/c, ...) are grouped,
and whole groups are assigned to train/val/test. comma2k19 is split by route
(the demo's 64 one-minute segments come from 21 routes; adjacent minutes of
one route are not independent).

The split is fixed and shared by every experiment (split.json).

Run once:  python -m experiments.drift.prep
"""
from __future__ import annotations

import glob
import json
import pickle
import re
from pathlib import Path

import numpy as np

from src.data.io_vnbd_loader import load_sequence, recover_yaw_axis
from src.data.windowing import resample_uniform

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
IOVNBD = Path("data/IO-VNBD/Synchronised V abd S datasets")
COMMA = Path("data/comma2k19_demo/data")
HZ = 10.0
# Groups sorted by size, then dealt out with this repeating pattern:
# 5/7 train, 1/7 val, 1/7 test, so every split gets big and small drives.
PATTERN = ["train", "train", "test", "train", "train", "val", "train"]


def iovnbd_group(stem: str) -> str:
    """S-Vw14a / S-Vw14b -> s-vw14 ; S-S3c -> s-s3 ; S-Vfa01 -> s-vfa01."""
    s = stem.lower()
    return re.sub(r"(\d)[abc]$", r"\1", s)


def load_all():
    CACHE.mkdir(exist_ok=True)
    seqs, skipped = {}, {}
    seen = set()
    paths = sorted(glob.glob(str(IOVNBD / "**" / "S-*.csv"), recursive=True))
    # prefer the Uncategorised copy; the Categorised twin is skipped
    paths.sort(key=lambda p: (0 if "Uncategorised" in p else 1, p))
    for p in paths:
        stem = Path(p).stem
        if stem.lower() in seen:
            continue
        seen.add(stem.lower())
        try:
            seq = load_sequence(Path(p), None)
            seq = resample_uniform(seq, HZ)
            seq = recover_yaw_axis(seq)
            seqs[f"iovnbd:{stem}"] = seq
        except Exception as e:  # same skips as the main pipeline
            skipped[f"iovnbd:{stem}"] = str(e).split(": ", 1)[-1]
    print(f"IO-VNBD: {len(seen)} unique drives, {len(seqs)} usable, {len(skipped)} skipped")

    from src.data.comma2k19_loader import load_all_segments
    for seq in load_all_segments(sorted(glob.glob(str(COMMA / "*.parquet"))), target_hz=HZ):
        seqs[f"comma:{seq.path}"] = seq
    return seqs, skipped


def group_of(key: str) -> str:
    ds, name = key.split(":", 1)
    if ds == "iovnbd":
        return "iovnbd:" + iovnbd_group(name)
    return "comma:" + name.split("/")[0]


def make_split(seqs) -> dict:
    split = {}
    for ds in ("iovnbd", "comma"):
        sizes: dict[str, int] = {}
        for k, s in seqs.items():
            if k.startswith(ds + ":"):
                g = group_of(k)
                sizes[g] = sizes.get(g, 0) + s.n
        order = sorted(sizes, key=lambda g: (-sizes[g], g))
        for i, g in enumerate(order):
            split[g] = PATTERN[i % len(PATTERN)]
    return split


def main():
    seqs, skipped = load_all()
    split = make_split(seqs)
    seq_split = {k: split[group_of(k)] for k in seqs}
    with open(CACHE / "sequences.pkl", "wb") as f:
        pickle.dump(seqs, f)
    summary = {}
    for k, sp in seq_split.items():
        ds = k.split(":")[0]
        d = summary.setdefault(ds, {}).setdefault(sp, {"drives": 0, "minutes": 0.0})
        d["drives"] += 1
        d["minutes"] += round(seqs[k].n / HZ / 60, 2)
    json.dump({"group_split": split, "sequence_split": seq_split, "skipped": skipped, "summary": summary},
              open(ROOT / "split.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("test drives:", sorted(k for k, v in seq_split.items() if v == "test"))


if __name__ == "__main__":
    main()
