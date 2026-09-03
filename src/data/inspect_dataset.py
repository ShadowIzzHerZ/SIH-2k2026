"""
One-off inspection script: run this after the IO-VNBD git-lfs pull finishes
to print the real column headers and folder layout, so configs/default.yaml's
column_map can be filled in correctly instead of guessed.

Usage:
    python src/data/inspect_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "IO-VNBD"


def find_sample_csvs(n: int = 6) -> list[Path]:
    candidates = sorted(DATA_ROOT.rglob("*.csv"))
    # skip anything still an LFS pointer (tiny text file)
    real = [p for p in candidates if p.stat().st_size > 5000]
    return real[:n]


def main():
    if not DATA_ROOT.exists():
        print(f"Dataset not found at {DATA_ROOT}. Clone it first (see README).")
        sys.exit(1)

    print("=== Top-level layout ===")
    for p in sorted(DATA_ROOT.iterdir()):
        print(" -", p.relative_to(DATA_ROOT))

    samples = find_sample_csvs()
    if not samples:
        print("\nNo fully-downloaded CSVs found yet — git-lfs pull may still be running.")
        sys.exit(1)

    print(f"\n=== Inspecting {len(samples)} sample CSVs ===")
    for p in samples:
        print(f"\n--- {p.relative_to(DATA_ROOT)} ---")
        try:
            df = pd.read_csv(p, nrows=5)
            print("columns:", list(df.columns))
            print(df.head(3).to_string())
        except Exception as e:
            print("failed to read:", e)


if __name__ == "__main__":
    main()
