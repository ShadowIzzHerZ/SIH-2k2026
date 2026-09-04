"""
Sanity check for the comma2k19 loader against the real demo split (needs
the demo parquet files already downloaded — see comma2k19_loader.py's
module docstring for the one-time download command; this test is skipped,
not failed, if they're not present, same spirit as test_map_matching.py
needing network access).

Run:
    python -m tests.test_comma2k19_loader
"""
from __future__ import annotations

import glob

import numpy as np

from src.data.comma2k19_loader import load_all_segments
from src.data.windowing import build_windows, calibrate_sequence


def test_loads_and_windows_cleanly():
    paths = sorted(glob.glob("data/comma2k19_demo/data/*.parquet"))
    if not paths:
        print("SKIPPED: data/comma2k19_demo/data/*.parquet not found — "
              "download the demo split first (see comma2k19_loader.py docstring)")
        return

    seqs = load_all_segments(paths)
    assert len(seqs) > 0, "loaded zero segments — check the parquet files aren't corrupt"

    n_windows = 0
    for seq in seqs:
        assert np.isfinite(seq.accel).all() and np.isfinite(seq.gyro).all()
        assert np.isfinite(seq.lat).all() and np.isfinite(seq.lon).all()
        # Sanity range checks, not exact values — this is real recorded data:
        assert 30 < abs(seq.lat[0]) < 50, "lat looks wrong (comma2k19 is California, ~37°N)"
        assert 100 < abs(seq.lon[0]) < 130, "lon looks wrong (comma2k19 is California, ~122°W)"

        calibrate_sequence(seq)
        windows = build_windows(seq, window_size=50, stride=10, dt=0.1)
        n_windows += len(windows)

    print(f"{len(seqs)} segments -> {n_windows} windows")
    assert n_windows > 0, "zero windows built — calibration or build_windows silently dropped everything"


if __name__ == "__main__":
    test_loads_and_windows_cleanly()
    print("\n✅ comma2k19 loader test passed")
