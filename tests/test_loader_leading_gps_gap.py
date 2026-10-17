"""
A recording that starts before the first GPS fix (blank lat/lon on the first
rows, normal for data/own_recordings/) must still give windows.
latlon_to_local_xy used lat[0]/lon[0] as its origin, so a NaN first row made
the whole track NaN and build_windows dropped every window of the file:
3 of the 10 committed own recordings produced 0 windows that way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.io_vnbd_loader import ImuSequence, latlon_to_local_xy
from src.data.windowing import build_windows


def test_origin_is_first_finite_fix():
    lat = np.array([np.nan, np.nan, 31.0, 31.0001, 31.0002])
    lon = np.array([np.nan, np.nan, 75.0, 75.0, 75.0])
    xy = latlon_to_local_xy(lat, lon)
    assert np.isnan(xy[:2]).all()
    assert np.allclose(xy[2], 0.0)
    assert np.isfinite(xy[2:]).all() and xy[4, 1] > xy[3, 1] > 0


def test_unchanged_when_first_fix_is_valid():
    lat = np.array([31.0, 31.0001, 31.0002])
    lon = np.array([75.0, 75.0001, 75.0002])
    R = 6371000.0
    expected = np.stack([np.radians(lon - 75.0) * R * np.cos(np.radians(31.0)),
                         np.radians(lat - 31.0) * R], axis=1)
    assert np.allclose(latlon_to_local_xy(lat, lon), expected)


def test_build_windows_after_leading_gap():
    n, dt, gap = 400, 0.1, 30
    t = np.arange(n) * dt
    lat = 31.0 + (10.0 * t) / 6371000.0 * 180 / np.pi   # 10 m/s due north
    lon = np.full(n, 75.0)
    speed = np.full(n, 10.0)
    head = np.zeros(n)
    for a in (lat, lon, speed, head):
        a[:gap] = np.nan
    seq = ImuSequence(Path("rec.csv"), t, np.zeros((n, 3)), np.zeros((n, 3)), speed, lat, lon, head)
    windows = build_windows(seq, window_size=50, stride=10, dt=dt)
    assert windows, "every window dropped after a leading GPS gap"
    assert all(w.start >= gap for w in windows)
    assert abs(windows[0].pos_gt[-1, 1] - 49 * dt * 10.0) < 0.5
