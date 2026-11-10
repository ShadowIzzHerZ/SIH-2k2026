"""
build_windows keeps the last window that fits exactly (synthetic, no data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.io_vnbd_loader import ImuSequence
from src.data.windowing import build_windows

DT = 0.1


def _straight_drive(n: int) -> ImuSequence:
    """10 m/s due north, so every window travels well over min_distance_m."""
    t = np.arange(n) * DT
    lat = 30.0 + np.degrees(10.0 * t / 6371000.0)
    return ImuSequence(path=Path("synthetic.csv"), time=t, accel=np.zeros((n, 3)), gyro=np.zeros((n, 3)),
                       speed_gt=np.full(n, 10.0), lat=lat, lon=np.full(n, 76.0), heading_gt=np.zeros(n))


def test_no_window_that_fits_is_skipped():
    for n in range(51, 80):
        starts = [w.start for w in build_windows(_straight_drive(n), window_size=50, stride=10, dt=DT)]
        assert starts, n
        assert starts[-1] + 50 <= n, (n, starts)
        assert starts[-1] + 10 + 50 > n, (n, starts)  # the next start on the stride grid would not fit
