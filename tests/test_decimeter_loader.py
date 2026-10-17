"""
decimeter_loader.load_drive_phone on a tiny synthetic drive (the real
Kaggle dataset isn't needed): a non-finite BearingDegrees must reject the
segment the same way a non-finite speed/lat/lon does, not slip through as a
NaN heading_gt (which becomes a NaN theta0 for every window built from it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.decimeter_loader import load_drive_phone


def _write_drive(tmp_path, bearing):
    n_imu, n_gt = 400, 20
    t_imu = 1_000_000 + np.arange(n_imu) * 50          # 20 Hz, 20 s
    rows = []
    for kind in ("UncalAccel", "UncalGyro"):
        for t in t_imu:
            rows.append({"MessageType": kind, "utcTimeMillis": t,
                         "MeasurementX": 0.1, "MeasurementY": 0.0, "MeasurementZ": 9.81})
    pd.DataFrame(rows).to_csv(tmp_path / "device_imu.csv", index=False)

    t_gt = 1_000_000 + np.arange(n_gt) * 1000          # 1 Hz
    pd.DataFrame({
        "MessageType": "Fix",
        "UnixTimeMillis": t_gt,
        "LatitudeDegrees": 37.0 + np.arange(n_gt) * 1e-5,
        "LongitudeDegrees": -122.0 + np.zeros(n_gt),
        "SpeedMps": np.full(n_gt, 1.1),
        "BearingDegrees": bearing,
    }).to_csv(tmp_path / "ground_truth.csv", index=False)
    return tmp_path / "device_imu.csv", tmp_path / "ground_truth.csv"


def test_clean_drive_loads(tmp_path):
    imu, gt = _write_drive(tmp_path, np.full(20, 10.0))
    seq = load_drive_phone(imu, gt)
    assert seq is not None
    assert np.isfinite(seq.heading_gt).all()


def test_nan_bearing_rejected(tmp_path):
    bearing = np.full(20, 10.0)
    bearing[7] = np.nan
    imu, gt = _write_drive(tmp_path, bearing)
    assert load_drive_phone(imu, gt) is None
