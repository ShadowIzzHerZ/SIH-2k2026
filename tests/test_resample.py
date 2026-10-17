"""
Resampling must land on exactly 1/target_hz spacing: every integrator and
the training windows assume dt = 1/sample_rate_hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.comma2k19_loader import load_segment
from src.data.io_vnbd_loader import ImuSequence
from src.data.windowing import resample_uniform


def _seq(time: np.ndarray) -> ImuSequence:
    n = len(time)
    ramp = np.stack([time, 2 * time, 3 * time], axis=1)
    return ImuSequence(path=Path("synthetic.csv"), time=time, accel=ramp.copy(), gyro=ramp.copy(),
                       speed_gt=time.copy(), lat=None, lon=None, heading_gt=(time * 10) % 360)


@pytest.mark.parametrize("duration_s", [5.0, 10.0, 61.3])
def test_resample_uniform_keeps_exact_sample_period(duration_s):
    rng = np.random.default_rng(0)
    # irregular source timestamps, roughly 25 Hz
    t = np.cumsum(rng.uniform(0.03, 0.05, size=int(duration_s / 0.04) + 5))
    t = t[t <= t[0] + duration_s]
    out = resample_uniform(_seq(t), target_hz=10.0)

    assert np.allclose(np.diff(out.time), 0.1)
    assert out.time[0] == t[0]
    assert out.time[-1] <= t[-1] + 1e-9
    assert t[-1] - out.time[-1] < 0.1 + 1e-9          # no more than one period left unused
    assert np.allclose(out.accel[:, 1], 2 * out.time)  # linear signal interpolated exactly
    assert np.allclose(out.speed_gt, out.time)


def test_comma2k19_segment_is_resampled_at_exact_period():
    t = np.linspace(0.0, 60.0, 1201)  # 20 Hz source streams
    # ECEF of a point near (37.4 N, -122.2 E), moving slowly east
    lat, lon = np.radians(37.4), np.radians(-122.2)
    base = 6378137.0 * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    pos = base + np.outer(20.0 * t, east)
    vel = np.tile(20.0 * east, (len(t), 1))
    log = {
        "processed_log__IMU__accelerometer__t": t, "processed_log__IMU__accelerometer__value": np.zeros((len(t), 3)),
        "processed_log__IMU__gyro__t": t, "processed_log__IMU__gyro__value": np.zeros((len(t), 3)),
        "processed_log__CAN__speed__t": t, "processed_log__CAN__speed__value": np.full((len(t), 1), 20.0),
        "global_pose__frame_times": t, "global_pose__frame_positions": pos, "global_pose__frame_velocities": vel,
    }
    seq = load_segment({"log": log, "segment_id": "synthetic"}, target_hz=10.0)

    assert seq is not None
    assert np.allclose(np.diff(seq.time), 0.1, atol=1e-6)
    assert len(seq.time) == 601
    assert np.allclose(seq.heading_gt, 90.0, atol=0.1)  # due east
