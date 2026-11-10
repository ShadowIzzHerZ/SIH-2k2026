"""
Synthetic check that windowing's ground truth lines up with the integrator
(no data needed): a perfect prediction must score ~0% drift, per window and
chained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.chain_eval import chained_drift
from src.data.io_vnbd_loader import ImuSequence
from src.data.windowing import IOVNBDWindowDataset, build_windows
from src.models.strapdown_ins import drift_metric
from src.train import forward_pass

DT = 0.1
R_EARTH = 6371000.0


class ZeroCorrection(torch.nn.Module):
    def forward(self, imu):
        return torch.zeros(imu.shape[0], imu.shape[1], 2)


def _drive(n=400, v0=10.0, accel=0.0, yaw_rate=0.0, course0_deg=30.0):
    """Exact constant-acceleration / constant-turn drive sampled at 10 Hz,
    integrated the same way the model does (speed and heading updated
    first, then position), so a zero correction is a perfect prediction."""
    lat0, lon0 = 30.0, 76.0
    speed = np.empty(n)
    theta = np.empty(n)
    xy = np.zeros((n, 2))
    speed[0], theta[0] = v0, np.radians(90.0 - course0_deg)
    for k in range(1, n):
        speed[k] = speed[k - 1] + accel * DT
        theta[k] = theta[k - 1] + yaw_rate * DT
        xy[k] = xy[k - 1] + speed[k] * DT * np.array([np.cos(theta[k]), np.sin(theta[k])])
    lat = lat0 + np.degrees(xy[:, 1] / R_EARTH)
    lon = lon0 + np.degrees(xy[:, 0] / (R_EARTH * np.cos(np.radians(lat0))))
    imu_a = np.zeros((n, 3))
    imu_a[:, 0] = accel
    imu_g = np.zeros((n, 3))
    imu_g[:, 2] = yaw_rate
    return ImuSequence(path=Path("synthetic.csv"), time=np.arange(n) * DT, accel=imu_a, gyro=imu_g,
                       speed_gt=speed, lat=lat, lon=lon, heading_gt=(90.0 - np.degrees(theta)) % 360.0)


def _window_drift(seq):
    windows = build_windows(seq, window_size=50, stride=10, dt=DT)
    assert windows
    batch = {k: torch.stack([IOVNBDWindowDataset(windows)[i][k] for i in range(len(windows))])
             for k in ("imu", "imu_raw", "speed_gt", "pos_gt", "v0", "theta0")}
    speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(ZeroCorrection(), batch, DT, "cpu")
    return drift_metric(pos_pred, pos_gt), speed_pred - speed_gt


def test_constant_velocity_scores_zero_drift():
    drift, speed_err = _window_drift(_drive())
    assert drift.max().item() < 0.01, drift.max().item()
    assert speed_err.abs().max().item() < 1e-4


def test_accelerating_turn_scores_zero_drift():
    drift, speed_err = _window_drift(_drive(v0=5.0, accel=0.5, yaw_rate=0.1))
    assert drift.max().item() < 0.01, drift.max().item()
    assert speed_err.abs().max().item() < 1e-3


def test_chained_windows_do_not_drop_the_boundary_step():
    seq = _drive(n=400, v0=8.0, yaw_rate=0.05)
    ds = IOVNBDWindowDataset(build_windows(seq, window_size=50, stride=50, dt=DT))
    res = chained_drift(ZeroCorrection(), ds, "cpu", DT, window_size=50, n_chunks=6, max_chains=None)
    assert res["n_chains"] >= 1
    assert res["mean_drift_pct"] < 0.01, res


def test_anchor_is_the_sample_before_the_first_imu_sample():
    seq = _drive(v0=10.0, accel=0.3)
    w = build_windows(seq, window_size=50, stride=10, dt=DT)[0]
    assert w.start >= 1
    assert np.isclose(w.v0, seq.speed_gt[w.start - 1])
    assert np.allclose(w.speed_gt, seq.speed_gt[w.start:w.start + 50])
    assert np.isclose(w.lat0, seq.lat[w.start - 1]) and np.isclose(w.lon0, seq.lon[w.start - 1])
    assert np.linalg.norm(w.pos_gt[0]) > 0.5  # one 10 m/s step from the anchor, not 0
