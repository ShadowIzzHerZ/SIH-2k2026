"""
Offline unit tests for src/data/windowing.py and src/chain_eval.py's chain
building, on small synthetic sequences (no dataset).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.chain_eval import build_chains
from src.data.io_vnbd_loader import ImuSequence, latlon_to_local_xy
from src.data.windowing import (
    IOVNBDWindowDataset,
    build_windows,
    compute_channel_norm_stats,
    engineer_features,
)

DT = 0.1
M_PER_DEG_LAT = np.radians(1.0) * 6371000.0


def _drive(n: int = 200, speed: float = 10.0, bearing_deg: float = 0.0, lat0: float = 48.0) -> ImuSequence:
    """Straight drive at constant speed along a compass bearing."""
    t = np.arange(n) * DT
    north = speed * t * np.cos(np.radians(bearing_deg))
    east = speed * t * np.sin(np.radians(bearing_deg))
    lat = lat0 + north / M_PER_DEG_LAT
    lon = 11.0 + east / (M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    rng = np.random.default_rng(0)
    return ImuSequence(path=Path("drive.csv"), time=t,
                       accel=rng.normal(0, 0.1, (n, 3)) + [0, 0, 9.81], gyro=rng.normal(0, 0.01, (n, 3)),
                       speed_gt=np.full(n, speed), lat=lat, lon=lon, heading_gt=np.full(n, bearing_deg))


def test_latlon_to_local_xy_is_east_north_metres():
    seq = _drive(bearing_deg=90.0)
    xy = latlon_to_local_xy(seq.lat, seq.lon)
    assert np.allclose(xy[-1], [10.0 * (len(xy) - 1) * DT, 0.0], atol=0.05)


def test_build_windows_shapes_starts_and_relative_position():
    seq = _drive(n=200)
    ws = build_windows(seq, window_size=50, stride=10, dt=DT)
    assert len(ws) >= 15
    for w in ws:
        assert w.accel.shape == (50, 3) and w.gyro.shape == (50, 3)
        assert w.speed_gt.shape == (50,) and w.pos_gt.shape == (50, 2)
        assert w.start % 10 == 0 and w.start + 50 <= seq.n
        assert np.allclose(w.pos_gt[0], 0.0)
        assert np.allclose(w.accel, seq.accel[w.start:w.start + 50])
        assert w.lat0 == seq.lat[w.start] and w.lon0 == seq.lon[w.start]
        assert w.v0 == 10.0 and w.dt == DT
    # due north: 10 m/s for 49 steps
    assert np.allclose(ws[0].pos_gt[-1], [0.0, 49.0], atol=0.05)


@pytest.mark.parametrize("bearing_deg, theta0", [(0.0, np.pi / 2), (90.0, 0.0), (180.0, -np.pi / 2), (270.0, np.pi)])
def test_theta0_is_compass_bearing_in_east_north_math_angle(bearing_deg, theta0):
    w = build_windows(_drive(bearing_deg=bearing_deg), window_size=50, stride=50, dt=DT)[0]
    assert np.isclose(np.cos(w.theta0), np.cos(theta0), atol=1e-6)
    assert np.isclose(np.sin(w.theta0), np.sin(theta0), atol=1e-6)
    # and it agrees with the direction of the window's own position track
    d = w.pos_gt[-1]
    assert np.isclose(np.arctan2(d[1], d[0]), np.arctan2(np.sin(theta0), np.cos(theta0)), atol=1e-3)


def test_theta0_falls_back_to_track_direction_without_heading():
    seq = _drive(bearing_deg=45.0)
    seq.heading_gt = None
    w = build_windows(seq, window_size=50, stride=50, dt=DT)[0]
    assert np.isclose(w.theta0, np.pi / 4, atol=1e-3)


def test_parked_and_gps_gap_windows_are_dropped():
    seq = _drive(n=200)
    seq.lat[:60] = seq.lat[0]
    seq.lon[:60] = seq.lon[0]          # parked for the first 6 s
    seq.lat[150:160] = np.nan          # GPS gap
    ws = build_windows(seq, window_size=50, stride=10, dt=DT)
    starts = {w.start for w in ws}
    assert 0 not in starts             # < 2 m travelled
    assert not any(s <= 159 and s + 50 > 150 for s in starts)
    for w in ws:
        assert np.isfinite(w.pos_gt).all() and np.isfinite(w.speed_gt).all()


def test_engineer_features_columns():
    rng = np.random.default_rng(1)
    accel, gyro = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    f = engineer_features(accel, gyro, DT)
    assert f.shape == (50, 6) and f.dtype == np.float32
    assert np.allclose(f[:, 0], np.linalg.norm(accel, axis=1), atol=1e-5)
    assert np.allclose(f[:, 1], np.linalg.norm(accel[:, :2], axis=1), atol=1e-5)
    assert np.allclose(f[:, 2], np.linalg.norm(gyro, axis=1), atol=1e-5)
    assert np.allclose(f[:, 3], np.gradient(accel[:, 0], DT), atol=1e-4)
    assert (f[:, 4:] >= 0).all()


def test_dataset_items_and_normalisation():
    ws = build_windows(_drive(n=200), window_size=50, stride=10, dt=DT)
    mean, std = compute_channel_norm_stats(ws, extra_features=True)
    assert mean.shape == (12,) and std.shape == (12,) and (std > 0).all()

    plain = IOVNBDWindowDataset(ws)[0]
    assert plain["imu"].shape == (50, 6) and torch.equal(plain["imu"], plain["imu_raw"])
    assert plain["pos_gt"].shape == (50, 2) and plain["speed_gt"].shape == (50,)

    ext = IOVNBDWindowDataset(ws, extra_features=True, norm_mean=mean, norm_std=std)
    item = ext[0]
    assert item["imu"].shape == (50, 12)
    assert torch.equal(item["imu_raw"], plain["imu_raw"])   # physics input stays in real units
    stacked = torch.cat([ext[i]["imu"] for i in range(len(ext))]).numpy()
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-4)


def test_build_chains_uses_back_to_back_windows_only():
    seq = _drive(n=400)
    ws = build_windows(seq, window_size=50, stride=50, dt=DT)   # starts 0, 50, ..., 300
    ds = IOVNBDWindowDataset(ws)
    chains = build_chains(ds, window_size=50, n_chunks=3, chain_stride_chunks=1, max_chains=None)
    for ids in chains:
        starts = [ds.windows[i].start for i in ids]
        assert starts == [starts[0], starts[0] + 50, starts[0] + 100]
    assert len(chains) == len(ws) - 2

    # remove one window: no chain may bridge the gap
    ds_gap = IOVNBDWindowDataset([w for w in ws if w.start != 100])
    for ids in build_chains(ds_gap, window_size=50, n_chunks=3, chain_stride_chunks=1, max_chains=None):
        assert 100 not in [ds_gap.windows[i].start for i in ids]
