"""
comma2k19 loader with repeated / unsorted timestamps (synthetic segment, no
data needed): the result must match the clean stream.
"""
from __future__ import annotations

import numpy as np

from src.data.comma2k19_loader import load_segment

_A, _E2 = 6378137.0, 6.69437999014e-3


def _lla_to_ecef(lat_deg, lon_deg, alt=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.stack([(n + alt) * np.cos(lat) * np.cos(lon), (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1 - _E2) + alt) * np.sin(lat)], axis=1)


def _segment():
    imu_t = np.arange(0.0, 30.0, 0.01)
    acc = np.stack([np.sin(imu_t), np.cos(imu_t), 9.8 + 0.1 * np.sin(3 * imu_t)], axis=1)
    gyr = np.stack([0.01 * imu_t, -0.02 * np.ones_like(imu_t), 0.05 * np.sin(imu_t)], axis=1)
    spd_t = np.arange(0.0, 30.0, 0.01)
    spd = (25.0 + np.sin(0.2 * spd_t))[:, None]
    pos_t = np.arange(0.0, 30.0, 0.05)
    lat = 37.4 + 25.0 * pos_t / 111_000.0
    pos = _lla_to_ecef(lat, -122.2 + 0 * pos_t)
    vel = np.gradient(pos, pos_t, axis=0)
    return {"acc_t": imu_t, "acc": acc, "gyr_t": imu_t.copy(), "gyr": gyr, "spd_t": spd_t, "spd": spd,
            "pos_t": pos_t, "pos": pos, "vel": vel}


def _row(s):
    return {"segment_id": "synthetic", "log": {
        "processed_log__IMU__accelerometer__t": s["acc_t"], "processed_log__IMU__accelerometer__value": s["acc"],
        "processed_log__IMU__gyro__t": s["gyr_t"], "processed_log__IMU__gyro__value": s["gyr"],
        "processed_log__CAN__speed__t": s["spd_t"], "processed_log__CAN__speed__value": s["spd"],
        "global_pose__frame_times": s["pos_t"], "global_pose__frame_positions": s["pos"],
        "global_pose__frame_velocities": s["vel"]}}


def _corrupt(s):
    """Repeat every gyro timestamp (the repeat carries a junk reading, the
    first one is kept), and shuffle the accelerometer and pose order."""
    rng = np.random.default_rng(0)
    out = dict(s)
    out["gyr_t"] = np.repeat(s["gyr_t"], 2)
    junk = np.repeat(s["gyr"], 2, axis=0)
    junk[1::2] = 5.0
    out["gyr"] = junk
    p = rng.permutation(len(s["acc_t"]))
    out["acc_t"], out["acc"] = s["acc_t"][p], s["acc"][p]
    q = rng.permutation(len(s["pos_t"]))
    out["pos_t"], out["pos"], out["vel"] = s["pos_t"][q], s["pos"][q], s["vel"][q]
    return out


def test_repeated_and_unsorted_timestamps_match_the_clean_stream():
    s = _segment()
    clean = load_segment(_row(s))
    messy = load_segment(_row(_corrupt(s)))
    assert clean is not None and messy is not None
    for f in ("accel", "gyro", "speed_gt", "lat", "lon", "heading_gt"):
        assert np.allclose(getattr(messy, f), getattr(clean, f), atol=1e-5), f
