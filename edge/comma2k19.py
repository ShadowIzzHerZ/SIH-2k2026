"""
comma2k19 segment -> edge CSV at the sensor's native rate.

src/data/comma2k19_loader.py interpolates everything down to 10 Hz for
training. This keeps the raw ~100 Hz IMU timestamps instead, so the edge
engine can be checked on a real automotive MEMS IMU at a high rate, doing
its own resampling. GNSS rows come from the segment's global pose (~20 Hz)
with the CAN wheel speed and the pose velocity's course, the same sources
the training loader uses for speed_gt / heading_gt.

Needs pandas + pyarrow (already in requirements.txt).
"""
from __future__ import annotations

import numpy as np

from src.data.comma2k19_loader import ecef_to_lla

from .io import write_csv


def convert(parquet_path: str, segment: int, out_path: str) -> dict:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    row = df.iloc[segment]
    log = row["log"]

    def arr(key):
        return np.asarray(log[key], dtype=np.float64)

    def arr2(key):
        return np.asarray(list(log[key]), dtype=np.float64)

    acc_t, acc_v = arr("processed_log__IMU__accelerometer__t"), arr2("processed_log__IMU__accelerometer__value")
    gyr_t, gyr_v = arr("processed_log__IMU__gyro__t"), arr2("processed_log__IMU__gyro__value")
    spd_t, spd_v = arr("processed_log__CAN__speed__t"), arr2("processed_log__CAN__speed__value").reshape(-1)
    pos_t = arr("global_pose__frame_times")
    pos = arr2("global_pose__frame_positions")
    vel = arr2("global_pose__frame_velocities")

    # The gyro stream repeats timestamps; keep the first of each, then put
    # gyro on the accelerometer's clock so every IMU row has all six axes.
    gyr_t, first = np.unique(gyr_t, return_index=True)
    gyr_v = gyr_v[first]
    order = np.argsort(acc_t, kind="stable")
    acc_t, acc_v = acc_t[order], acc_v[order]
    t0 = max(acc_t[0], gyr_t[0], spd_t.min(), pos_t.min())
    t1 = min(acc_t[-1], gyr_t[-1], spd_t.max(), pos_t.max())
    keep = (acc_t >= t0) & (acc_t <= t1)
    acc_t, acc_v = acc_t[keep], acc_v[keep]
    gyr = np.stack([np.interp(acc_t, gyr_t, gyr_v[:, i]) for i in range(3)], axis=1)

    lat, lon, _ = ecef_to_lla(pos[:, 0], pos[:, 1], pos[:, 2])
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    e_hat = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    n_hat = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    course = np.degrees(np.arctan2(vel @ e_hat, vel @ n_hat)) % 360.0
    gk = (pos_t >= t0) & (pos_t <= t1)
    speed = np.interp(pos_t[gk], spd_t, spd_v)

    events = [(t, 0, i) for i, t in enumerate(acc_t)] + [(t, 1, i) for i, t in zip(np.where(gk)[0], pos_t[gk])]
    events.sort(key=lambda e: (e[0], e[1]))
    gidx = {j: k for k, j in enumerate(np.where(gk)[0])}
    rows = []
    for t, kind, i in events:
        if kind == 0:
            rows.append([f"{t - t0:.5f}", *(f"{a:.6f}" for a in acc_v[i]), *(f"{g:.7f}" for g in gyr[i]), "", "", "", ""])
        else:
            rows.append([f"{t - t0:.5f}", "", "", "", "", "", "", f"{lat[i]:.9f}", f"{lon[i]:.9f}",
                         f"{speed[gidx[i]]:.3f}", f"{course[i]:.3f}"])
    write_csv(out_path, ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed", "heading"], rows)
    return {
        "segment_id": str(row["segment_id"]),
        "duration_s": round(float(t1 - t0), 2),
        "imu_hz": round(float((len(acc_t) - 1) / (acc_t[-1] - acc_t[0])), 1),
        "gnss_hz": round(float((gk.sum() - 1) / (pos_t[gk][-1] - pos_t[gk][0])), 1),
        "mean_speed_mps": round(float(speed.mean()), 2),
    }
