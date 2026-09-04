"""
Loader for comma2k19 (Hugging Face `commaai/comma2k19`, "demo" split — 64
one-minute highway-driving segments, ~227MB, no auth/Kaggle account needed).
Not the primary training set (that's IO-VNBD) — this is a second, real,
independently-collected dataset (comma.ai EON device: road-facing camera +
9-axis IMU + GNSS + CAN, California Highway 280) used to check whether the
model generalizes beyond IO-VNBD, or whether more/different data helps.

Caveats vs. IO-VNBD, honestly:
  - Not literally a phone — an EON dashcam device. Similar sensor class
    (MEMS IMU), different exact mounting/hardware than a phone.
  - Highway-only, ~constant speed — none of IO-VNBD's stop-and-go turning
    variety, so weaker for the tunnel/urban-canyon blackout scenario this
    PS actually targets. Useful as a generalization check, not a full
    substitute.
  - Only the "demo" split is used here (64 segments, ~1hr total) — the
    full dataset (raw_data/Chunk_*.zip, ~97GB) is not pulled; that's a
    deliberate scope decision given the time available, not a technical
    limit of this loader.
  - No genuinely stationary window at a segment's start (unlike IO-VNBD,
    which starts from parked) — calibrate_sequence()'s gravity/yaw
    estimation runs on early *cruise* samples instead. In practice this
    still works reasonably (the demo segments are steady highway cruise,
    so average net accel is close to zero and gravity still dominates the
    mean), but it's a real difference worth knowing about if results look
    off.

Download (one-time, ~227MB — only the 3 "demo" parquet files, not the
97GB full dataset):
    from huggingface_hub import hf_hub_download
    for i in range(3):
        hf_hub_download("commaai/comma2k19", f"data/demo-0000{i}-of-00003.parquet",
                         repo_type="dataset", local_dir="data/comma2k19_demo")
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .io_vnbd_loader import ImuSequence

# WGS84 ellipsoid constants, for ECEF -> lat/lon/alt (Bowring's method).
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def ecef_to_lla(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 ECEF (metres) -> (lat_deg, lon_deg, alt_m). comma2k19's
    global_pose__frame_positions is ECEF; everything downstream in this
    pipeline (latlon_to_local_xy, map-matching) expects lat/lon degrees."""
    a, e2 = _WGS84_A, _WGS84_E2
    b = a * np.sqrt(1 - e2)
    ep2 = (a**2 - b**2) / b**2
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def load_segment(row, target_hz: float = 10.0, min_duration_s: float = 5.0) -> ImuSequence | None:
    """One parquet row = one ~1-minute segment. Returns None for a segment
    too short/sparse to be usable (a few streams occasionally drop out)."""
    log = row["log"]

    def arr(key):
        return np.asarray(log[key], dtype=np.float64)

    def arr2(key):
        return np.asarray(list(log[key]), dtype=np.float64)

    try:
        acc_t, acc_v = arr("processed_log__IMU__accelerometer__t"), arr2("processed_log__IMU__accelerometer__value")
        gyro_t, gyro_v = arr("processed_log__IMU__gyro__t"), arr2("processed_log__IMU__gyro__value")
        speed_t = arr("processed_log__CAN__speed__t")
        speed_v = arr2("processed_log__CAN__speed__value").reshape(-1)
        pos_t = arr("global_pose__frame_times")
        pos_ecef = arr2("global_pose__frame_positions")
        vel_ecef = arr2("global_pose__frame_velocities")
    except Exception:
        return None

    lens = [len(acc_t), len(gyro_t), len(speed_t), len(pos_t)]
    if min(lens) < 20 or pos_ecef.shape[0] < 20:
        return None

    lat, lon, _alt = ecef_to_lla(pos_ecef[:, 0], pos_ecef[:, 1], pos_ecef[:, 2])

    # Heading from ECEF velocity projected into a local East/North basis at
    # the segment's start (fine over a ~1min/~2km highway segment — this is
    # the same single-reference-point approximation io_vnbd_loader.py's
    # latlon_to_local_xy already makes for position).
    lat0_rad, lon0_rad = np.radians(lat[0]), np.radians(lon[0])
    e_hat = np.array([-np.sin(lon0_rad), np.cos(lon0_rad), 0.0])
    n_hat = np.array([
        -np.sin(lat0_rad) * np.cos(lon0_rad),
        -np.sin(lat0_rad) * np.sin(lon0_rad),
        np.cos(lat0_rad),
    ])
    # A handful of segments trigger benign overflow/invalid-value warnings on
    # this matmul (same false-positive class as the Apple Accelerate BLAS
    # warnings in windowing.py's calibrate_sequence) without ever actually
    # producing a non-finite result — verified by the isfinite check below,
    # which would return None if it weren't true.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        vel_east = vel_ecef @ e_hat
        vel_north = vel_ecef @ n_hat
    heading_deg = np.degrees(np.arctan2(vel_east, vel_north)) % 360.0

    t0 = max(acc_t.min(), gyro_t.min(), speed_t.min(), pos_t.min())
    t1 = min(acc_t.max(), gyro_t.max(), speed_t.max(), pos_t.max())
    if t1 - t0 < min_duration_s:
        return None

    n = int((t1 - t0) * target_hz)
    if n < 20:
        return None
    t_new = np.linspace(t0, t1, n)

    def interp_vec(t_src, v_src):
        return np.stack([np.interp(t_new, t_src, v_src[:, i]) for i in range(v_src.shape[1])], axis=1)

    accel = interp_vec(acc_t, acc_v)
    gyro = interp_vec(gyro_t, gyro_v)
    speed_gt = np.interp(t_new, speed_t, speed_v)
    lat_i = np.interp(t_new, pos_t, lat)
    lon_i = np.interp(t_new, pos_t, lon)

    # Circular-safe heading interpolation (same reasoning as windowing.py's
    # interp_heading_deg — a plain np.interp breaks at the 0/360 wrap).
    hr = np.radians(heading_deg)
    c = np.interp(t_new, pos_t, np.cos(hr))
    s = np.interp(t_new, pos_t, np.sin(hr))
    heading_i = np.degrees(np.arctan2(s, c)) % 360.0

    if not (np.isfinite(accel).all() and np.isfinite(gyro).all() and np.isfinite(speed_gt).all()
            and np.isfinite(lat_i).all() and np.isfinite(lon_i).all()):
        return None

    return ImuSequence(
        path=Path(str(row["segment_id"])),
        time=(t_new - t_new[0]).astype(np.float64),
        accel=accel.astype(np.float32),
        gyro=gyro.astype(np.float32),
        speed_gt=speed_gt.astype(np.float32),
        lat=lat_i.astype(np.float64),
        lon=lon_i.astype(np.float64),
        heading_gt=heading_i.astype(np.float32),
    )


def load_all_segments(parquet_paths: list[str], target_hz: float = 10.0) -> list[ImuSequence]:
    """Load every usable segment across the given parquet files (pass all
    3 demo-*.parquet files for the full 64-segment demo split)."""
    sequences: list[ImuSequence] = []
    skipped = 0
    for p in parquet_paths:
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            seq = load_segment(row, target_hz=target_hz)
            if seq is not None:
                sequences.append(seq)
            else:
                skipped += 1
    print(f"comma2k19: loaded {len(sequences)} usable segments, skipped {skipped}")
    return sequences
