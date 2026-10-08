"""
Loader for PPC-Dataset (taroz/PPC-Dataset on GitHub), a small real-vehicle
dataset collected in **urban Japan** (3 runs in Nagoya, 3 in Tokyo) —
pulled in specifically to help IO-VNBD's urban/low-speed drift number
(62%+ mean, the project's known weak spot — see docs/understanding.md
and diagnose_drift.py), not as a highway-flavored addition like comma2k19
or decimeter. Opt-in the same way --comma2k19_dir/--decimeter_dir are.

Download (~155MB, no registration — see the repo's README for the current
OneDrive link):
    unzip PPC-Dataset.zip -d data/PPC/

Layout (confirmed against the real files, 2026-09-26):
    <root>/<city>/run{1,2,3}/imu.csv        100Hz, real accel (m/s^2) + gyro (deg/s)
    <root>/<city>/run{1,2,3}/reference.csv  5Hz, already RTK-processed —
                                              lat/lon, roll/pitch/heading (deg),
                                              east/north/up velocity (m/s).
                                              No RTK post-processing needed:
                                              this is ready-to-use ground
                                              truth, not the raw base.nav/
                                              base.obs/rover.obs RINEX files
                                              also present in each run folder
                                              (those are for redoing your own
                                              RTK solve — unused here).

Both files share the same "GPS TOW (s)" (time of week) clock, so no
UTC/GPS-time reconciliation is needed the way decimeter_loader.py's
utcTimeMillis vs UnixTimeMillis did — just resample both onto one common
target_hz timeline using GPS TOW directly.

speed_gt is derived as the horizontal speed from East/North Velocity
(m/s) — PPC has no separate scalar speed column, but this is exactly
what a GPS chip's own speed report would give (magnitude of the
horizontal velocity vector), same quantity as IO-VNBD/comma2k19/
decimeter's speed_gt.

Heading (deg) is already GPS course-over-ground convention (0=N, 90=E,
clockwise) — see windowing.py's compass_deg_to_xy_unit docstring for why
that convention matters — so it's used as heading_gt directly, no
conversion beyond wrapping to [0, 360).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .io_vnbd_loader import ImuSequence


def load_run(imu_path: Path, ref_path: Path, target_hz: float = 10.0,
             min_duration_s: float = 5.0) -> ImuSequence | None:
    """One <city>/run<n> folder. Returns None for a run too short/sparse
    to be usable."""
    try:
        imu = pd.read_csv(imu_path, skipinitialspace=True)
        ref = pd.read_csv(ref_path, skipinitialspace=True)
    except Exception:
        return None

    required_imu = {"GPS TOW (s)", "Acc X (m/s^2)", "Acc Y (m/s^2)", "Acc Z (m/s^2)",
                     "Ang Rate X (deg/s)", "Ang Rate Y (deg/s)", "Ang Rate Z (deg/s)"}
    required_ref = {"GPS TOW (s)", "Latitude (deg)", "Longitude (deg)", "Heading (deg)",
                     "East Velocity (m/s)", "North Velocity (m/s)"}
    if not required_imu.issubset(imu.columns) or not required_ref.issubset(ref.columns):
        return None
    if len(imu) < 20 or len(ref) < 5:
        return None

    acc_t = imu["GPS TOW (s)"].to_numpy(dtype=np.float64)
    acc_v = imu[["Acc X (m/s^2)", "Acc Y (m/s^2)", "Acc Z (m/s^2)"]].to_numpy(dtype=np.float64)
    # deg/s -> rad/s, matching every other loader's gyro convention
    gyro_v = np.radians(imu[["Ang Rate X (deg/s)", "Ang Rate Y (deg/s)", "Ang Rate Z (deg/s)"]].to_numpy(dtype=np.float64))

    ref_t = ref["GPS TOW (s)"].to_numpy(dtype=np.float64)
    lat = ref["Latitude (deg)"].to_numpy(dtype=np.float64)
    lon = ref["Longitude (deg)"].to_numpy(dtype=np.float64)
    heading_gt = ref["Heading (deg)"].to_numpy(dtype=np.float64) % 360.0
    east_v = ref["East Velocity (m/s)"].to_numpy(dtype=np.float64)
    north_v = ref["North Velocity (m/s)"].to_numpy(dtype=np.float64)
    speed_gt = np.sqrt(east_v ** 2 + north_v ** 2)

    t0 = max(acc_t.min(), ref_t.min())
    t1 = min(acc_t.max(), ref_t.max())
    if t1 - t0 < min_duration_s:
        return None

    n = int((t1 - t0) * target_hz)
    if n < 20:
        return None
    t_new = np.linspace(t0, t1, n)

    def interp_vec(t_src, v_src):
        return np.stack([np.interp(t_new, t_src, v_src[:, i]) for i in range(v_src.shape[1])], axis=1)

    accel_i = interp_vec(acc_t, acc_v)
    gyro_i = interp_vec(acc_t, gyro_v)
    speed_i = np.interp(t_new, ref_t, speed_gt)
    lat_i = np.interp(t_new, ref_t, lat)
    lon_i = np.interp(t_new, ref_t, lon)

    # Circular-safe heading interpolation — same reasoning as decimeter_
    # loader.py's heading_i (a plain np.interp breaks at the 0/360 wrap).
    hr = np.radians(heading_gt)
    c = np.interp(t_new, ref_t, np.cos(hr))
    s = np.interp(t_new, ref_t, np.sin(hr))
    heading_i = np.degrees(np.arctan2(s, c)) % 360.0

    if not (np.isfinite(accel_i).all() and np.isfinite(gyro_i).all() and np.isfinite(speed_i).all()
            and np.isfinite(lat_i).all() and np.isfinite(lon_i).all()):
        return None

    return ImuSequence(
        path=imu_path.parent,  # .../<city>/run<n> — one segment, same
                                # granularity as comma2k19's per-parquet-row
                                # and decimeter's per-drive/phone segments
        time=(t_new - t_new[0]).astype(np.float64),
        accel=accel_i.astype(np.float32),
        gyro=gyro_i.astype(np.float32),
        speed_gt=speed_i.astype(np.float32),
        lat=lat_i.astype(np.float64),
        lon=lon_i.astype(np.float64),
        heading_gt=heading_i.astype(np.float32),
    )


def load_all_segments(root_dir: str, target_hz: float = 10.0) -> list[ImuSequence]:
    """root_dir: e.g. data/PPC/PPC-Dataset (the folder containing nagoya/
    and tokyo/ directly)."""
    sequences: list[ImuSequence] = []
    skipped = 0
    imu_paths = sorted(glob.glob(str(Path(root_dir) / "*" / "run*" / "imu.csv")))
    for imu_path_str in imu_paths:
        imu_path = Path(imu_path_str)
        ref_path = imu_path.parent / "reference.csv"
        if not ref_path.exists():
            skipped += 1
            continue
        seq = load_run(imu_path, ref_path, target_hz=target_hz)
        if seq is not None:
            sequences.append(seq)
        else:
            skipped += 1
    print(f"ppc: loaded {len(sequences)} usable runs, skipped {skipped}")
    return sequences
