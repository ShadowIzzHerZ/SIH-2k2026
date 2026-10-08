"""
Loader for PVS - Passive Vehicular Sensors Datasets (Kaggle, UFSC Brazil),
a real-vehicle dataset originally built for road-surface classification —
pulled in here for the same reason as ppc_loader.py: real driving data to
help IO-VNBD's urban/low-speed drift number, not another highway-flavored
addition. Speeds observed across the 9 real recordings top out around
27 m/s (~97 km/h) with plenty of low-speed/stop-and-go content — genuine
mixed urban Brazilian road conditions, not steady highway cruising.

Download (~41GB zipped — almost all of it is dashboard/environment video
we don't use; only the combined GPS+IMU CSVs, ~600MB total, are actually
needed):
    kaggle datasets download -d jefmenegazzo/pvs-passive-vehicular-sensors-datasets -p data/PVS
    # then selectively extract just the file this loader reads, per folder:
    for i in 1 2 3 4 5 6 7 8 9; do
      unzip -j data/PVS/pvs-passive-vehicular-sensors-datasets.zip \
        "PVS $i/dataset_gps_mpu_left.csv" -d "data/PVS/PVS-Dataset/PVS $i"
    done

Layout actually pulled down (confirmed against the real files, 2026-09-26):
    <root>/PVS <n>/dataset_gps_mpu_left.csv    9 folders (3 drivers x 3 routes)

One real format quirk, handled here: this file interleaves ~100Hz IMU
samples with GPS fixes that update much slower (confirmed: `timestamp_gps`
repeats identically across many consecutive IMU rows, i.e. the last known
GPS fix is forward-filled onto every IMU sample, not a fresh reading each
row). Naively differencing consecutive *rows'* lat/lon would mostly diff a
fix against itself (zero delta) — this loader dedupes to the actual unique
GPS fixes first, derives heading from *those* (no heading/bearing column
exists at all in this dataset, unlike IO-VNBD/comma2k19/decimeter/PPC),
then interpolates everything onto one common target_hz timeline the same
way decimeter_loader.py/ppc_loader.py do.

Three physical sensor placements are recorded per row (dashboard,
above_suspension, below_suspension) — dashboard is used here as the
closest real analog to a phone mounted on/near the dashboard, per this
project's own PS framing ("dashboard-mounted or mobile holder"), not the
suspension-mounted units (those exist for the dataset's original
road-quality-classification purpose, not ours).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .io_vnbd_loader import ImuSequence, latlon_to_local_xy


def load_folder(csv_path: Path, target_hz: float = 10.0, min_duration_s: float = 5.0) -> ImuSequence | None:
    """One "PVS <n>" folder's dataset_gps_mpu_left.csv. Returns None for a
    folder too short/sparse to be usable."""
    try:
        df = pd.read_csv(csv_path, usecols=[
            "timestamp", "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard",
            "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard",
            "timestamp_gps", "latitude", "longitude", "speed",
        ])
    except Exception:
        return None
    if len(df) < 20:
        return None

    acc_t = df["timestamp"].to_numpy(dtype=np.float64)
    acc_v = df[["acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard"]].to_numpy(dtype=np.float64)
    # Real bug found and fixed this session: unlike PPC's explicitly-
    # labeled "Ang Rate X (deg/s)" columns, PVS's gyro_*_dashboard headers
    # carry no unit suffix at all, and a first pass here assumed rad/s
    # (matching every other loader's *output* convention) without actually
    # checking the *input*. Confirmed via dataset_settings_left.csv's own
    # "gyroscope_full_scale: GFS_1000" (a standard MEMS ±1000 deg/s full-
    # scale setting) and a direct sanity check: raw values gave a std of
    # 4-5 "rad/s" (up to 60+ rad/s range) — physically impossible for a
    # real vehicle (a car doesn't spin at ~10 revolutions/second); treated
    # as deg/s and converted, std drops to ~0.08 rad/s with a ~-1.1..1.2
    # rad/s range, exactly the shape a real car's yaw/pitch/roll rate
    # should have. Silently feeding the uncorrected ~60x-too-large values
    # into the physics integrator would spin heading out instantly —
    # enough on its own to explain this dataset's first (terrible, ~107%
    # median) drift result before this fix.
    gyro_v = np.radians(df[["gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard"]].to_numpy(dtype=np.float64))

    # Dedup to the actual unique GPS fixes (see module doc — the raw rows
    # forward-fill the last fix onto every ~100Hz IMU sample).
    gps = df[["timestamp_gps", "latitude", "longitude", "speed"]].drop_duplicates(subset="timestamp_gps")
    gps = gps.dropna(subset=["timestamp_gps", "latitude", "longitude"])
    if len(gps) < 5:
        return None
    gps_t = gps["timestamp_gps"].to_numpy(dtype=np.float64)
    lat = gps["latitude"].to_numpy(dtype=np.float64)
    lon = gps["longitude"].to_numpy(dtype=np.float64)
    speed_gt = gps["speed"].to_numpy(dtype=np.float64)

    order = np.argsort(gps_t)
    gps_t, lat, lon, speed_gt = gps_t[order], lat[order], lon[order], speed_gt[order]

    # Heading from consecutive real fixes — this dataset has no bearing/
    # heading column at all, unlike every other loader in this project.
    xy = latlon_to_local_xy(lat, lon)
    d = np.diff(xy, axis=0)
    step_heading = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0  # (n-1,)
    # one heading value per *interval* between fixes -- assign it to the
    # earlier endpoint, repeat the last for the final fix so lengths match.
    heading_gt = np.concatenate([step_heading, step_heading[-1:]]) if len(step_heading) else np.zeros(1)

    t0 = max(acc_t.min(), gps_t.min())
    t1 = min(acc_t.max(), gps_t.max())
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
    speed_i = np.interp(t_new, gps_t, speed_gt)
    lat_i = np.interp(t_new, gps_t, lat)
    lon_i = np.interp(t_new, gps_t, lon)

    # Circular-safe heading interpolation — same reasoning as decimeter_
    # loader.py's heading_i / ppc_loader.py's heading_i.
    hr = np.radians(heading_gt)
    c = np.interp(t_new, gps_t, np.cos(hr))
    s = np.interp(t_new, gps_t, np.sin(hr))
    heading_i = np.degrees(np.arctan2(s, c)) % 360.0

    if not (np.isfinite(accel_i).all() and np.isfinite(gyro_i).all() and np.isfinite(speed_i).all()
            and np.isfinite(lat_i).all() and np.isfinite(lon_i).all()):
        return None

    return ImuSequence(
        path=csv_path.parent,  # .../PVS <n> — one segment
        time=(t_new - t_new[0]).astype(np.float64),
        accel=accel_i.astype(np.float32),
        gyro=gyro_i.astype(np.float32),
        speed_gt=speed_i.astype(np.float32),
        lat=lat_i.astype(np.float64),
        lon=lon_i.astype(np.float64),
        heading_gt=heading_i.astype(np.float32),
    )


def load_all_segments(root_dir: str, target_hz: float = 10.0) -> list[ImuSequence]:
    """root_dir: e.g. data/PVS/PVS-Dataset (the folder containing "PVS 1"
    .. "PVS 9" directly)."""
    sequences: list[ImuSequence] = []
    skipped = 0
    csv_paths = sorted(glob.glob(str(Path(root_dir) / "PVS *" / "dataset_gps_mpu_left.csv")))
    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str)
        seq = load_folder(csv_path, target_hz=target_hz)
        if seq is not None:
            sequences.append(seq)
        else:
            skipped += 1
    print(f"pvs: loaded {len(sequences)} usable folders, skipped {skipped}")
    return sequences
