"""
Loader for the Google Smartphone Decimeter Challenge (2023-2024 edition),
a Kaggle competition dataset — real Android phones (Pixel 4/4XL/5, Galaxy
S20/S21 etc.) driven around the SF Bay Area with a high-grade GNSS receiver
providing ground truth. Not the primary training set (that's IO-VNBD) — a
third, independently-collected real-phone dataset (alongside comma2k19) to
check generalization, opt-in the same way --comma2k19_dir is.

Download (~3.5GB zipped, ~3.9GB unzipped; needs a Kaggle account with the
competition rules accepted at
kaggle.com/competitions/smartphone-decimeter-2023/rules):
    kaggle competitions download -c smartphone-decimeter-2023 -p DECIMETER/
    unzip DECIMETER/smartphone-decimeter-2023.zip -d DECIMETER/sdc2023

Layout actually pulled down (confirmed against the real files, 2026-09-25):
    <root>/train/<drive_id>/<phone>/device_imu.csv
    <root>/train/<drive_id>/<phone>/device_gnss.csv     (raw GNSS, unused here)
    <root>/train/<drive_id>/<phone>/ground_truth.csv
    <root>/test/<drive_id>/<phone>/device_imu.csv       (no ground_truth.csv —
                                                          this is the Kaggle
                                                          competition's blind
                                                          submission set, so
                                                          it's unusable for
                                                          training/eval here
                                                          and never touched by
                                                          this loader)

Two real format differences from IO-VNBD/comma2k19, both handled here:
  - device_imu.csv is one interleaved stream tagged by `MessageType`
    (UncalAccel/UncalGyro/UncalMag) at each sensor's own native rate
    (~50-100Hz, not synced row-per-sample like IO-VNBD) — accel and gyro are
    pulled out separately and interpolated onto one common `target_hz`
    timeline, same approach as comma2k19_loader.load_segment.
  - ground_truth.csv is a genuine 1Hz phone-GPS-style fix track
    (LatitudeDegrees/LongitudeDegrees/SpeedMps/BearingDegrees), already SI
    (m/s, degrees, compass bearing) — no unit conversion needed, unlike
    IO-VNBD's km/h speed column.

Android's TYPE_ACCELEROMETER_UNCALIBRATED / TYPE_GYROSCOPE_UNCALIBRATED
report m/s^2 / rad/s (SI) directly, so — also unlike IO-VNBD — no unit
conversion is needed on the IMU columns either, only the MessageType split
and the resample.

One <drive_id>/<phone> folder is treated as one segment, the same
granularity comma2k19_loader.load_segment uses for one parquet row. Segments
are typically much longer than comma2k19's ~1-minute clips (a full drive,
often 15-20+ minutes) — build_windows() in windowing.py just slides its
usual 5s windows across however long the sequence turns out to be, so this
needs no special handling.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .io_vnbd_loader import ImuSequence


def load_drive_phone(imu_path: Path, gt_path: Path, target_hz: float = 10.0,
                      min_duration_s: float = 5.0) -> ImuSequence | None:
    """One <drive_id>/<phone> folder. Returns None for a segment too
    short/sparse to be usable (a handful of drives have a stream drop out
    partway through, or a ground_truth.csv with too few fixes)."""
    try:
        imu = pd.read_csv(imu_path, usecols=[
            "MessageType", "utcTimeMillis", "MeasurementX", "MeasurementY", "MeasurementZ",
        ])
        gt = pd.read_csv(gt_path, usecols=[
            "MessageType", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees",
            "SpeedMps", "BearingDegrees",
        ])
    except Exception:
        return None

    accel = imu[imu["MessageType"] == "UncalAccel"]
    gyro = imu[imu["MessageType"] == "UncalGyro"]
    gt = gt[gt["MessageType"] == "Fix"]
    if len(accel) < 20 or len(gyro) < 20 or len(gt) < 5:
        return None

    acc_t = accel["utcTimeMillis"].to_numpy(dtype=np.float64) / 1000.0
    acc_v = accel[["MeasurementX", "MeasurementY", "MeasurementZ"]].to_numpy(dtype=np.float64)
    gyro_t = gyro["utcTimeMillis"].to_numpy(dtype=np.float64) / 1000.0
    gyro_v = gyro[["MeasurementX", "MeasurementY", "MeasurementZ"]].to_numpy(dtype=np.float64)

    gt_t = gt["UnixTimeMillis"].to_numpy(dtype=np.float64) / 1000.0
    lat = gt["LatitudeDegrees"].to_numpy(dtype=np.float64)
    lon = gt["LongitudeDegrees"].to_numpy(dtype=np.float64)
    speed_gt = gt["SpeedMps"].to_numpy(dtype=np.float64)
    heading_gt = gt["BearingDegrees"].to_numpy(dtype=np.float64)

    t0 = max(acc_t.min(), gyro_t.min(), gt_t.min())
    t1 = min(acc_t.max(), gyro_t.max(), gt_t.max())
    if t1 - t0 < min_duration_s:
        return None

    n = int((t1 - t0) * target_hz)
    if n < 20:
        return None
    t_new = np.linspace(t0, t1, n)

    def interp_vec(t_src, v_src):
        return np.stack([np.interp(t_new, t_src, v_src[:, i]) for i in range(v_src.shape[1])], axis=1)

    accel_i = interp_vec(acc_t, acc_v)
    gyro_i = interp_vec(gyro_t, gyro_v)
    speed_i = np.interp(t_new, gt_t, speed_gt)
    lat_i = np.interp(t_new, gt_t, lat)
    lon_i = np.interp(t_new, gt_t, lon)

    # Circular-safe heading interpolation (same reasoning as windowing.py's
    # interp_heading_deg / comma2k19_loader's heading_i — a plain np.interp
    # breaks at the 0/360 wraparound).
    hr = np.radians(heading_gt)
    c = np.interp(t_new, gt_t, np.cos(hr))
    s = np.interp(t_new, gt_t, np.sin(hr))
    heading_i = np.degrees(np.arctan2(s, c)) % 360.0

    if not (np.isfinite(accel_i).all() and np.isfinite(gyro_i).all() and np.isfinite(speed_i).all()
            and np.isfinite(lat_i).all() and np.isfinite(lon_i).all()):
        return None

    return ImuSequence(
        path=imu_path.parent,  # .../<drive_id>/<phone> — build_combined_dataset_splits groups
                                # by path.parent.name (drive_id) to split at the drive level
        time=(t_new - t_new[0]).astype(np.float64),
        accel=accel_i.astype(np.float32),
        gyro=gyro_i.astype(np.float32),
        speed_gt=speed_i.astype(np.float32),
        lat=lat_i.astype(np.float64),
        lon=lon_i.astype(np.float64),
        heading_gt=heading_i.astype(np.float32),
    )


def load_all_segments(root_dir: str, target_hz: float = 10.0) -> list[ImuSequence]:
    """root_dir: the competition's train/ directory, e.g.
    DECIMETER/sdc2023/sdc2023/train (pass this, not test/ — test/ has no
    ground_truth.csv, so it's never usable here regardless of what's
    passed)."""
    sequences: list[ImuSequence] = []
    skipped = 0
    imu_paths = sorted(glob.glob(str(Path(root_dir) / "*" / "*" / "device_imu.csv")))
    for imu_path_str in imu_paths:
        imu_path = Path(imu_path_str)
        gt_path = imu_path.parent / "ground_truth.csv"
        if not gt_path.exists():
            skipped += 1
            continue
        seq = load_drive_phone(imu_path, gt_path, target_hz=target_hz)
        if seq is not None:
            sequences.append(seq)
        else:
            skipped += 1
    print(f"decimeter: loaded {len(sequences)} usable drive/phone segments, skipped {skipped}")
    return sequences
