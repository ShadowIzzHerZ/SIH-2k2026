"""
IO-VNBD raw CSV -> structured numpy arrays.

Confirmed against the real headers (see inspect output, 2026-09-03): this
dataset has two genuinely different file types living in the same folders,
distinguished by filename prefix:

  - `V-*.csv` — vehicle CAN-bus data. Has "Indicated Longitudinal/Lateral
    Acceleration" (2-axis, in g) and "Yaw Rate" (1-axis), but no full
    3-axis accelerometer/gyroscope. Not usable for this pipeline, which
    needs 6-axis IMU — these files are expected to fail to resolve and get
    skipped (see windowing.py's `file_prefix` filter, which excludes them
    up front instead of relying on the per-file skip).

  - `S-*.csv` — smartphone recordings. Has real 3-axis accelerometer +
    gyroscope + GPS, which is exactly the "phone in the car" scenario this
    PS is about. This is the file type we actually train on.

Two more real quirks confirmed from the headers:
  - Gyro axes are labeled inconsistently across sub-folders: some files
    use `GYROSCOPE X/Y/Z (rad/s)`, others use `GYROSCOPE Yaw/Pitch/Roll
    (rad/s)`. We treat Yaw=Z (yaw rate is literally what strapdown_ins.py
    uses as heading rate), Pitch=Y, Roll=X — the standard vehicle-axis
    convention, and a reasonable read of a differently-labeled but
    equivalent quantity.
  - Units aren't SI everywhere: "GPS SPEED (Kmh)" is km/h, not m/s;
    "TIME SINCE START (ms)" is milliseconds, not seconds. Both get
    converted to SI right after column resolution, so everything
    downstream (windowing.py, strapdown_ins.py) can assume SI units.

Column resolution itself:
  1. tries configs/default.yaml's `column_map` if it's been filled in, and
  2. falls back to fuzzy substring matching on normalized column names.

Everything downstream (windowing.py, calibration.py, strapdown_ins.py)
consumes the plain dict this returns, not raw DataFrames, so a schema
change here doesn't ripple through the rest of the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# substring -> canonical field, checked against normalized headers (spaces
# become underscores, not stripped — "ACCELEROMETER X (m/s²)" normalizes to
# "accelerometer_x_(m/s²)", which is what these patterns are written against)
_FUZZY_PATTERNS: dict[str, list[str]] = {
    "time": ["time_since_start", "timestamp", "time", "t(s)", "t_s"],
    "accel_x": ["accelerometer_x", "acc_x", "accel_x", "accx"],
    "accel_y": ["accelerometer_y", "acc_y", "accel_y", "accy"],
    "accel_z": ["accelerometer_z", "acc_z", "accel_z", "accz"],
    # Yaw/Pitch/Roll variant seen in some sub-folders (see module docstring)
    # checked before the bare single-letter patterns so it isn't shadowed.
    "gyro_x": ["gyroscope_roll", "gyroscope_x", "gyro_x", "gyrox"],
    "gyro_y": ["gyroscope_pitch", "gyroscope_y", "gyro_y", "gyroy"],
    "gyro_z": ["gyroscope_yaw", "gyroscope_z", "gyro_z", "gyroz", "yaw_rate"],
    "speed_gt": ["gps_speed", "wheel_speed", "wheelspeed", "obd_speed", "speed", "velocity"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng"],
    "heading_gt": ["gps_orientation", "heading", "course", "bearing"],
}


@dataclass
class ImuSequence:
    path: Path
    time: np.ndarray            # (N,) seconds
    accel: np.ndarray           # (N,3) [ax, ay, az] m/s^2, raw phone/vehicle frame
    gyro: np.ndarray            # (N,3) [gx, gy, gz] rad/s
    speed_gt: np.ndarray | None  # (N,) m/s, ground truth (wheel-speed / GPS-derived)
    lat: np.ndarray | None
    lon: np.ndarray | None
    heading_gt: np.ndarray | None

    @property
    def n(self) -> int:
        return len(self.time)


def _match_column(columns: list[str], patterns: list[str]) -> str | None:
    # underscore (not strip) so "ACCELEROMETER X (...)" -> "accelerometer_x_(...)"
    # keeps the word boundary the patterns above are written against.
    lower = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in columns}
    for pat in patterns:
        for lc, orig in lower.items():
            if pat in lc:
                return orig
    return None


def resolve_columns(df: pd.DataFrame, column_map: dict[str, str | None] | None = None) -> dict[str, str | None]:
    """Resolve canonical field -> actual column name, preferring an explicit
    config mapping and falling back to fuzzy matching."""
    resolved: dict[str, str | None] = {}
    columns = list(df.columns)
    for field, patterns in _FUZZY_PATTERNS.items():
        explicit = (column_map or {}).get(field)
        if explicit and explicit in columns:
            resolved[field] = explicit
        else:
            resolved[field] = _match_column(columns, patterns)
    return resolved


def load_sequence(path: Path, column_map: dict[str, str | None] | None = None) -> ImuSequence:
    # IO-VNBD's CSVs aren't consistently UTF-8 (some contain stray bytes from
    # degree/superscript symbols in free-text fields) — latin-1 never raises
    # a decode error since it maps every byte 0-255, and the columns we
    # actually use here are numeric, so mis-decoded text elsewhere is harmless.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    cols = resolve_columns(df, column_map)

    missing_required = [f for f in ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z") if cols[f] is None]
    if missing_required:
        raise ValueError(
            f"{path}: could not resolve required IMU columns {missing_required}. "
            f"Available columns: {list(df.columns)}. "
            f"Fill in configs/default.yaml's data.column_map with the real names "
            f"(run src/data/inspect_dataset.py to see them)."
        )

    def col(field):
        c = cols[field]
        return df[c].to_numpy(dtype=np.float64) if c else None

    time = col("time")
    if time is None:
        # no explicit timestamp column — assume uniform sampling, filled in by caller via sample_rate
        time = np.arange(len(df), dtype=np.float64)
    elif cols["time"] and "(ms)" in cols["time"].lower():
        time = time / 1000.0  # -> seconds

    accel = np.stack([col("accel_x"), col("accel_y"), col("accel_z")], axis=1)
    gyro = np.stack([col("gyro_x"), col("gyro_y"), col("gyro_z")], axis=1)

    speed_gt = col("speed_gt")
    if speed_gt is not None and cols["speed_gt"] and "kmh" in cols["speed_gt"].lower().replace("/", ""):
        speed_gt = speed_gt / 3.6  # -> m/s

    return ImuSequence(
        path=path,
        time=time,
        accel=accel,
        gyro=gyro,
        speed_gt=speed_gt,
        lat=col("lat"),
        lon=col("lon"),
        heading_gt=col("heading_gt"),
    )


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection to local metres, referenced to the first
    point in the sequence. Good enough over a single trip's extent (a few
    km at most) — not meant for anything requiring true geodesy."""
    lat0, lon0 = lat[0], lon[0]
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    x = np.radians(lon - lon0) * R * np.cos(lat0_rad)
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], axis=1)


def derive_speed_from_gps(lat: np.ndarray, lon: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Fallback ground-truth speed from consecutive GPS fixes, if the file
    has no direct wheel-speed / speed column."""
    xy = latlon_to_local_xy(lat, lon)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(time)
    dt[dt <= 0] = np.nan
    v = d / dt
    v = np.concatenate([[v[0]], v])
    return np.nan_to_num(v, nan=0.0)
