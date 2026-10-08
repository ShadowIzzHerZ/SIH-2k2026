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


def _kmh_header_unit_factor(speed: np.ndarray, lat: np.ndarray | None, lon: np.ndarray | None,
                            time_s: np.ndarray) -> float:
    """Multiplier that turns a column headed "GPS SPEED (Kmh)" into m/s —
    decided from the file's own data, not from the header.

    Real bug found this session, after months of it sitting in plain sight:
    IO-VNBD's phone files label this column "Kmh", but the values are
    metres per second. Checked on every file where the comparison is
    possible (108 of 144): integrating the column as m/s lands within 6% of
    the GPS lat/lon path length for the median file (p10-p90: 0.94-1.06),
    93% of files sit within 15% of exactly 1.00, and only 2% look like
    genuine km/h (ratio near 3.6). The old rule (divide by 3.6 whenever the
    header says Kmh) therefore made every IO-VNBD speed label 3.6x too
    small: the physics integrator started each window at a third of the
    true speed, speed_loss pushed the network toward a deflated target
    while drift_loss pushed it toward the real (undeflated) position track,
    and docs/understanding.md's "IO-VNBD is urban, low-speed" framing was
    partly an artifact of speeds that were 3.6x too low (its real median
    speed is ~16 m/s, not ~4).

    Compares the path length of the GPS track with the speed column's own
    integral treated as m/s: a ratio near 1 means m/s, near 3.6 means
    genuine km/h. Falls back to m/s (what 93% of the evidence supports,
    not the header's claim) when the file has too little usable GPS to
    decide."""
    default = 1.0
    if lat is None or lon is None or time_s is None or len(speed) < 50:
        return default
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(speed)
    if ok.sum() < 50:
        return default
    duration = float(time_s[-1] - time_s[0])
    if not duration > 30.0:
        return default
    xy = latlon_to_local_xy(lat[ok], lon[ok])
    path_m = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    integral_as_ms = float(np.sum(speed[ok]) * duration / len(speed))
    if path_m < 50.0 or integral_as_ms < 50.0:
        return default
    ratio = path_m / integral_as_ms
    # nearest of the two candidate units, compared in log space
    return 1.0 if abs(np.log(ratio)) <= abs(np.log(ratio / 3.6)) else 1.0 / 3.6


def _fill_held_samples(time_s: np.ndarray, arrays: list[np.ndarray],
                       circular_deg: bool = False) -> list[np.ndarray]:
    """Turn a held-then-jump (stair-step) GPS signal into a continuous one by
    linearly interpolating between the rows where a new value actually
    arrived.

    Real problem found this session: IO-VNBD's phone files log GPS at the
    10Hz sensor rate, but the GPS itself updates far less often, and the
    last fix is simply repeated on every row in between. Measured over 136
    files, in 96% of them more than a fifth of the gaps between new
    positions exceed 3 s, and the typical worst-case gap is about 9 s. The
    position label was therefore a staircase: flat for seconds, then a jump.
    Two consequences, both fixed by this function:
      - a 5 s window's end-point "truth" was often seconds stale, so the
        drift metric was scored against a position the car had already left
        (windows that caught a jump showed ~1.8x the distance the speed
        label said; windows that did not were flat and got thrown out by
        build_windows as "parked"); and
      - build_windows' note that ~47% of windows are "essentially
        stationary ... bimodal, nothing in between" was describing this
        artifact (the speed label shows those windows moving at ~7 m/s),
        not idling cars.

    Rows where a value is NaN (real GPS blackout stretches in own recordings)
    are left NaN: interpolation only happens inside each contiguous run of
    valid rows, never across a gap, so build_windows' skip-windows-with-NaN
    logic still sees real blackouts as blackouts.

    arrays share one change mask (a row counts as "new" if ANY of them
    changed), which is what you want for lat+lon that arrive together. A
    column that changes on every row comes back unchanged. circular_deg
    interpolates compass degrees through unit vectors so 358 -> 3 does not
    swing through 180. Interpolating between arrival times lags the true
    motion by the phone's GPS latency (a fraction of a second); that is a
    far smaller error than seconds of staleness."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    outs = [a.copy() for a in arrays]
    finite = np.all([np.isfinite(a) for a in arrays], axis=0)
    idx = np.flatnonzero(finite)
    if len(idx) < 3:
        return outs
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        seg_t = time_s[s:e + 1]
        changed = np.zeros(e - s + 1, dtype=bool)
        changed[0] = True
        for a in arrays:
            changed[1:] |= a[s + 1:e + 1] != a[s:e]
        if changed.all() or changed.sum() < 2:
            continue
        t_c = seg_t[changed]
        keep = np.r_[True, np.diff(t_c) > 0]  # np.interp needs strictly increasing x
        t_c = t_c[keep]
        for out, a in zip(outs, arrays):
            vals = a[s:e + 1][changed][keep]
            if circular_deg:
                r = np.radians(vals)
                c = np.interp(seg_t, t_c, np.cos(r))
                sn = np.interp(seg_t, t_c, np.sin(r))
                out[s:e + 1] = np.degrees(np.arctan2(sn, c)) % 360.0
            else:
                out[s:e + 1] = np.interp(seg_t, t_c, vals)
    return outs


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

    lat = col("lat")
    lon = col("lon")
    speed_gt = col("speed_gt")
    heading_gt = col("heading_gt")
    if speed_gt is not None and cols["speed_gt"] and "kmh" in cols["speed_gt"].lower().replace("/", ""):
        speed_gt = speed_gt * _kmh_header_unit_factor(speed_gt, lat, lon, time)

    # Held GPS samples -> continuous track (see _fill_held_samples). Position
    # fixes arrive together, so lat/lon share one change mask; speed and
    # heading are detected independently (a column that really does change
    # every row is left untouched).
    if lat is not None and lon is not None:
        lat, lon = _fill_held_samples(time, [lat, lon])
    if speed_gt is not None:
        (speed_gt,) = _fill_held_samples(time, [speed_gt])
    if heading_gt is not None:
        (heading_gt,) = _fill_held_samples(time, [heading_gt], circular_deg=True)

    return ImuSequence(
        path=path,
        time=time,
        accel=accel,
        gyro=gyro,
        speed_gt=speed_gt,
        lat=lat,
        lon=lon,
        heading_gt=heading_gt,
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


def compass_deg_to_xy_unit(bearing_deg: np.ndarray) -> np.ndarray:
    """Convert a compass bearing (0°=North, 90°=East, clockwise — GPS
    course-over-ground convention) into a unit vector in the [x=east,
    y=north] frame latlon_to_local_xy uses, so heading_gt is directly
    comparable to position-derived directions without a separate
    degrees<->math-angle conversion step at every call site."""
    rad = np.radians(bearing_deg)
    return np.stack([np.sin(rad), np.cos(rad)], axis=-1)


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
