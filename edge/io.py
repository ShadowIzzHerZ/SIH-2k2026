"""
The edge engine's file format, and the small coordinate helpers it needs.

One CSV, one row per sensor event, sorted by time:

    t,ax,ay,az,gx,gy,gz,lat,lon,speed,heading

  t            seconds (any epoch; only differences matter). ms/us/ns stamps
               need --time-unit; load_csv raises if the IMU rate they
               imply is implausible.
  ax..gz       raw IMU, sensor frame. Units set on the command line
               (--accel-unit m/s2|g, --gyro-unit rad/s|deg/s).
  lat, lon     GNSS fix in degrees. Or give x, y in local metres instead.
  speed        GNSS Doppler speed, m/s (optional but strongly recommended,
               see src/fusion.py's run_fusion docstring for why).
  heading      GNSS course over ground, compass degrees (0 = north,
               clockwise; optional, same reason).

A row can carry IMU only, GNSS only, or both; leave the other columns empty.
A 200 Hz IMU with a 10 Hz receiver is 200 IMU rows and 10 GNSS rows a second.
No GNSS rows for a while is a blackout; nothing else needs marking.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

EARTH_R = 6371000.0
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
# Plausible IMU rates once t is in seconds. The model runs at 10 Hz, so a
# slower "IMU" can't feed it; phone and automotive IMUs top out at a few kHz.
IMU_HZ_RANGE = (5.0, 10000.0)


@dataclass
class EdgeRecording:
    t_imu: np.ndarray        # (N,)
    accel: np.ndarray        # (N, 3) sensor frame, SI after unit conversion
    gyro: np.ndarray         # (N, 3)
    t_gnss: np.ndarray       # (M,)
    gnss_xy: np.ndarray      # (M, 2) local metres, x east / y north
    gnss_speed: np.ndarray   # (M,) NaN where absent
    gnss_course: np.ndarray  # (M,) compass degrees, NaN where absent
    origin: tuple[float, float] | None  # (lat0, lon0) if the file used lat/lon


def latlon_to_xy(lat, lon, lat0, lon0):
    """Equirectangular projection, same as io_vnbd_loader.latlon_to_local_xy."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return np.stack([x, y], axis=-1)


def xy_to_latlon(xy, lat0, lon0):
    lat = lat0 + np.degrees(xy[..., 1] / EARTH_R)
    lon = lon0 + np.degrees(xy[..., 0] / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


def compass_to_math(deg):
    """Compass degrees (0 = north, clockwise) -> radians CCW from east."""
    return np.arctan2(np.cos(np.radians(deg)), np.sin(np.radians(deg)))


def math_to_compass(rad):
    return (90.0 - np.degrees(rad)) % 360.0


def _col(rows, header, name):
    if name not in header:
        return None
    i = header.index(name)
    return np.array([float(r[i]) if i < len(r) and r[i].strip() not in ("", "nan", "NaN") else np.nan for r in rows])


def check_imu_rate(t_imu, path: str = "input") -> None:
    """Raise if the IMU rate implied by t_imu (seconds) is implausible, which
    almost always means t is in ms, us or ns. Uses the median step, so GNSS
    gaps and dropped samples don't move it."""
    dt = np.diff(np.asarray(t_imu, dtype=float))
    dt = dt[dt > 0]
    if not len(dt):
        return
    hz = 1.0 / float(np.median(dt))
    lo, hi = IMU_HZ_RANGE
    if lo <= hz <= hi:
        return
    guess = [u for u, k in TIME_UNITS.items() if u != "s" and lo <= hz / k <= hi]
    hint = (f"t looks like it is in {guess[0]}: pass --time-unit {guess[0]} "
            f"(load_csv(..., time_unit={guess[0]!r}))" if guess else "t must be in seconds")
    raise ValueError(f"{path}: IMU rate from t comes out at {hz:.3g} Hz, outside the plausible "
                     f"{lo:g}-{hi:g} Hz; {hint}")


def load_csv(path: str, accel_unit: str = "m/s2", gyro_unit: str = "rad/s",
             time_unit: str = "s") -> EdgeRecording:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        rows = [r for r in reader if r]
    missing = [c for c in ["t"] + IMU_COLS if c not in header]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; see edge/io.py for the format")

    t = _col(rows, header, "t")
    imu = np.stack([_col(rows, header, c) for c in IMU_COLS], axis=1)
    has_imu = np.isfinite(imu).all(axis=1)
    accel, gyro = imu[has_imu, :3], imu[has_imu, 3:]
    if accel_unit == "g":
        accel = accel * 9.80665
    if gyro_unit == "deg/s":
        gyro = np.radians(gyro)

    origin = None
    if "lat" in header and "lon" in header:
        lat, lon = _col(rows, header, "lat"), _col(rows, header, "lon")
        has_fix = np.isfinite(lat) & np.isfinite(lon)
        if has_fix.any():
            origin = (float(lat[has_fix][0]), float(lon[has_fix][0]))
            xy = latlon_to_xy(lat[has_fix], lon[has_fix], *origin)
        else:
            xy = np.zeros((0, 2))
    elif "x" in header and "y" in header:
        x, y = _col(rows, header, "x"), _col(rows, header, "y")
        has_fix = np.isfinite(x) & np.isfinite(y)
        xy = np.stack([x[has_fix], y[has_fix]], axis=1)
    else:
        has_fix = np.zeros(len(rows), dtype=bool)
        xy = np.zeros((0, 2))

    def opt(name):
        c = _col(rows, header, name)
        return c[has_fix] if c is not None else np.full(int(has_fix.sum()), np.nan)

    t = t * TIME_UNITS[time_unit]
    check_imu_rate(t[has_imu], path)
    return EdgeRecording(t[has_imu], accel, gyro, t[has_fix], xy, opt("speed"), opt("heading"), origin)


def write_csv(path: str, header: list[str], rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
