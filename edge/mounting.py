"""
Sensor -> vehicle frame for a fixed-mount IMU.

A phone gets re-calibrated every trip because it moves in its holder. A FOG
or industrial MEMS IMU is bolted down, so its mounting is usually known
from the installation drawing. Three ways to get the rotation, in order of
preference:

  1. --axes "x,-y,-z": a signed axis remap, read as "vehicle forward is
     sensor +x, vehicle left is sensor -y, vehicle up is sensor -z". Covers
     the usual 90/180 degree installs.
  2. --mount-rpy r,p,y: roll/pitch/yaw in degrees for an install at an
     arbitrary measured angle.
     Both are followed by a small gravity leveling pass (unless --no-level),
     because a fraction of a degree of residual tilt leaks gravity into the
     forward axis (0.1 rad of tilt is ~1 m/s^2, 15 m/s of speed error after
     15 s of blackout).
  3. auto (default): level from gravity over the first seconds (reusing
     src/calibration.py's leveling), then solve the yaw offset by matching
     the leveled horizontal acceleration against the acceleration GNSS
     says the vehicle had: longitudinal dv/dt from Doppler speed and
     lateral v * yaw-rate from the change in course over ground.

Why the yaw step does not reuse src.calibration.estimate_yaw_misalignment:
that function subtracts the sensor-frame accel angle from the *world*
course angle, so its answer is the vehicle's average compass heading plus
the mount offset, not the mount offset alone (a perfectly aligned sensor
on a north-bound road comes out at 90 degrees). Both quantities here are in
the vehicle frame, so the difference is the mount offset only, and braking
samples (accel pointing backwards) count correctly instead of voting for
a 180 degree flip.

The vehicle frame is the one the model was trained in: x forward, y left,
z up, with gravity reading about +9.8 m/s^2 on z.
"""
from __future__ import annotations

import warnings

import numpy as np

from src.calibration import estimate_gravity_vector, leveling_rotation, yaw_rotation

_AXIS = {"x": 0, "y": 1, "z": 2}


def rotation_from_axes(spec: str) -> np.ndarray:
    """'x,-y,-z' -> 3x3 R with vehicle = R @ sensor."""
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--axes needs three comma-separated axes, got {spec!r}")
    R = np.zeros((3, 3))
    for row, p in enumerate(parts):
        sign = -1.0 if p.startswith("-") else 1.0
        axis = p.lstrip("+-")
        if axis not in _AXIS:
            raise ValueError(f"bad axis {p!r} in {spec!r}")
        R[row, _AXIS[axis]] = sign
    if abs(abs(np.linalg.det(R)) - 1) > 1e-9:
        raise ValueError(f"--axes {spec!r} repeats an axis")
    if np.linalg.det(R) < 0:
        raise ValueError(f"--axes {spec!r} is a reflection, not a rotation (left-handed frame)")
    return R


def rotation_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Z-Y-X (yaw, then pitch, then roll) mounting angles -> vehicle = R @ sensor."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


def level(accel: np.ndarray, R_pre: np.ndarray | None = None) -> tuple[np.ndarray, list[float]]:
    """Roll/pitch leveling on top of an optional known rotation R_pre.

    src.calibration.leveling_rotation takes the shortest rotation from the
    measured gravity to +z. For a sensor mounted upside down (gravity
    reading about -9.8 on z, e.g. comma2k19's x-forward / y-right / z-down
    device) that shortest rotation is a ~180 degree turn about a horizontal
    axis set by the few-percent tilt components, so the yaw it leaves
    behind is essentially arbitrary: across the 64 comma2k19 demo segments
    the leveled x axis landed between -179 and +178 degrees off forward
    (src.calibration.leveling_rotation on each segment's first 2 s). Flipping 180 degrees about x first (so gravity is already
    near +z) makes the remaining correction a small tilt with no yaw.
    """
    R0 = np.eye(3) if R_pre is None else np.asarray(R_pre, dtype=np.float64)
    g = estimate_gravity_vector(accel @ R0.T)
    if g[2] < 0:
        R0 = np.diag([1.0, -1.0, -1.0]) @ R0
        g = estimate_gravity_vector(accel @ R0.T)
    return leveling_rotation(g) @ R0, [round(float(v), 4) for v in g]


def estimate_mount(accel: np.ndarray, dt: float, level_samples: int, gnss_heading_deg: np.ndarray | None = None,
                   gnss_speed: np.ndarray | None = None, R_pre: np.ndarray | None = None, estimate_yaw: bool = True,
                   min_speed_mps: float = 2.0, min_yaw_samples: int = 300,
                   min_horiz_accel: float = 0.3, min_fit_quality: float = 0.5) -> tuple[np.ndarray, dict]:
    """Mount estimate from model-rate (10 Hz) samples: level, then
    (optionally) solve the yaw offset against GNSS.

    accel: (N, 3) sensor-frame accel. gnss_heading_deg / gnss_speed: (N,)
    compass course and Doppler speed per sample, NaN where there was no
    fix. Only consecutive samples that both have a fix are used, so a
    blackout never leaks ground truth into the calibration.

    min_yaw_samples defaults to 30 s of usable moving data. Below that the
    yaw offset stays 0 and info["yaw_deg"] says so. The same happens when
    the fit is poor (quality = |sum of m.e as complex products| / sum |m||e|,
    1.0 when every sample agrees on the angle): steady highway driving has
    so little horizontal acceleration that one comma2k19 segment fitted
    173 degrees from 300+ samples, which would have run the whole blackout
    backwards.
    """
    R_level, g = level(accel[:max(1, min(level_samples, len(accel)))], R_pre)
    info = {"gravity_after_premount": g, "yaw_deg": 0.0, "yaw_samples": 0}
    if not estimate_yaw or gnss_heading_deg is None or gnss_speed is None or len(accel) < 3:
        return R_level, info

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # same benign BLAS warning as calibrate_sequence
        level_acc = accel @ R_level.T

    # Central differences of GNSS speed and course -> expected vehicle-frame
    # horizontal accel [forward, left]. Compass course grows clockwise, the
    # vehicle frame's y is left, so lateral accel = -v * d(course)/dt.
    v, crs = gnss_speed, np.radians(gnss_heading_deg)
    a_lon = (v[2:] - v[:-2]) / (2 * dt)
    dcrs = np.arctan2(np.sin(crs[2:] - crs[:-2]), np.cos(crs[2:] - crs[:-2])) / (2 * dt)
    a_lat = -v[1:-1] * dcrs
    meas = level_acc[1:-1, :2]
    expect = np.stack([a_lon, a_lat], axis=1)
    ok = (np.isfinite(expect).all(axis=1) & (np.nan_to_num(v[1:-1]) > min_speed_mps)
          & (np.linalg.norm(expect, axis=1) > min_horiz_accel))
    info["yaw_samples"] = int(ok.sum())
    if ok.sum() < min_yaw_samples:
        return R_level, info

    # Least-squares 2D rotation taking measured onto expected: psi such that
    # yaw_rotation(psi) @ meas ~ expect.
    m, e = meas[ok], expect[ok]
    cross = np.sum(m[:, 0] * e[:, 1] - m[:, 1] * e[:, 0])
    dot = np.sum(m[:, 0] * e[:, 0] + m[:, 1] * e[:, 1])
    quality = float(np.hypot(cross, dot) / np.sum(np.linalg.norm(m, axis=1) * np.linalg.norm(e, axis=1)))
    info["yaw_fit_quality"] = round(quality, 3)
    if quality < min_fit_quality:
        return R_level, info
    psi = float(np.arctan2(cross, dot))
    info["yaw_deg"] = round(float(np.degrees(psi)), 2)
    return yaw_rotation(psi) @ R_level, info
