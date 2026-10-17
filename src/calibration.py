"""
Phone-to-vehicle frame calibration/alignment.

The PS explicitly calls out "in-vehicle phone alignment/calibration" as a
named component, and covers both dashboard-mounted and hand-held-in-holder
placement. A phone's IMU reports in the *phone's* body frame, which is
rotated arbitrarily relative to the car (mount angle, phone orientation,
whether it's in portrait/landscape). Before any dead-reckoning math means
anything, we need to rotate raw IMU samples into a consistent vehicle
frame: x = forward, y = left, z = up.

Two-stage estimate, standard for this problem:
  1. Leveling (roll/pitch) from gravity: while the vehicle is stationary
     (or from the low-pass/gravity component of accel while moving), the
     accelerometer measures ~[0,0,g] in the *level* frame. The tilt
     between that and the raw phone reading gives roll/pitch.
  2. Yaw (heading) misalignment: the phone's x-axis heading offset from
     the vehicle's forward direction isn't observable from gravity alone.
     We estimate it by matching the horizontal acceleration against the
     acceleration GPS speed and course imply in the vehicle frame (or, at
     inference in the field with no GPS yet,
     we require a short calibration drive/segment before blackout, same
     as the PS's "instant seamless mode-switching" implies: leveling
     happens continuously, yaw-alignment happens once at trip start).

This module gives you the rotation only; it does not integrate anything
(see strapdown_ins.py for that).
"""
from __future__ import annotations

import numpy as np


def estimate_gravity_vector(accel: np.ndarray, method: str = "mean") -> np.ndarray:
    """Estimate the gravity direction from a (mostly) stationary accel window.

    Args:
        accel: (N, 3) raw accelerometer samples, m/s^2, phone frame.
        method: "mean" (assumes truly stationary) — swap for a low-pass
            filter if you calibrate during slow motion instead.

    Returns:
        (3,) unit gravity vector in the phone frame.
    """
    g = accel.mean(axis=0) if method == "mean" else accel[0]
    norm = np.linalg.norm(g)
    if norm < 1e-6:
        raise ValueError("Degenerate gravity estimate — accel window is all zero.")
    return g / norm


def leveling_rotation(gravity_phone: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps the phone frame to a level frame
    (z-axis aligned with true vertical), leaving yaw unresolved.

    Uses the standard "align two vectors" construction (Rodrigues) mapping
    gravity_phone -> [0, 0, -1] (accelerometer reads -g when level & still,
    ENU-style: z up, gravity vector measured pointing up due to reaction force).
    """
    target = np.array([0.0, 0.0, 1.0])
    v = gravity_phone / np.linalg.norm(gravity_phone)

    axis = np.cross(v, target)
    s = np.linalg.norm(axis)
    c = np.dot(v, target)

    if s < 1e-8:
        # Already aligned (or exactly opposite) — no rotation needed / undefined axis.
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])

    axis = axis / s
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + K * s + (K @ K) * (1 - c)
    return R


def yaw_rotation(psi: float) -> np.ndarray:
    """Rotation about z by yaw angle psi (radians)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def estimate_yaw_misalignment(level_accel_xy: np.ndarray, gt_heading_xy: np.ndarray, speed_mps: np.ndarray,
                              dt: float, min_speed_mps: float = 2.0, min_horiz_accel: float = 0.3,
                              min_fit_quality: float = 0.5) -> float:
    """Estimate the constant yaw offset between the leveled phone frame and
    the vehicle forward direction, against GPS speed and course.

    The leveled phone frame has an unknown yaw, so its accel direction can't
    be compared with a world-frame course directly (that used to return the
    mount offset plus the average course). Instead both sides are put in the
    vehicle frame: GPS says the car had longitudinal accel dv/dt and lateral
    accel v * d(course)/dt, and psi is the least-squares 2D rotation taking
    the measured horizontal accel onto that. Braking counts the same as
    accelerating instead of voting for a 180 degree flip. Same method as
    edge/mounting.py's estimate_mount.

    Args:
        level_accel_xy: (N, 2) horizontal accel in the *leveled* phone frame,
            consecutive samples dt apart.
        gt_heading_xy: (N, 2) unit direction-of-travel vectors from GT (GPS
            course converted with compass_deg_to_xy_unit), same samples.
        speed_mps: (N,) GT speed, same samples.
        dt: sample period, s.
        min_speed_mps: samples slower than this are ignored (course is noise).
        min_horiz_accel: samples whose GPS-implied accel is smaller than this
            (m/s^2) are ignored; near-zero accel has no direction.
        min_fit_quality: |sum m.e| / sum |m||e| over the used samples (1.0 when
            they all agree on the angle). Below it, or with no usable sample,
            returns 0.0 (uncalibrated) rather than a guess.

    Returns:
        psi: yaw offset in radians such that yaw_rotation(psi) applied to
        the leveled frame aligns it with the vehicle forward axis.
    """
    if len(level_accel_xy) < 3:
        return 0.0
    v = np.asarray(speed_mps, dtype=np.float64)
    crs = np.arctan2(gt_heading_xy[:, 1], gt_heading_xy[:, 0])  # math angle, counter-clockwise
    a_lon = (v[2:] - v[:-2]) / (2 * dt)
    a_lat = v[1:-1] * np.arctan2(np.sin(crs[2:] - crs[:-2]), np.cos(crs[2:] - crs[:-2])) / (2 * dt)
    expect = np.stack([a_lon, a_lat], axis=1)
    meas = np.asarray(level_accel_xy, dtype=np.float64)[1:-1]
    ok = (np.isfinite(expect).all(axis=1) & np.isfinite(meas).all(axis=1)
          & (np.nan_to_num(v[1:-1]) > min_speed_mps) & (np.linalg.norm(expect, axis=1) > min_horiz_accel))
    if not ok.any():
        return 0.0
    m, e = meas[ok], expect[ok]
    cross = np.sum(m[:, 0] * e[:, 1] - m[:, 1] * e[:, 0])
    dot = np.sum(m[:, 0] * e[:, 0] + m[:, 1] * e[:, 1])
    norm = np.sum(np.linalg.norm(m, axis=1) * np.linalg.norm(e, axis=1))
    if norm <= 0 or np.hypot(cross, dot) / norm < min_fit_quality:
        return 0.0
    return float(np.arctan2(cross, dot))


def calibrate(accel: np.ndarray, gyro: np.ndarray, R_level: np.ndarray, psi_yaw: float = 0.0):
    """Apply the full phone->vehicle rotation to raw IMU streams.

    Args:
        accel, gyro: (N, 3) raw phone-frame samples.
        R_level: from leveling_rotation().
        psi_yaw: from estimate_yaw_misalignment(), 0.0 if not yet calibrated.

    Returns:
        accel_v, gyro_v: (N, 3) in the vehicle frame [forward, left, up].
    """
    R = yaw_rotation(psi_yaw) @ R_level
    accel_v = accel @ R.T
    gyro_v = gyro @ R.T
    return accel_v, gyro_v
