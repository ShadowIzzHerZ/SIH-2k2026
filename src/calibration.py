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
     We estimate it by comparing the direction of horizontal acceleration
     during a straightish accelerating/braking segment against the
     GPS-derived heading (or, at inference in the field with no GPS yet,
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


def estimate_yaw_misalignment(level_accel_xy: np.ndarray, gt_heading_xy: np.ndarray) -> float:
    """Estimate constant yaw offset between the leveled phone frame and the
    vehicle forward direction, using a segment with known GT heading
    (e.g. GPS course-over-ground during a straight accelerating run).

    Args:
        level_accel_xy: (N, 2) horizontal accel in the *leveled* phone frame.
        gt_heading_xy: (N, 2) unit forward-direction vectors from GT (e.g.
            GPS bearing converted to a unit vector) for the same samples.

    Returns:
        psi: yaw offset in radians such that yaw_rotation(psi) applied to
        the leveled frame aligns it with the vehicle forward axis.
    """
    # Use the accel direction during the highest-magnitude (most confident)
    # samples only — near-zero accel gives an ill-defined direction.
    mag = np.linalg.norm(level_accel_xy, axis=1)
    keep = mag > np.percentile(mag, 75)
    a = level_accel_xy[keep]
    g = gt_heading_xy[keep]

    a_ang = np.arctan2(a[:, 1], a[:, 0])
    g_ang = np.arctan2(g[:, 1], g[:, 0])
    diff = np.arctan2(np.sin(g_ang - a_ang), np.cos(g_ang - a_ang))
    return float(np.median(diff))


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
