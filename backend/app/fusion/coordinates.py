# 10Hz Dead Reckoning Backend Engine - SIH 2026
import numpy as np
from typing import Tuple

GRAVITY = 9.80665

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Computes 3x3 Direction Cosine Matrix (DCM) from body frame to navigation frame.
    Uses direct closed-form Z-Y-X (yaw-pitch-roll) formulation for optimal 10Hz throughput.
    roll (phi): rotation about X
    pitch (theta): rotation about Y
    yaw (psi): rotation about Z
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr]
    ], dtype=np.float64)

def body_to_nav_acceleration(accel_body: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Rotates body acceleration into navigation frame and removes Earth gravity (9.81 m/s²).
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    accel_nav = R @ accel_body
    # Subtract gravity in the vertical Z-axis (pointing down)
    accel_nav[2] -= GRAVITY
    return accel_nav

def calculate_tilt_compensated_heading(mag_body: np.ndarray, roll: float, pitch: float) -> float:
    """
    Calculates compass heading from 3-axis magnetometer compensated for phone tilt (roll & pitch).
    """
    mx, my, mz = mag_body[0], mag_body[1], mag_body[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # De-tilt magnetometer readings into horizontal plane
    xh = mx * cp + my * sr * sp + mz * cr * sp
    yh = my * cr - mz * sr

    heading = np.arctan2(-yh, xh)
    return float(heading)

