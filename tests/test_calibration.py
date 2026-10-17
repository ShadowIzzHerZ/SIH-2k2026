"""
Offline unit tests for src/calibration.py (no data, no network).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    calibrate,
    estimate_gravity_vector,
    estimate_yaw_misalignment,
    leveling_rotation,
    yaw_rotation,
)


def _is_rotation(R: np.ndarray) -> bool:
    return np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_gravity_vector_is_unit_mean_direction():
    rng = np.random.default_rng(0)
    accel = np.array([0.3, -0.2, 9.8]) + rng.normal(0, 0.05, size=(200, 3))
    g = estimate_gravity_vector(accel)
    assert np.isclose(np.linalg.norm(g), 1.0)
    assert np.allclose(g, np.array([0.3, -0.2, 9.8]) / np.linalg.norm([0.3, -0.2, 9.8]), atol=1e-3)


def test_gravity_vector_rejects_all_zero_window():
    with pytest.raises(ValueError):
        estimate_gravity_vector(np.zeros((10, 3)))


@pytest.mark.parametrize("g", [
    [0.0, 0.0, 1.0],
    [0.1, -0.2, 0.97],
    [0.6, 0.0, 0.8],
    [0.0, 1.0, 0.0],        # phone on its side
    [0.01, 0.02, -1.0],     # nearly upside down
    [0.0, 0.0, -1.0],       # exactly upside down (degenerate axis branch)
])
def test_leveling_rotation_maps_gravity_to_up(g):
    g = np.asarray(g, dtype=float)
    R = leveling_rotation(g)
    assert _is_rotation(R)
    assert np.allclose(R @ (g / np.linalg.norm(g)), [0.0, 0.0, 1.0], atol=1e-9)


def test_leveling_rotation_is_identity_when_already_level():
    assert np.allclose(leveling_rotation(np.array([0.0, 0.0, 9.81])), np.eye(3))


def test_yaw_rotation_turns_x_towards_y():
    R = yaw_rotation(np.pi / 2)
    assert _is_rotation(R)
    assert np.allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert np.allclose(R @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])


def test_calibrate_applies_level_then_yaw_to_both_streams():
    rng = np.random.default_rng(1)
    accel = rng.normal(size=(20, 3))
    gyro = rng.normal(size=(20, 3))
    R_level = leveling_rotation(np.array([0.2, 0.1, 0.97]))
    psi = 0.4
    a_v, g_v = calibrate(accel, gyro, R_level, psi)
    R = yaw_rotation(psi) @ R_level
    assert np.allclose(a_v, (R @ accel.T).T)
    assert np.allclose(g_v, (R @ gyro.T).T)
    # rotation preserves magnitudes
    assert np.allclose(np.linalg.norm(a_v, axis=1), np.linalg.norm(accel, axis=1))


def _forward_accel_segment(psi_true: float, course: float, n: int = 400):
    """Leveled-phone horizontal accel of a car accelerating along its own
    forward axis, where yaw_rotation(psi_true) takes the phone frame to the
    vehicle frame; plus the GPS course as a world-frame unit vector."""
    rng = np.random.default_rng(0)
    a_fwd = rng.uniform(0.2, 2.0, n)
    fwd_in_phone = np.array([np.cos(-psi_true), np.sin(-psi_true)])
    accel_xy = a_fwd[:, None] * fwd_in_phone
    heading_xy = np.tile([np.cos(course), np.sin(course)], (n, 1))
    return accel_xy, heading_xy


def test_yaw_misalignment_recovered_when_driving_along_world_x():
    accel_xy, heading_xy = _forward_accel_segment(np.radians(30), course=0.0)
    psi = estimate_yaw_misalignment(accel_xy, heading_xy)
    assert np.isclose(psi, np.radians(30), atol=1e-6)
    a_vehicle = (yaw_rotation(psi)[:2, :2] @ accel_xy.T).T
    assert np.all(a_vehicle[:, 0] > 0) and np.allclose(a_vehicle[:, 1], 0, atol=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: estimate_yaw_misalignment subtracts the sensor-frame accel angle from the "
    "WORLD-frame GPS course, so the estimate is offset by the course itself (psi + course) "
    "whenever the car is not heading along world +x."))
def test_yaw_misalignment_does_not_depend_on_course():
    accel_xy, heading_xy = _forward_accel_segment(np.radians(30), course=np.radians(90))
    assert np.isclose(estimate_yaw_misalignment(accel_xy, heading_xy), np.radians(30), atol=1e-3)
