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


def _drive_segment(psi_true: float, course0: float, n: int = 600, dt: float = 0.1):
    """Leveled-phone horizontal accel of a car that speeds up, brakes and
    turns, where yaw_rotation(psi_true) takes the phone frame to the vehicle
    frame; plus its GPS course (world-frame unit vectors) and speed."""
    t = np.arange(n) * dt
    a_lon = 1.5 * np.sin(2 * np.pi * t / 20.0)            # accelerate, then brake
    yaw_rate = 0.15 * np.sin(2 * np.pi * t / 13.0)        # weave left and right
    v = 12.0 + np.cumsum(a_lon) * dt
    crs = course0 + np.cumsum(yaw_rate) * dt
    a_vehicle = np.stack([a_lon, v * yaw_rate], axis=1)   # [forward, left]
    accel_xy = (yaw_rotation(-psi_true)[:2, :2] @ a_vehicle.T).T
    heading_xy = np.stack([np.cos(crs), np.sin(crs)], axis=1)
    return accel_xy, heading_xy, v, dt


def test_yaw_misalignment_recovered_when_driving_along_world_x():
    accel_xy, heading_xy, v, dt = _drive_segment(np.radians(30), course0=0.0)
    psi = estimate_yaw_misalignment(accel_xy, heading_xy, v, dt)
    assert np.isclose(psi, np.radians(30), atol=np.radians(1))
    # rotating by psi puts braking back on -x and turns back on y
    a_vehicle = (yaw_rotation(psi)[:2, :2] @ accel_xy.T).T
    a_lon = 1.5 * np.sin(2 * np.pi * np.arange(len(v)) * dt / 20.0)
    assert np.allclose(a_vehicle[:, 0], a_lon, atol=0.05)


@pytest.mark.parametrize("course_deg", [90, 180, -135, 17])
@pytest.mark.parametrize("psi_deg", [30, -100, 170])
def test_yaw_misalignment_does_not_depend_on_course(course_deg, psi_deg):
    rng = np.random.default_rng(0)
    accel_xy, heading_xy, v, dt = _drive_segment(np.radians(psi_deg), course0=np.radians(course_deg))
    accel_xy = accel_xy + rng.normal(0, 0.2, accel_xy.shape)
    psi = estimate_yaw_misalignment(accel_xy, heading_xy, v, dt)
    assert abs(np.angle(np.exp(1j * (psi - np.radians(psi_deg))))) < np.radians(2), np.degrees(psi)


def test_yaw_misalignment_without_usable_samples_is_zero_not_nan():
    n = 400
    accel_xy = np.tile([0.4, 0.3], (n, 1))                # every magnitude equal
    heading_xy = np.tile([1.0, 0.0], (n, 1))
    for v in (np.full(n, 15.0), np.full(n, 0.5)):        # steady cruise / crawling
        assert estimate_yaw_misalignment(accel_xy, heading_xy, v, 0.1) == 0.0
    assert estimate_yaw_misalignment(accel_xy[:2], heading_xy[:2], np.full(2, 15.0), 0.1) == 0.0


def test_calibrate_sequence_finds_the_mount_yaw():
    from pathlib import Path

    from src.data.io_vnbd_loader import ImuSequence
    from src.data.windowing import calibrate_sequence

    n, psi_true = 4000, np.radians(-70)
    accel_xy, heading_xy, v, dt = _drive_segment(psi_true, course0=np.radians(200), n=n)
    accel = np.concatenate([accel_xy, np.full((n, 1), 9.81)], axis=1)
    bearing = (90.0 - np.degrees(np.arctan2(heading_xy[:, 1], heading_xy[:, 0]))) % 360.0
    seq = ImuSequence(path=Path("synthetic.csv"), time=np.arange(n) * dt, accel=accel.copy(),
                      gyro=np.zeros((n, 3)), speed_gt=v, lat=None, lon=None, heading_gt=bearing)
    calibrate_sequence(seq)
    a_lon = 1.5 * np.sin(2 * np.pi * np.arange(n) * dt / 20.0)
    assert np.corrcoef(seq.accel[:, 0], a_lon)[0, 1] > 0.99
