"""
leveling_rotation on a z-down sensor: a small tilt must not leave an
arbitrary yaw (synthetic, no data).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import leveling_rotation


def _yaw_deg(R: np.ndarray) -> float:
    """Where the sensor's +x axis points in the leveled frame, degrees."""
    x = R @ np.array([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(x[1], x[0])))


@pytest.mark.parametrize("seed", range(20))
def test_z_down_with_tilt_noise_keeps_x_forward(seed):
    rng = np.random.default_rng(seed)
    g = np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.01, 3)
    R = leveling_rotation(g)
    assert np.allclose(R @ (g / np.linalg.norm(g)), [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    assert abs(_yaw_deg(R)) < 2.0


def test_exactly_z_down_is_a_flip_about_x():
    assert np.allclose(leveling_rotation(np.array([0.0, 0.0, -9.81])), np.diag([1.0, -1.0, -1.0]))


@pytest.mark.parametrize("g", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.6, 0.0]])
def test_upright_mount_is_continuous_across_the_horizon(g):
    g = np.asarray(g)
    up = leveling_rotation(g + [0.0, 0.0, 0.05])
    down = leveling_rotation(g - [0.0, 0.0, 0.05])
    assert np.allclose(up, down, atol=0.15)
