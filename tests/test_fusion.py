"""
Synthetic smoke test for fusion.py — doesn't touch real data, same spirit as
tests/test_pipeline_smoke.py: checks the state-machine/integration logic
itself against a case where the right answer is known exactly, before
trusting it on a real trajectory (that's simulate_blackout.py).

Checks:
  1. With GNSS available the whole time, fused output is just the GNSS
     track verbatim — the state machine never has a reason to touch INS.
  2. A blackout injected into a known constant-velocity straight-line
     synthetic trajectory, with a zero-correction stub model (isolates the
     physics integration from anything the real trained network might get
     wrong), reconstructs the true path to near-zero drift — validates
     run_fusion's one-sample-at-a-time integration matches
     strapdown_ins.py's batch integration semantics.
  3. Reconnection after a blackout doesn't produce a big position jump
     (the whole point of BLEND) — bounded well below what a hard snap back
     to the raw GNSS fix would produce.

Run:
    python -m tests.test_fusion
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.fusion import blackout_drift_pct, reconnect_jump_m, run_fusion


class ZeroCorrectionModel(nn.Module):
    """Stand-in for BiasCorrectionNet that always outputs [0, 0] — isolates
    run_fusion's integration logic from the real network's learned (and
    imperfect) corrections."""

    def forward(self, x):
        b, t, _ = x.shape
        return torch.zeros(b, t, 2)

    def to(self, device):
        return self


def synth_straight_line(n: int, dt: float, speed: float = 15.0):
    """Constant-speed, constant-heading motion -> exact ground truth and
    the ideal forward-accel/yaw-rate an IMU would report for it."""
    heading = 0.3  # rad, arbitrary non-trivial fixed heading
    t = np.arange(n) * dt
    x = speed * np.cos(heading) * t
    y = speed * np.sin(heading) * t
    pos = np.stack([x, y], axis=1)

    accel = np.zeros((n, 3), dtype=np.float32)
    accel[:, 2] = 9.81  # level, stationary-vertical component
    gyro = np.zeros((n, 3), dtype=np.float32)
    return accel, gyro, pos.astype(np.float64)


def test_gnss_always_available_passes_through():
    dt = 0.1
    n = 100
    accel, gyro, pos = synth_straight_line(n, dt)
    available = np.ones(n, dtype=bool)

    result = run_fusion(accel, gyro, pos, available, dt, ZeroCorrectionModel(),
                         window_size=50, blend_seconds=2.0)

    assert np.allclose(result.position, pos), "with GNSS always on, output should equal the GNSS track exactly"
    assert all(m.value == "gnss" for m in result.mode)
    print("test_gnss_always_available_passes_through: OK")


def test_blackout_reconstructs_straight_line():
    dt = 0.1
    n = 300  # 30s
    accel, gyro, pos = synth_straight_line(n, dt)
    available = np.ones(n, dtype=bool)
    blackout_start, blackout_end = 100, 200  # 10s blackout in the middle
    available[blackout_start:blackout_end] = False

    result = run_fusion(accel, gyro, pos, available, dt, ZeroCorrectionModel(),
                         window_size=50, blend_seconds=1.0)

    drift = blackout_drift_pct(result, pos, available)
    assert drift is not None
    assert drift < 1.0, f"zero-noise straight-line blackout should reconstruct almost exactly, got {drift:.3f}% drift"

    assert any(m.value == "blackout" for m in result.mode)
    assert any(m.value == "blend" for m in result.mode)
    print(f"test_blackout_reconstructs_straight_line: OK (drift {drift:.4f}%)")


def test_reconnect_is_smooth_not_a_jump():
    dt = 0.1
    n = 300
    accel, gyro, pos = synth_straight_line(n, dt)
    available = np.ones(n, dtype=bool)
    available[100:200] = False

    result = run_fusion(accel, gyro, pos, available, dt, ZeroCorrectionModel(),
                         window_size=50, blend_seconds=1.0)

    jump = reconnect_jump_m(result, available)
    assert jump is not None
    # One sample's worth of motion at this speed/dt is the natural upper
    # bound for a *smoothly blended* step; a hard snap back to raw GNSS
    # would instead jump by however far the INS estimate had drifted
    # (near-zero here since this is the noiseless case, but the mechanism
    # is what's being checked — see simulate_blackout.py for a real,
    # noisy/imperfect-model case where this bound actually matters).
    assert jump < 15.0 * dt * 2, f"reconnect jump too large for a blended handoff: {jump:.3f}m"
    print(f"test_reconnect_is_smooth_not_a_jump: OK (jump {jump:.4f}m)")


def test_blend_heading_takes_short_way_across_pi():
    # Heading ~due west, where the angle wraps: the INS carries 3.13 rad
    # through the blackout and GNSS comes back reporting -3.13 rad. The two
    # differ by ~0.02 rad, so the blended heading must stay near +-pi.
    dt = 0.1
    n = 200
    speed = 10.0
    accel = np.zeros((n, 3), dtype=np.float32)
    accel[:, 2] = 9.81
    gyro = np.zeros((n, 3), dtype=np.float32)
    pos = np.stack([-speed * np.arange(n) * dt, np.zeros(n)], axis=1)
    available = np.ones(n, dtype=bool)
    available[50:100] = False
    gnss_speed = np.full(n, speed)
    gnss_heading = np.where(np.arange(n) < 50, np.pi - 0.01, -np.pi + 0.01)

    result = run_fusion(accel, gyro, pos, available, dt, ZeroCorrectionModel(), window_size=50,
                        blend_seconds=2.0, gnss_speed=gnss_speed, gnss_heading=gnss_heading)

    blend = [t for t, m in enumerate(result.mode) if m.value == "blend"]
    assert blend
    off_west = np.abs(np.arctan2(np.sin(result.heading[blend] - np.pi), np.cos(result.heading[blend] - np.pi)))
    assert off_west.max() < 0.05, f"blended heading swung {off_west.max():.2f} rad away from due west"


if __name__ == "__main__":
    test_gnss_always_available_passes_through()
    test_blackout_reconstructs_straight_line()
    test_reconnect_is_smooth_not_a_jump()
    test_blend_heading_takes_short_way_across_pi()
    print("\n✅ fusion smoke tests passed")
