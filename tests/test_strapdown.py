"""
Offline unit tests for src/models/strapdown_ins.py on synthetic motion with
a closed-form answer: constant velocity and a constant-rate turn.

Convention under test (see integrate_heading's docstring): index k of every
integrated output is the state AFTER applying sample k, i.e. k+1 steps
from the initial state.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.models.strapdown_ins import (
    DeadReckoner,
    dead_reckon_position,
    drift_metric,
    integrate_heading,
    integrate_speed,
)

DT = 0.1


def test_constant_velocity_straight_line():
    T, v0, theta0 = 50, 12.0, 0.7
    speed, heading, pos = DeadReckoner(DT).integrate(
        torch.zeros(2, T, dtype=torch.float64), torch.zeros(2, T, dtype=torch.float64),
        v0=v0, theta0=theta0)
    assert speed.shape == (2, T) and heading.shape == (2, T) and pos.shape == (2, T, 2)
    assert torch.allclose(speed, torch.full_like(speed, v0))
    assert torch.allclose(heading, torch.full_like(heading, theta0))
    k = torch.arange(1, T + 1, dtype=torch.float64)
    expected = torch.stack([v0 * k * DT * math.cos(theta0), v0 * k * DT * math.sin(theta0)], dim=-1)
    assert torch.allclose(pos[0], expected) and torch.allclose(pos[1], expected)


def test_constant_acceleration_speed_and_distance():
    T, a = 40, 1.5
    speed = integrate_speed(torch.full((1, T), a, dtype=torch.float64), DT, v0=2.0)
    k = torch.arange(1, T + 1, dtype=torch.float64)
    assert torch.allclose(speed[0], 2.0 + a * k * DT)
    pos = dead_reckon_position(speed, torch.zeros_like(speed), DT)
    # rectangle-rule sum of the speeds above
    assert torch.allclose(pos[0, -1, 0], (speed[0] * DT).sum())
    assert torch.allclose(pos[0, :, 1], torch.zeros(T, dtype=torch.float64))


def test_constant_turn_traces_a_circle():
    v, omega = 10.0, 0.2           # radius 50 m, one lap every 2*pi/0.2 = 31.4 s
    T = round(2 * math.pi / omega / DT)
    speed, heading, pos = DeadReckoner(DT).integrate(
        torch.zeros(1, T, dtype=torch.float64), torch.full((1, T), omega, dtype=torch.float64), v0=v)
    radius = v / omega
    centre = torch.tensor([0.0, radius], dtype=torch.float64)  # left turn from heading 0 at the origin
    r = torch.linalg.norm(pos[0] - centre, dim=-1)
    # forward-Euler on a circle: radius error stays within one step's chord
    assert torch.all((r - radius).abs() < v * DT)
    assert torch.isclose(heading[0, -1], torch.tensor(omega * T * DT, dtype=torch.float64))
    # after ~one full lap it ends back near the start
    assert torch.linalg.norm(pos[0, -1]) < 2 * v * DT


def test_heading_accepts_per_batch_initial_state():
    theta0 = torch.tensor([0.0, 1.0, -2.0])
    th = integrate_heading(torch.full((3, 5), 0.1), DT, theta0=theta0)
    assert torch.allclose(th[:, 0], theta0 + 0.01)
    assert torch.allclose(th[:, -1], theta0 + 0.05)


def test_speed_clamps_at_zero_unless_disabled():
    decel = torch.full((1, 30), -2.0)
    assert integrate_speed(decel, DT, v0=1.0).min() == 0.0
    assert integrate_speed(decel, DT, v0=1.0, clamp_min=None).min() < 0.0


def test_dead_reckon_position_offsets_by_p0():
    speed = torch.full((2, 10), 5.0)
    p0 = torch.tensor([[100.0, -3.0], [0.0, 7.0]])
    pos = dead_reckon_position(speed, torch.zeros(2, 10), DT, p0=p0)
    assert torch.allclose(pos[:, -1], p0 + torch.tensor([5.0, 0.0]))


def test_drift_metric_is_final_error_over_path_length():
    gt = torch.zeros(1, 11, 2)
    gt[0, :, 0] = torch.arange(11.0) * 10    # 100 m straight line
    pred = gt.clone()
    pred[0, -1] = torch.tensor([100.0, 5.0])  # 5 m off at the end
    assert drift_metric(pred, gt).item() == pytest.approx(5.0)
    assert drift_metric(gt, gt).item() == 0.0


def test_integrator_is_differentiable():
    acc = torch.zeros(1, 20, requires_grad=True)
    yaw = torch.zeros(1, 20, requires_grad=True)
    _, _, pos = DeadReckoner(DT).integrate(acc, yaw, v0=10.0)
    pos[0, -1, 1].backward()
    assert acc.grad is not None and yaw.grad is not None
    assert yaw.grad.abs().sum() > 0   # turning left moves the end point in +y
