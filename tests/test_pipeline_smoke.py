"""
Synthetic smoke test — doesn't touch the real dataset at all.

Generates IMU that's internally consistent with a known circular-arc
trajectory (constant speed + constant turn rate), runs it through
BiasCorrectionNet + the strapdown integrator + the compound loss exactly
as train.py does, and checks:
  1. shapes line up end to end
  2. a few optimizer steps actually reduce the loss (i.e. gradients flow
     correctly through the integrator back into the network)
  3. with zero learned correction and zero-noise "IMU", the physics-only
     integrator alone reconstructs the known trajectory to near-zero drift
     (i.e. strapdown_ins.py's integration math is correct)

This is the fast thing to run whenever any of strapdown_ins.py,
bias_correction_net.py, or train.py's forward_pass/compute_loss change —
real-data training runs take much longer to reveal a broken integration.

Run:
    python -m tests.test_pipeline_smoke
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.bias_correction_net import BiasCorrectionNet
from src.models.strapdown_ins import dead_reckon_position, drift_metric, integrate_heading, integrate_speed
from src.train import compute_loss


def synth_circular_arc(T: int, dt: float, speed: float = 10.0, turn_rate: float = 0.05, noise: float = 0.0):
    """Constant-speed, constant-turn-rate motion -> exact ground truth,
    and the forward-accel / yaw-rate an ideal IMU would report for it."""
    t = np.arange(T) * dt
    heading = turn_rate * t
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt
    pos = np.stack([x, y], axis=1)

    forward_accel = np.zeros(T)  # constant speed -> zero forward accel
    yaw_rate = np.full(T, turn_rate)

    if noise > 0:
        forward_accel = forward_accel + np.random.normal(0, noise, T)
        yaw_rate = yaw_rate + np.random.normal(0, noise * 0.1, T)

    accel = np.stack([forward_accel, np.zeros(T), np.full(T, 9.81)], axis=1)
    gyro = np.stack([np.zeros(T), np.zeros(T), yaw_rate], axis=1)

    speed_gt = np.full(T, speed)
    return accel.astype(np.float32), gyro.astype(np.float32), speed_gt.astype(np.float32), pos.astype(np.float32)


def test_physics_only_reconstructs_known_trajectory():
    dt = 0.1
    T = 50
    accel, gyro, speed_gt, pos_gt = synth_circular_arc(T, dt, noise=0.0)

    forward_accel = torch.from_numpy(accel[:, 0]).unsqueeze(0)
    yaw_rate = torch.from_numpy(gyro[:, 2]).unsqueeze(0)

    speed_pred = integrate_speed(forward_accel, dt, v0=10.0)  # v0 known exactly here
    heading_pred = integrate_heading(yaw_rate, dt, theta0=0.0)
    pos_pred = dead_reckon_position(speed_pred, heading_pred, dt)

    drift = drift_metric(pos_pred, torch.from_numpy(pos_gt).unsqueeze(0))
    print(f"physics-only drift on noise-free synthetic arc: {drift.item():.4f}%")
    assert drift.item() < 1.0, "integrator math is wrong — should be ~0% drift on a noise-free synthetic case"


def test_gradients_flow_and_loss_decreases():
    dt = 0.1
    T, B = 50, 8
    batch_accel, batch_gyro, batch_speed, batch_pos = [], [], [], []
    for _ in range(B):
        speed = np.random.uniform(5, 15)
        turn = np.random.uniform(-0.1, 0.1)
        a, g, s, p = synth_circular_arc(T, dt, speed=speed, turn_rate=turn, noise=0.3)
        batch_accel.append(a)
        batch_gyro.append(g)
        batch_speed.append(s)
        batch_pos.append(p)

    imu = torch.from_numpy(np.concatenate([np.stack(batch_accel), np.stack(batch_gyro)], axis=2))
    speed_gt = torch.from_numpy(np.stack(batch_speed))
    pos_gt = torch.from_numpy(np.stack(batch_pos))

    model = BiasCorrectionNet(input_channels=6, cnn_channels=[16, 32], gru_hidden=32, gru_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    weights = {"speed": 1.0, "drift": 1.0}

    losses = []
    for step in range(20):
        opt.zero_grad()
        corrections = model(imu)
        delta_v, delta_theta = corrections[..., 0], corrections[..., 1]
        forward_accel = imu[..., 0] + delta_v
        yaw_rate = imu[..., 5] + delta_theta
        speed_pred = integrate_speed(forward_accel, dt, v0=speed_gt[:, 0])
        heading_pred = integrate_heading(yaw_rate, dt)
        pos_pred = dead_reckon_position(speed_pred, heading_pred, dt)
        loss, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, weights)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "no gradient reached the network — integrator broke the graph"
        opt.step()
        losses.append(loss.item())

    print(f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0], "loss did not decrease — training loop is broken"


if __name__ == "__main__":
    test_physics_only_reconstructs_known_trajectory()
    test_gradients_flow_and_loss_decreases()
    print("\n✅ smoke test passed")
