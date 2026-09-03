"""
Differentiable 2D dead-reckoning integrator for ground vehicles.

Why 2D and not full 3D strapdown:
    A vehicle on a road is a non-holonomic system — it can't move sideways
    or vertically relative to itself. That constraint (explicitly called
    out in the PS) means we don't need a full 3D strapdown mechanization
    with quaternion attitude propagation; we only need:
        1. heading (yaw) angle, tracked by integrating calibrated yaw rate
        2. forward speed, estimated from calibrated forward acceleration
    and then dead-reckon x/y from heading + speed. This is the standard
    approach in vehicle/pedestrian inertial dead reckoning (RIDI, IONet,
    and the WhONet-style work the IO-VNBD dataset itself was built for).

This module is written to be differentiable (pure torch ops, no in-place
numpy) so it can sit *inside* the training loop: the network's residual
corrections feed into this integrator, and the loss is computed on the
integrated trajectory, not just on raw network output. That's what makes
this "physics-informed" rather than a black-box regressor.

The same function (without gradient tracking) is used again at inference
time on-device, which is why it takes plain tensors in/out and has no
framework-specific bells on it — this file is what gets carried into the
ONNX export path.
"""
from __future__ import annotations

import torch


def integrate_heading(yaw_rate: torch.Tensor, dt: float, theta0: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Integrate yaw rate (rad/s) into heading (rad) over time.

    Args:
        yaw_rate: (batch, T) calibrated gyro-z, already in the vehicle frame.
        dt: sample period in seconds.
        theta0: initial heading, scalar or (batch,).

    Returns:
        theta: (batch, T) heading at each timestep (theta[:,0] is the first
        *post*-integration sample, i.e. theta0 + yaw_rate[:,0]*dt).
    """
    if not torch.is_tensor(theta0):
        theta0 = torch.full((yaw_rate.shape[0],), float(theta0), device=yaw_rate.device, dtype=yaw_rate.dtype)
    dtheta = yaw_rate * dt
    theta = theta0.unsqueeze(1) + torch.cumsum(dtheta, dim=1)
    return theta


def integrate_speed(forward_accel: torch.Tensor, dt: float, v0: torch.Tensor | float = 0.0,
                     clamp_min: float = 0.0) -> torch.Tensor:
    """Integrate forward (vehicle x-axis) acceleration into speed.

    clamp_min=0.0 enforces the non-holonomic / no-reverse assumption used
    by the PS's demo scenarios (forward driving only). Set to None to allow
    negative speed (reversing) if you extend this later.
    """
    if not torch.is_tensor(v0):
        v0 = torch.full((forward_accel.shape[0],), float(v0), device=forward_accel.device, dtype=forward_accel.dtype)
    dv = forward_accel * dt
    v = v0.unsqueeze(1) + torch.cumsum(dv, dim=1)
    if clamp_min is not None:
        v = torch.clamp(v, min=clamp_min)
    return v


def dead_reckon_position(speed: torch.Tensor, heading: torch.Tensor, dt: float,
                          p0: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate speed+heading into a 2D nav-frame trajectory.

    Args:
        speed: (batch, T) forward speed, m/s.
        heading: (batch, T) heading, rad, same convention (0 = +x/east).
        dt: sample period, s.
        p0: (batch, 2) starting position, defaults to origin.

    Returns:
        positions: (batch, T, 2) [x, y] at each timestep.
    """
    batch = speed.shape[0]
    if p0 is None:
        p0 = torch.zeros(batch, 2, device=speed.device, dtype=speed.dtype)

    vx = speed * torch.cos(heading)
    vy = speed * torch.sin(heading)
    dx = vx * dt
    dy = vy * dt
    x = p0[:, 0:1] + torch.cumsum(dx, dim=1)
    y = p0[:, 1:2] + torch.cumsum(dy, dim=1)
    return torch.stack([x, y], dim=-1)


def drift_metric(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> torch.Tensor:
    """% positional drift at the end of each sequence, relative to distance travelled.

    This mirrors the PS benchmark directly: "positional drift < 10% of
    distance travelled during GNSS blackout".

    Args:
        pred_pos: (batch, T, 2)
        gt_pos:   (batch, T, 2)

    Returns:
        (batch,) drift percentage per sequence.
    """
    final_error = torch.linalg.norm(pred_pos[:, -1, :] - gt_pos[:, -1, :], dim=-1)
    step_dist = torch.linalg.norm(gt_pos[:, 1:, :] - gt_pos[:, :-1, :], dim=-1)
    distance_travelled = step_dist.sum(dim=1).clamp(min=1e-6)
    return 100.0 * final_error / distance_travelled


class DeadReckoner:
    """Convenience wrapper bundling calibrated-IMU -> trajectory in one call.

    Kept as a plain class (not nn.Module) since it holds no learnable
    parameters — corrections come from BiasCorrectionNet and are passed in.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def integrate(self, forward_accel: torch.Tensor, yaw_rate: torch.Tensor,
                  v0: torch.Tensor | float = 0.0, theta0: torch.Tensor | float = 0.0,
                  p0: torch.Tensor | None = None):
        speed = integrate_speed(forward_accel, self.dt, v0=v0)
        heading = integrate_heading(yaw_rate, self.dt, theta0=theta0)
        pos = dead_reckon_position(speed, heading, self.dt, p0=p0)
        return speed, heading, pos
