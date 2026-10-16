"""
Synthetic high-rate IMU drive, written in the edge CSV format.

For exercising the engine end to end without hardware: a known trajectory
(pull away, cruise, a few turns, a stop), simulated IMU samples at any rate
with a chosen sensor grade, an optional mounting rotation, and a 10 Hz GNSS
receiver. Because the true path is known exactly, it checks the plumbing:
units, mounting, resampling, the state machine and drift scoring.

It is not a benchmark for the network. The model learned real vehicle IMU
behaviour (comma2k19 / IO-VNBD), and a clean synthetic signal is out of
distribution for it. Use the comma2k19 converter (edge/comma2k19.py) for a
real-sensor check.
"""
from __future__ import annotations

import numpy as np

from .io import math_to_compass, write_csv, xy_to_latlon

# accel bias m/s^2, accel noise m/s^2/sqrt(Hz), gyro bias rad/s, gyro noise rad/s/sqrt(Hz)
GRADES = {
    "fog": (0.002, 0.0005, np.radians(0.5 / 3600), np.radians(0.01) / 60),   # tactical FOG + servo accels
    "mems": (0.05, 0.002, np.radians(10 / 3600), np.radians(0.2) / 60),      # automotive MEMS
}


def speed_and_yaw_rate(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise drive: accelerate to 15 m/s, cruise with three turns, brake to a stop."""
    T = t[-1]
    v = np.interp(t, [0, 5, 15, T - 15, T - 5, T], [0, 0, 15, 15, 0, 0])
    w = np.zeros_like(t)
    for start, dur, rate in [(25, 6, np.radians(15)), (45, 10, -np.radians(9)), (70, 5, np.radians(18))]:
        seg = (t >= start) & (t < start + dur)
        w[seg] = rate * np.sin(np.pi * (t[seg] - start) / dur) ** 2
    return v, w


def generate(path: str, duration_s: float = 100.0, imu_hz: float = 200.0, gnss_hz: float = 10.0,
             grade: str = "fog", vibration: float = 0.3, mount_axes: np.ndarray | None = None,
             origin=(28.6139, 77.2090), heading0_deg: float = 30.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / imu_hz
    t = np.arange(0.0, duration_s, dt)
    v, w = speed_and_yaw_rate(t)
    theta = np.radians(90 - heading0_deg) + np.cumsum(w) * dt
    xy = np.cumsum(np.stack([v * np.cos(theta), v * np.sin(theta)], axis=1), axis=0) * dt

    a_fwd = np.gradient(v, dt)
    a_lat = v * w
    accel_v = np.stack([a_fwd, a_lat, np.full_like(t, 9.80665)], axis=1)
    gyro_v = np.stack([np.zeros_like(t), np.zeros_like(t), w], axis=1)

    ab, an, gb, gn = GRADES[grade]
    accel_v += ab * rng.standard_normal(3) + an * np.sqrt(imu_hz) * rng.standard_normal(accel_v.shape)
    gyro_v += gb * rng.standard_normal(3) + gn * np.sqrt(imu_hz) * rng.standard_normal(gyro_v.shape)
    # Engine/road vibration while moving, well above the 5 Hz Nyquist of a
    # 10 Hz model input: what "mean" resampling averages out and "interp" aliases.
    moving = (v > 0.5).astype(float)[:, None]
    accel_v += vibration * moving * np.sin(2 * np.pi * 27.0 * t[:, None] + rng.uniform(0, 2 * np.pi, 3)[None, :])

    R = np.eye(3) if mount_axes is None else mount_axes  # vehicle = R @ sensor
    accel_s, gyro_s = accel_v @ R, gyro_v @ R              # sensor = R.T @ vehicle

    lat, lon = xy_to_latlon(xy, *origin)
    step = int(round(imu_hz / gnss_hz))
    rows = []
    for i in range(len(t)):
        row = [f"{t[i]:.5f}", *(f"{a:.6f}" for a in accel_s[i]), *(f"{g:.7f}" for g in gyro_s[i])]
        if i % step == 0:
            nx, ny = rng.normal(0, 1.0, 2)
            la, lo = xy_to_latlon(xy[i] + [nx, ny], *origin)
            rows.append(row + [f"{la:.8f}", f"{lo:.8f}", f"{max(0, v[i] + rng.normal(0, 0.05)):.3f}",
                               f"{math_to_compass(theta[i]) + rng.normal(0, 0.3) * (v[i] > 1):.2f}"])
        else:
            rows.append(row + ["", "", "", ""])
    write_csv(path, ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed", "heading"], rows)
    return t, xy
