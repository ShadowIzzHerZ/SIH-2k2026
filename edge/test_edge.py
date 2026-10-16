"""
Checks for the edge engine, in the same plain-assert style as tests/.

    python -m edge.test_edge

1. Resampling: a constant survives "mean", a ramp survives "interp", and
   duplicated / irregular timestamps don't break either.
2. Mounting: axis remaps, the upside-down leveling fix, and the GNSS yaw
   fit recovering a known mounting offset.
3. Engine physics: a noiseless 200 Hz drive with a zero-correction model
   comes back through a blackout with near-zero drift, and the high-rate
   output never jumps.
4. Parity with src/fusion.py: same chunks, same anchors, same output, for a
   blackout that is a whole number of chunks. Needs torch; skipped without.
5. The shipped ONNX model loads and has the shape the engine expects.
"""
from __future__ import annotations

import os

import numpy as np

from edge.engine import BLACKOUT, EdgeEngine, Fix, ModelRateFusion, OnnxPredictor, ZeroPredictor
from edge.mounting import estimate_mount, level, rotation_from_axes
from edge.resample import resample


def test_resample():
    t = np.sort(np.concatenate([np.arange(0, 2, 0.005), np.arange(0, 2, 0.05)]))  # 200 Hz + duplicates
    x = np.full((len(t), 2), 3.0)
    _, m = resample(t, x, 10.0, "mean")
    assert np.allclose(m, 3.0)
    ramp = np.stack([t, 2 * t], axis=1)
    tk, r = resample(t, ramp, 10.0, "interp")
    assert np.allclose(r[:, 0], tk) and np.allclose(r[:, 1], 2 * tk)
    # 27 Hz vibration: "mean" averages it out, "interp" aliases it
    t = np.arange(0, 10, 0.005)
    v = np.sin(2 * np.pi * 27 * t)[:, None]
    assert np.abs(resample(t, v, 10.0, "mean")[1]).max() < 0.15
    assert np.abs(resample(t, v, 10.0, "interp")[1]).max() > 0.8


def test_mounting():
    R = rotation_from_axes("y,-x,z")
    assert np.allclose(R @ [0, 1, 0], [1, 0, 0]) and np.allclose(R @ [1, 0, 0], [0, -1, 0])  # fwd = sensor +y, left = sensor -x
    try:
        rotation_from_axes("x,y,-z")
        raise AssertionError("reflection accepted")
    except ValueError:
        pass

    # Upside-down sensor with a little tilt: vehicle x must stay sensor x.
    g = np.array([-0.3, 0.1, -9.8])
    Rl, _ = level(np.tile(g, (20, 1)))
    assert np.allclose(Rl @ g / np.linalg.norm(g), [0, 0, 1], atol=1e-9)
    fwd = Rl @ np.array([1.0, 0, 0])
    assert np.degrees(abs(np.arctan2(fwd[1], fwd[0]))) < 2.0, fwd

    # Yaw fit: a sensor yawed 20 degrees, driving with real accel and turns.
    dt = 0.1
    t = np.arange(0, 120, dt)
    v = 10 + 5 * np.sin(2 * np.pi * t / 30)
    crs_rate = np.radians(8) * np.sin(2 * np.pi * t / 17)  # compass deg/s, clockwise
    crs = np.degrees(np.cumsum(crs_rate) * dt) % 360
    a_vehicle = np.stack([np.gradient(v, dt), -v * crs_rate, np.full_like(t, 9.81)], axis=1)
    psi = np.radians(20)
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    a_sensor = a_vehicle @ Rz  # sensor = Rz.T @ vehicle
    R, info = estimate_mount(a_sensor, dt, len(t), crs, v)  # level on the whole drive: horizontal accel averages out
    assert abs(info["yaw_deg"] - 20) < 1.0, info
    assert np.allclose(R @ a_sensor[50], a_vehicle[50], atol=0.05)


def _drive(imu_hz=200.0, duration=60.0, speed=12.0, yaw_rate=0.05):
    dt = 1.0 / imu_hz
    t = np.arange(0, duration, dt)
    th = yaw_rate * t
    xy = np.cumsum(np.stack([speed * np.cos(th), speed * np.sin(th)], axis=1), axis=0) * dt
    accel = np.tile([0.0, speed * yaw_rate, 9.81], (len(t), 1))
    gyro = np.tile([0.0, 0.0, yaw_rate], (len(t), 1))
    return t, xy, th, accel, gyro


def test_engine_zero_model():
    t, xy, th, accel, gyro = _drive()
    xy = xy + [500.0, -300.0]  # start away from the origin
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp", gnss_timeout_s=0.15)
    live = []
    for i in range(len(t)):
        blackout = 20.0 <= t[i] < 40.0
        if i % 20 == 0 and not blackout:
            eng.push_gnss(Fix(t[i], *xy[i], 12.0, th[i]))
        live.append(eng.push_imu(t[i], accel[i], gyro[i]))
    ticks = eng.finish()
    bo = [s for s in ticks if s.mode == BLACKOUT]
    assert 190 <= len(bo) <= 200, len(bo)
    last = bo[-1]
    true = xy[np.searchsorted(t, last.t)]
    err = np.hypot(last.x - true[0], last.y - true[1])
    assert err / (12.0 * 20) < 0.01, f"drift {err:.2f} m over 240 m"
    pts = np.array([[s.x, s.y] for s in live])  # from the first sample: no placeholder at the origin
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1).max()
    assert step < 3 * 12.0 / 200, f"high-rate output jumped {step:.3f} m in one sample"


def test_imu_gap_with_gnss():
    """An IMU dropout while GNSS keeps arriving is not a blackout. The 10 Hz
    ticks inside the gap are only emitted when the next IMU sample lands, by
    which time newer fixes have been pushed; each tick must still see the
    fix that was current at its own time."""
    t, xy, th, accel, gyro = _drive()
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    tg = np.arange(0.037, t[-1], 0.1)  # receiver not phase-locked to the IMU
    gi = 0
    for i in range(len(t)):
        while gi < len(tg) and tg[gi] <= t[i]:
            j = np.searchsorted(t, tg[gi])
            eng.push_gnss(Fix(tg[gi], *xy[j], 12.0, th[j]))
            gi += 1
        if 30.0 <= t[i] < 30.5:
            continue  # 0.5 s of IMU lost
        eng.push_imu(t[i], accel[i], gyro[i])
    ticks = eng.finish()
    assert eng.core.model_calls == 0, eng.core.model_calls
    assert all(s.mode == "gnss" for s in ticks), sorted({s.mode for s in ticks})


def test_jittered_timestamps():
    """3 ms of timestamp jitter on a 200 Hz stream (timestamps step backwards
    hundreds of times) must not make the high-rate track drift away from
    the 10 Hz one."""
    t, xy, th, accel, gyro = _drive()
    t_rep = t + 0.003 * np.random.default_rng(0).standard_normal(len(t))
    assert (np.diff(t_rep) < 0).sum() > 100
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp", gnss_timeout_s=0.15)
    live = []
    for i in range(len(t)):
        if i % 20 == 0 and not 20.0 <= t[i] < 40.0:
            eng.push_gnss(Fix(t[i], *xy[i], 12.0, th[i]))
        live.append(eng.push_imu(t_rep[i], accel[i], gyro[i]))
    eng.finish()
    last = [s for s in live if s.mode == BLACKOUT][-1]
    true = xy[np.searchsorted(t, last.t)]
    assert np.hypot(last.x - true[0], last.y - true[1]) < 1.5


def test_first_tick_does_not_jump():
    """Before the first 10 Hz tick the live output shows the fix extrapolated
    with its Doppler speed. When Doppler and positions disagree (IO-VNBD's
    phone GPS often does), the first tick must fade in like every later one
    instead of jumping."""
    t, xy, th, accel, gyro = _drive()
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    live = []
    for i in range(len(t)):
        if i % 20 == 0:
            eng.push_gnss(Fix(t[i], *xy[i], 24.0, th[i]))  # Doppler twice the true speed
        live.append(eng.push_imu(t[i], accel[i], gyro[i]))
    pts = np.array([[s.x, s.y] for s in live[:200]])
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1).max()
    assert step < 3 * 24.0 / 200, f"live output jumped {step:.3f} m at the first tick"


def test_clock_reset():
    """The sensor clock restarts at 0 halfway through (a logger restart).
    The engine must notice, keep its position and carry on, instead of
    ignoring every sample until the new clock passes the old one."""
    t, xy, th, accel, gyro = _drive()
    t_rep = np.where(t < 30.0, t, t - 30.0)
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp", gnss_timeout_s=0.15)
    live = []
    for i in range(len(t)):
        if i % 20 == 0 and not 40.0 <= t[i] < 50.0:  # blackout after the restart
            eng.push_gnss(Fix(t_rep[i], *xy[i], 12.0, th[i]))
        live.append(eng.push_imu(t_rep[i], accel[i], gyro[i]))
    ticks = eng.finish()
    assert eng.clock_resets == 1
    assert len(ticks) > 590, len(ticks)
    bo = [s for s in ticks if s.mode == BLACKOUT]
    assert 95 <= len(bo) <= 100, len(bo)
    true = xy[np.searchsorted(t_rep[t >= 30.0], bo[-1].t) + int((t < 30.0).sum())]
    assert np.hypot(bo[-1].x - true[0], bo[-1].y - true[1]) < 1.5
    pts = np.array([[s.x, s.y] for s in live])
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).max() < 3 * 12.0 / 200


class _BiasedPredictor:
    """Adds a constant 2 m/s^2 forward correction: each 5 s chunk lands ~25 m
    away from the raw strapdown the live output was showing."""
    window_size = 50

    def __call__(self, w):
        return np.tile([2.0, 0.0], (len(w), 1))


def test_live_fade_rate():
    t, xy, th, accel, gyro = _drive()
    eng = EdgeEngine(_BiasedPredictor(), np.eye(3), resample="interp", gnss_timeout_s=0.15)
    live = []
    for i in range(len(t)):
        if i % 20 == 0 and not 20.0 <= t[i] < 40.0:
            eng.push_gnss(Fix(t[i], *xy[i], 12.0, th[i]))
        live.append(eng.push_imu(t[i], accel[i], gyro[i]))
    pts = np.array([[s.x, s.y, s.speed] for s in live if 20.0 <= s.t < 40.0])
    v_shown = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1) * 200
    excess = v_shown - pts[1:, 2]
    assert excess.max() < 5.0 + 1.0, f"live output ran {excess.max():.1f} m/s faster than its own speed"


class _LinearPredictor:
    """Deterministic stand-in for the network: corrections depend on the input
    and on position in the window, like the real one."""
    window_size = 50

    def __call__(self, w):
        pos = np.arange(len(w))[:, None] / len(w)
        return np.concatenate([0.05 * w[:, 1:2] - 0.02 * pos, 0.1 * w[:, 3:4] + 0.001 * pos], axis=1)


def test_parity_with_src_fusion():
    try:
        import torch
        from src.fusion import run_fusion
    except ImportError:
        print("  skipped: torch not installed")
        return

    class Stub(torch.nn.Module):
        def forward(self, x):
            return torch.from_numpy(_LinearPredictor()(x[0].numpy()).astype(np.float32))[None]

    rng = np.random.default_rng(1)
    n, dt = 400, 0.1
    accel = rng.normal(0, 0.3, (n, 3))
    accel[:, 2] += 9.81
    gyro = rng.normal(0, 0.02, (n, 3))
    xy = np.cumsum(rng.normal(1.0, 0.1, (n, 2)), axis=0)
    spd = rng.uniform(5, 10, n)
    hdg = rng.uniform(-np.pi, np.pi, n)
    avail = np.ones(n, dtype=bool)
    avail[100:250] = False  # three whole chunks
    ref = run_fusion(accel, gyro, xy, avail, dt, Stub(), 50, gnss_speed=spd, gnss_heading=hdg)

    core = ModelRateFusion(_LinearPredictor(), dt=dt)
    ticks = []
    for i in range(n):
        fix = (xy[i, 0], xy[i, 1], spd[i], hdg[i]) if avail[i] else None
        ticks += core.tick(i * dt, np.concatenate([accel[i], gyro[i]]), fix)
    ticks += core.flush()
    got = np.array([[s.x, s.y] for s in ticks])
    assert np.allclose(got, ref.position, atol=1e-3), np.abs(got - ref.position).max()
    assert [s.mode for s in ticks] == [m.value for m in ref.mode]


def test_onnx_model():
    path = "checkpoints/dead_reckoning_model.onnx"
    if not os.path.exists(path):
        print(f"  skipped: {path} missing")
        return
    p = OnnxPredictor(path)
    assert p.window_size == 50
    out = p(np.tile([0, 0, 9.81, 0, 0, 0], (50, 1)))
    assert out.shape == (50, 2) and np.isfinite(out).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all edge checks passed")
