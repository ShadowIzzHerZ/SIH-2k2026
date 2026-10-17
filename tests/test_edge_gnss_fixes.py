"""
EdgeEngine with a GNSS receiver that does not tick in step with the 10 Hz
model clock: fixes that land between a model tick and the raw IMU sample that
closes it, and position-only receivers slower than 10 Hz.
"""
from __future__ import annotations

import collections

import numpy as np

from edge.engine import BLACKOUT, EdgeEngine, Fix, ZeroPredictor


def _run(t_imu, t_gnss, make_fix, resample="interp", accel=(0.0, 0.0, 9.81)):
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample=resample)
    gi = 0
    for ti in t_imu:
        while gi < len(t_gnss) and t_gnss[gi] <= ti:
            eng.push_gnss(make_fix(t_gnss[gi]))
            gi += 1
        eng.push_imu(ti, np.array(accel), np.zeros(3))
    return eng.finish()


def test_no_spurious_blackout_from_fix_newer_than_tick():
    """200 Hz IMU with timestamp jitter, 10 Hz GNSS on a slightly different
    clock, no outage at all. A fix stamped between a tick and the raw sample
    that emits it used to replace the fix that tick needed, so the tick saw
    a "future" fix, called it stale and ran a one-tick blackout plus a 2 s
    blend."""
    rng = np.random.default_rng(0)
    t_imu = np.arange(0, 30, 0.005) + rng.uniform(0, 0.002, 6000)
    t_gnss = np.arange(0.0, 30, 0.1003)
    for method in ("interp", "mean"):
        ticks = _run(t_imu, t_gnss, lambda tg: Fix(tg, 10 * tg, 0.0, 10.0, 0.0), resample=method)
        modes = collections.Counter(s.mode for s in ticks)
        assert set(modes) == {"gnss"}, (method, modes)


def test_fix_lookup_after_imu_gap():
    """An IMU dropout while GNSS keeps arriving: the ticks filled in across
    the gap must use the fixes from the gap, not be called blackout because
    only the newest fix was kept."""
    t_imu = np.concatenate([np.arange(0, 5, 0.005), np.arange(8, 12, 0.005)])
    t_gnss = np.arange(0.0, 12, 0.1)
    ticks = _run(t_imu, t_gnss, lambda tg: Fix(tg, 10 * tg, 0.0, 10.0, 0.0))
    assert not [s for s in ticks if s.mode == BLACKOUT]


def test_position_only_slow_receiver_speed():
    """1 Hz lat/lon-only GNSS at a steady 10 m/s, then a blackout. Speed was
    finite-differenced per 10 Hz tick against a fix that only changes once a
    second (0 m/s for nine ticks, 100 m/s on the tenth), so the blackout
    anchored on 0 or 100 m/s."""
    t_imu = np.arange(0, 40, 0.005)
    t_gnss = np.arange(0, 20.01, 1.0)
    ticks = _run(t_imu, t_gnss, lambda tg: Fix(tg, 10 * tg, 0.0))
    gnss_speeds = np.array([s.speed for s in ticks if 3 < s.t < 20 and s.mode == "gnss"])
    assert np.allclose(gnss_speeds, 10.0, atol=0.5), sorted(set(np.round(gnss_speeds, 1)))
    bo = [s for s in ticks if s.mode == BLACKOUT]
    assert abs(bo[0].speed - 10.0) < 0.5, bo[0].speed
    end = bo[-1]
    assert abs(end.x - 10 * end.t) < 0.02 * 10 * (end.t - 20), (end.t, end.x)
