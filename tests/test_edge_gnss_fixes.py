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


def test_late_fix_into_full_buffer():
    """A late fix arriving when the fix buffer is full (no IMU for a while, so
    nothing prunes it). deque.insert raises IndexError on a full bounded
    deque; the oldest fix must be evicted instead, keeping time order."""
    eng = EdgeEngine(ZeroPredictor(), np.eye(3))
    n = eng.fixes.maxlen
    for k in range(1, n + 1):  # a burst, all within clock_reset_s
        eng.push_gnss(Fix(0.002 * k, k, 0.0, 10.0, 0.0))
    late = Fix(0.002 * n - 0.001, 0.0, 0.0, 10.0, 0.0)
    eng.push_gnss(late)
    ts = [f.t for f in eng.fixes]
    assert len(ts) == n and late in eng.fixes
    assert ts == sorted(ts) and ts[0] == 0.004
    eng.push_gnss(Fix(0.001, 0.0, 0.0, 10.0, 0.0))  # older than everything kept
    assert [f.t for f in eng.fixes] == ts
    assert eng.fix.t == 0.002 * n


def test_fix_on_new_clock_survives_clock_reset():
    """The first fix after a sensor clock restart arrives just before the IMU
    sample that reveals the restart. It is not a late fix to sort in among
    the old ones: the clock reset must keep it as the start of the new clock."""
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    for ti in np.arange(0, 10, 0.005):
        if round(ti / 0.005) % 20 == 0:
            eng.push_gnss(Fix(ti, 10 * ti, 0.0, 10.0, 0.0))
        eng.push_imu(ti, np.array([0.0, 0.0, 9.81]), np.zeros(3))
    first = Fix(0.0, 100.0, 0.0, 10.0, 0.0)
    eng.push_gnss(first)
    eng.push_imu(0.0, np.array([0.0, 0.0, 9.81]), np.zeros(3))
    assert eng.clock_resets == 1
    assert list(eng.fixes) == [first]


def test_fix_late_by_more_than_clock_reset_s():
    """A 1 Hz receiver delivers the 9.9 s fix after the 11.0 s one, on the
    same clock. Over clock_reset_s (1 s) late, it used to be taken for a clock
    restart and appended last, so the ticks after 11.0 found it first, called
    it stale from 11.5 s on and ran a spurious blackout until the next fix."""
    t_imu = np.arange(0, 20, 0.005)
    t_gnss = list(np.arange(0.0, 20, 1.0))
    t_gnss.insert(t_gnss.index(11.0) + 1, 9.9)
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    gi = 0
    for ti in t_imu:
        while gi < len(t_gnss) and (t_gnss[gi] <= ti or (gi and t_gnss[gi] < t_gnss[gi - 1])):
            eng.push_gnss(Fix(t_gnss[gi], 10 * t_gnss[gi], 0.0, 10.0, 0.0))
            gi += 1
        eng.push_imu(ti, np.array([0.0, 0.0, 9.81]), np.zeros(3))
    ticks = eng.finish()
    assert eng.clock_resets == 0
    assert not [s.t for s in ticks if s.mode != "gnss"], sorted({round(s.t, 1) for s in ticks if s.mode != "gnss"})


def test_two_fixes_on_new_clock_survive_clock_reset():
    """A clock restart, with two fixes in on the new clock before the IMU
    sample that reveals it. Both are sorted in ahead of the old clock's
    fixes; the reset keeps them, in order, and drops the old ones."""
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    for ti in np.arange(0, 10, 0.005):
        if round(ti / 0.005) % 20 == 0:
            eng.push_gnss(Fix(ti, 10 * ti, 0.0, 10.0, 0.0))
        eng.push_imu(ti, np.array([0.0, 0.0, 9.81]), np.zeros(3))
    new = [Fix(0.0, 100.0, 0.0, 10.0, 0.0), Fix(0.1, 101.0, 0.0, 10.0, 0.0)]
    for f in new:
        eng.push_gnss(f)
    eng.push_imu(0.1, np.array([0.0, 0.0, 9.81]), np.zeros(3))
    assert eng.clock_resets == 1
    assert list(eng.fixes) == new
