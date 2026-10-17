"""
EdgeEngine's high-rate (per IMU sample) output with timestamp jitter large
enough that some samples arrive out of order. A late sample must not wind
the propagation clock back, or the next sample integrates the same stretch
of time again.
"""
from __future__ import annotations

import numpy as np

from edge.engine import EdgeEngine, Fix, ZeroPredictor


def _live_at(t_imu, t_stop, wz=0.1):
    eng = EdgeEngine(ZeroPredictor(), np.eye(3), resample="interp")
    eng.push_gnss(Fix(0.0, 0.0, 0.0, 10.0, 0.0))  # one fix, then a blackout from 1.5 s
    out = None
    for ti in t_imu:
        st = eng.push_imu(ti, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, wz]))
        if ti < t_stop:
            out = st
    return out


def test_out_of_order_samples_do_not_double_integrate():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.005) + rng.uniform(-0.004, 0.004, 4000)
    assert (np.diff(t) < 0).sum() > 100  # really out of order
    # 6.4 s is inside the second 5 s blackout chunk, so the live state has
    # been propagating sample by sample since the first chunk resolved.
    ordered, jittered = _live_at(np.sort(t), 6.4), _live_at(t, 6.4)
    assert ordered.t == jittered.t
    assert abs(jittered.heading - ordered.heading) < 1e-3, (ordered.heading, jittered.heading)
    assert abs(jittered.x - ordered.x) < 0.05, (ordered.x, jittered.x)
