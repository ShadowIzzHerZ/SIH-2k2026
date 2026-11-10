"""blackout_drift_pct scores only the last blackout when there are several.

Run:
    python -m pytest tests/test_fusion_drift_span.py
"""
from __future__ import annotations

import numpy as np

from src.fusion import FusionResult, blackout_drift_pct


def _straight_line(n: int, step_m: float = 1.0) -> np.ndarray:
    return np.stack([np.arange(n) * step_m, np.zeros(n)], axis=1)


def test_single_blackout_distance_is_its_own_span():
    gt = _straight_line(100)
    avail = np.ones(100, dtype=bool)
    avail[40:60] = False                     # 19 m of travel inside the span
    est = gt.copy()
    est[59] += [0.0, 1.9]                    # 1.9 m off at the end -> 10 %
    res = FusionResult(position=est, heading=np.zeros(100), speed=np.zeros(100))
    assert abs(blackout_drift_pct(res, gt, avail) - 10.0) < 1e-9


def test_two_blackouts_scores_the_last_span_only():
    gt = _straight_line(100)
    avail = np.ones(100, dtype=bool)
    avail[10:20] = False
    avail[40:60] = False                     # last span: 19 m
    est = gt.copy()
    est[59] += [0.0, 1.9]
    res = FusionResult(position=est, heading=np.zeros(100), speed=np.zeros(100))
    # Before the fix the distance ran from sample 10 to 59 (49 m) -> 3.9 %.
    assert abs(blackout_drift_pct(res, gt, avail) - 10.0) < 1e-9
