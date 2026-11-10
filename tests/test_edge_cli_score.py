"""
edge.cli.score on numpy 1.x: requirements.txt allows numpy>=1.24, which has
np.trapz but not np.trapezoid (added in 2.0).
"""
from __future__ import annotations

import numpy as np

from edge.cli import score
from edge.engine import State
from edge.io import EdgeRecording


def _rec_and_ticks():
    t_g = np.arange(0, 30, 0.1)
    xy = np.stack([10 * t_g, np.zeros_like(t_g)], axis=1)
    rec = EdgeRecording(t_g, np.zeros((len(t_g), 3)), np.zeros((len(t_g), 3)), t_g, xy,
                        np.full(len(t_g), 10.0), np.full(len(t_g), 90.0), None)
    ticks = [State(t, 10 * t + (1.0 if 10 <= t <= 20 else 0.0), 0.0, 0.0, 10.0, "gnss") for t in t_g]
    return rec, ticks


def test_score_without_np_trapezoid(monkeypatch):
    rec, ticks = _rec_and_ticks()
    want = score(ticks, rec, [(10.0, 20.0)])
    if hasattr(np, "trapezoid"):
        monkeypatch.delattr(np, "trapezoid")
    if not hasattr(np, "trapz"):  # numpy >= 2.4 dropped trapz; nothing to emulate
        monkeypatch.setattr(np, "trapz", lambda y, x: float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2), raising=False)
    got = score(ticks, rec, [(10.0, 20.0)])
    assert got == want
    assert abs(got[0]["distance_m"] - 100.0) < 1.0 and abs(got[0]["end_error_m"] - 1.0) < 1e-6, got
