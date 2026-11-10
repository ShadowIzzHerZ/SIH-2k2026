"""
Offline check of simulate_blackout's plot markers: "GNSS reconnects" goes on
the first sample with GNSS back, not on the last blackout sample.

Run:
    python -m pytest -q tests/test_simulate_blackout_plot.py
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.fusion import FusionMode
from src.simulate_blackout import _plot_track


class _Axes:
    def __init__(self):
        self.markers = {}

    def plot(self, *a, **k):
        pass

    def scatter(self, x, y, **k):
        self.markers[k.get("label")] = (x, y)

    def set_aspect(self, *a, **k):
        pass


def _markers(avail, lo=0, hi=None):
    n = len(avail)
    xy = np.stack([np.arange(n, dtype=float), np.zeros(n)], axis=1)  # x = sample index
    result = SimpleNamespace(position=xy,
                             mode=[FusionMode.GNSS_TRACKING if a else FusionMode.BLACKOUT for a in avail])
    ax = _Axes()
    _plot_track(ax, xy, result, avail, lo, n if hi is None else hi)
    return ax.markers


def test_reconnect_marker_on_first_sample_with_gnss():
    avail = np.ones(30, dtype=bool)
    avail[10:20] = False
    m = _markers(avail)
    assert m["blackout starts"][0] == 10
    assert m["GNSS reconnects"][0] == 20
    assert _markers(avail, lo=5, hi=25)["GNSS reconnects"][0] == 20


def test_no_reconnect_marker_when_blackout_runs_to_the_end():
    avail = np.ones(30, dtype=bool)
    avail[10:] = False
    assert "GNSS reconnects" not in _markers(avail)
    avail[10:] = True
    avail[10:20] = False
    assert "GNSS reconnects" not in _markers(avail, hi=20)  # GNSS is back only past the plotted range
