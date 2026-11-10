"""
Offline check of simulate_blackout's span search: a candidate span that
ends exactly at the last sample of the recording must be tried too.

Run:
    python -m pytest -q tests/test_simulate_blackout.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.io_vnbd_loader import ImuSequence
from src.simulate_blackout import _find_clean_span

HZ = 10.0


def _straight_drive(n: int, jump_at: int | None = None) -> ImuSequence:
    """Due-east drive at 10 m/s (1 m per sample), with an optional 500 m
    GPS jump between samples jump_at and jump_at + 1."""
    east_m = np.arange(n, dtype=float)
    if jump_at is not None:
        east_m[jump_at + 1:] += 500.0
    lat0 = 31.3
    lon = 75.5 + east_m / (111_320.0 * np.cos(np.radians(lat0)))
    zeros = np.zeros((n, 3))
    return ImuSequence(
        path=Path("synthetic.csv"), time=np.arange(n) / HZ, accel=zeros, gyro=zeros,
        speed_gt=np.full(n, 10.0), lat=np.full(n, lat0), lon=lon, heading_gt=None,
    )


def _span(seq: ImuSequence) -> int | None:
    # lead-in 2 s + blackout 3 s + tail 1 s = 60 samples; candidate starts
    # every 15 s (150 samples).
    return _find_clean_span(seq, HZ, lead_in_s=2.0, blackout_duration_s=3.0, tail_s=1.0,
                            max_step_m=15.0, min_blackout_distance_m=10.0)


def test_recording_exactly_one_span_long_is_usable():
    assert _span(_straight_drive(60)) == 20


def test_last_exact_fit_start_is_tried():
    # The only start without the GPS jump is 150, whose span ends on the
    # last sample (150 + 60 == 210).
    assert _span(_straight_drive(210, jump_at=10)) == 170


def test_too_short_recording_is_rejected():
    assert _span(_straight_drive(59)) is None
