"""
High-rate IMU -> the model's 10 Hz input rate.

The network was trained on 10 Hz samples (configs/default.yaml), so a
200 Hz FOG/MEMS stream has to come down 20x before it reaches the model.
Two ways to do that, both offered:

  "mean" — average every sample that falls in each 100 ms bin.
      This is the anti-aliased choice: engine and road vibration well above
      5 Hz averages out instead of folding back into the band the model
      sees. It is also the physically exact input for the integrator, since
      the mean acceleration over a bin times dt is exactly the velocity
      change over that bin.

  "interp" (the CLI default) — linearly interpolate the stream at the
      10 Hz tick times. This is what every training loader did (np.interp in
      comma2k19_loader.py and windowing.resample_uniform), so it reproduces
      the training input distribution, aliasing included. On the 58 scored
      comma2k19 segments at native ~100 Hz it gave the model a slightly
      lower median drift than "mean" (edge/README.md has the numbers).
      Prefer "mean" for a sensor with strong vibration above 5 Hz.

Bins sit on an absolute clock (k * dt from the first sample), so irregular
or duplicated timestamps (comma2k19's gyro stream has many) are fine: a bin
just averages whatever arrived in it. A bin that receives nothing repeats
the previous bin's value.
"""
from __future__ import annotations

import numpy as np


class BlockAverager:
    """Streaming "mean" resampler. Feed raw samples with push(); each call
    returns the 10 Hz bins that closed, as a list of (t_bin_end, mean)."""

    def __init__(self, out_hz: float, channels: int):
        self.dt = 1.0 / out_hz
        self.channels = channels
        self.t0: float | None = None
        self.bin_idx = 0
        self._sum = np.zeros(channels)
        self._count = 0
        self._last = np.zeros(channels)

    def push(self, t: float, x: np.ndarray) -> list[tuple[float, np.ndarray]]:
        if self.t0 is None:
            self.t0 = t
        out = []
        k = int(np.floor((t - self.t0) / self.dt + 1e-9))
        while k > self.bin_idx:
            out.append(self._close())
        self._sum += x
        self._count += 1
        return out

    def flush(self) -> list[tuple[float, np.ndarray]]:
        return [self._close()] if self._count else []

    def _close(self) -> tuple[float, np.ndarray]:
        if self._count:
            self._last = self._sum / self._count
        t_end = self.t0 + (self.bin_idx + 1) * self.dt
        self.bin_idx += 1
        self._sum = np.zeros(self.channels)
        self._count = 0
        return t_end, self._last.copy()


class PointSampler:
    """Streaming "interp" resampler, same push()/flush() interface: emits the
    stream linearly interpolated at each tick time t0 + k*dt (k >= 1)."""

    def __init__(self, out_hz: float, channels: int):
        self.dt = 1.0 / out_hz
        self.t0: float | None = None
        self.k = 1
        self._prev: tuple[float, np.ndarray] | None = None

    def push(self, t: float, x: np.ndarray) -> list[tuple[float, np.ndarray]]:
        if self.t0 is None:
            self.t0 = t
        out = []
        while self._prev is not None and self.t0 + self.k * self.dt <= t:
            tk = self.t0 + self.k * self.dt
            tp, xp = self._prev
            w = 0.0 if t == tp else (tk - tp) / (t - tp)
            out.append((tk, xp + w * (x - xp)))
            self.k += 1
        if self._prev is None or t > self._prev[0]:
            self._prev = (t, np.asarray(x, dtype=np.float64).copy())
        return out

    def flush(self) -> list[tuple[float, np.ndarray]]:
        return []


def make_resampler(method: str, out_hz: float, channels: int):
    if method == "mean":
        return BlockAverager(out_hz, channels)
    if method == "interp":
        return PointSampler(out_hz, channels)
    raise ValueError(f"unknown resample method {method!r}")


def resample(t: np.ndarray, x: np.ndarray, out_hz: float, method: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Batch version: (N,) times + (N, C) samples -> (M,) tick times + (M, C)."""
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    rs = make_resampler(method, out_hz, x.shape[1])
    rows = []
    for ti, xi in zip(t, x):
        rows.extend(rs.push(float(ti), xi))
    rows.extend(rs.flush())
    return np.array([r[0] for r in rows]), np.stack([r[1] for r in rows])
