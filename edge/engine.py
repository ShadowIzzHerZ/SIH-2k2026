"""
Streaming GNSS/INS engine for a high-rate, non-phone IMU.

Two layers:

  ModelRateFusion — the authoritative estimate, at the model's 10 Hz. Same
      state machine as src/fusion.py's run_fusion (GNSS_TRACKING /
      BLACKOUT / BLEND, discrete non-overlapping 5 s blackout chunks, every
      correction in a chunk integrated from one anchor, the 50 m/s speed
      clamp, duo-model switching at 6 m/s), but causal: a chunk is
      resolved when its last sample arrives instead of looking ahead. For
      a full chunk the model input and the anchor are identical to
      run_fusion's, so the output is too. A short chunk (GNSS back
      mid-chunk) differs in one way that the fixed-shape ONNX graph forces:
      the model sees the last 50 real samples and only the last k
      corrections are used, the same fix the Android FusionEngine.kt uses.

  EdgeEngine — wraps the above for a raw IMU stream at any rate (200 Hz
      FOG, 100 Hz automotive MEMS, ...). Rotates samples into the vehicle
      frame, resamples to 10 Hz (edge/resample.py), feeds ModelRateFusion,
      and emits a position at every raw sample. Between model results the
      high-rate output is propagated with plain strapdown on the raw
      samples; when a 10 Hz result lands, the high-rate track re-anchors to
      it and the difference is faded out over `live_smooth_s` so the
      output never teleports. The 10 Hz track is the one to score; the
      high-rate track is the one to display or hand to a controller.

Inputs are plain numbers in SI units, vehicle frame x forward / y left /
z up, local metres x east / y north, headings in radians counter-clockwise
from east (the convention fusion.py uses). edge/cli.py handles units,
lat/lon and compass headings.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .resample import make_resampler

GNSS, BLACKOUT, BLEND = "gnss", "blackout", "blend"


def _wrap(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


@dataclass
class State:
    t: float
    x: float
    y: float
    heading: float
    speed: float
    mode: str


@dataclass
class Fix:
    t: float
    x: float
    y: float
    speed: float | None = None    # Doppler speed, m/s
    heading: float | None = None  # course over ground, rad CCW from east


class OnnxPredictor:
    """The exported BiasCorrectionNet. (W, 6) window -> (W, 2) corrections."""

    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        if len(inp.shape) != 3 or inp.shape[2] != 6:
            raise ValueError(f"{path}: expected input (1, W, 6), got {inp.shape}")
        self.input_name = inp.name
        self.window_size = int(inp.shape[1])
        self.path = path

    def __call__(self, window: np.ndarray) -> np.ndarray:
        x = window.astype(np.float32)[None]
        return self.sess.run(None, {self.input_name: x})[0][0]


class ZeroPredictor:
    """No learned correction: plain strapdown dead reckoning. The baseline the
    model has to beat."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def __call__(self, window: np.ndarray) -> np.ndarray:
        return np.zeros((len(window), 2))


class ModelRateFusion:
    def __init__(self, predictor, dt: float = 0.1, blend_seconds: float = 2.0,
                 v_clamp_min: float = 0.0, v_clamp_max: float = 50.0,
                 predictor_highway=None, switch_mps: float = 6.0):
        self.predictor = predictor
        self.predictor_highway = predictor_highway
        self.switch_mps = switch_mps
        self.W = predictor.window_size
        if predictor_highway is not None and predictor_highway.window_size != self.W:
            raise ValueError("urban and highway models need the same window size")
        self.dt = dt
        self.blend_samples = max(1, int(round(blend_seconds / dt)))
        self.v_min, self.v_max = v_clamp_min, v_clamp_max

        self.hist: deque[np.ndarray] = deque(maxlen=self.W)
        self.last: State | None = None      # last finalized tick
        self.prev_available = False
        self.prev_fix_xy: np.ndarray | None = None
        self.blend_remaining = 0
        self.blend_from: State | None = None
        self.pending: list[tuple[float, np.ndarray]] = []
        self.anchor: State | None = None
        self.model_calls = 0

    def tick(self, t: float, sample: np.ndarray, fix: tuple[float, float, float | None, float | None] | None) -> list[State]:
        """One 10 Hz sample. sample = [ax, ay, az, gx, gy, gz] (vehicle frame);
        fix = (x, y, speed, heading) or None during a blackout. Returns the
        ticks finalized by this call (none while a chunk is filling, a whole
        chunk when it resolves)."""
        self.hist.append(np.asarray(sample, dtype=np.float64))
        out: list[State] = []
        if fix is None:
            self.blend_remaining = 0
            self.prev_fix_xy = None
            if not self.pending:
                self.anchor = self.last
            self.pending.append((t, self.hist[-1]))
            if len(self.pending) == self.W:
                out += self._resolve()
            self.prev_available = False
            return out

        if self.pending:
            out += self._resolve()  # GNSS back mid-chunk: close the short chunk first
        x, y, spd, hdg = fix
        xy = np.array([x, y], dtype=np.float64)
        if spd is not None and hdg is not None:
            speed, heading = float(spd), float(hdg)
        elif self.prev_fix_xy is not None:
            d = xy - self.prev_fix_xy
            dist = float(np.linalg.norm(d))
            if dist > 1e-6:
                heading, speed = float(np.arctan2(d[1], d[0])), dist / self.dt
            else:
                heading, speed = (self.last.heading if self.last else 0.0), 0.0
        else:
            heading, speed = 0.0, 0.0

        if not self.prev_available and self.last is not None:
            self.blend_remaining = self.blend_samples
            self.blend_from = self.last
        if self.blend_remaining > 0:
            w = 1.0 - self.blend_remaining / self.blend_samples
            b = self.blend_from
            st = State(t, (1 - w) * b.x + w * x, (1 - w) * b.y + w * y,
                       _wrap((1 - w) * b.heading + w * heading), (1 - w) * b.speed + w * speed, BLEND)
            self.blend_remaining -= 1
        else:
            st = State(t, x, y, heading, speed, GNSS)
        self.prev_fix_xy = xy
        self.prev_available = True
        self.last = st
        out.append(st)
        return out

    def flush(self) -> list[State]:
        return self._resolve() if self.pending else []

    def _resolve(self) -> list[State]:
        k = len(self.pending)
        window = np.stack(self.hist)
        if len(window) < self.W:  # stream started inside a blackout: pad with the first sample
            window = np.concatenate([np.repeat(window[:1], self.W - len(window), axis=0), window])
        a = self.anchor or State(self.pending[0][0] - self.dt, 0.0, 0.0, 0.0, 0.0, BLACKOUT)
        active = self.predictor_highway if (self.predictor_highway is not None and a.speed >= self.switch_mps) else self.predictor
        corr = active(window)[-k:]
        self.model_calls += 1

        v, th, px, py = a.speed, a.heading, a.x, a.y
        out = []
        for i, (t, s) in enumerate(self.pending):
            v = v + float(s[0] + corr[i, 0]) * self.dt
            if self.v_min is not None:
                v = max(self.v_min, v)
            if self.v_max is not None:
                v = min(self.v_max, v)
            th = _wrap(th + float(s[5] + corr[i, 1]) * self.dt)
            px += v * np.cos(th) * self.dt
            py += v * np.sin(th) * self.dt
            out.append(State(t, px, py, th, v, BLACKOUT))
        self.pending = []
        self.last = out[-1]
        return out


class EdgeEngine:
    def __init__(self, predictor, R_mount: np.ndarray, model_hz: float = 10.0, resample: str = "mean",
                 gnss_timeout_s: float = 1.5, live_smooth_s: float = 1.0, **fusion_kwargs):
        self.R = np.asarray(R_mount, dtype=np.float64)
        self.core = ModelRateFusion(predictor, dt=1.0 / model_hz, **fusion_kwargs)
        self.resampler = make_resampler(resample, model_hz, 6)
        self.gnss_timeout_s = gnss_timeout_s
        self.live_smooth_s = live_smooth_s
        self.fixes: deque[Fix] = deque(maxlen=256)  # recent fixes, oldest first
        self.ticks: list[State] = []        # finalized 10 Hz track
        self.live: State | None = None      # high-rate propagated state (undisplayed)
        self._offset = np.zeros(2)          # display offset being faded out
        self._offset_t0 = 0.0

    def push_gnss(self, fix: Fix):
        self.fixes.append(fix)

    def _fix_at(self, t: float):
        """The newest fix taken at or before t, extrapolated to t, or None if
        it is stale. Lets a 1-5 Hz GNSS receiver drive the 10 Hz state machine.

        Not simply the newest fix: a 10 Hz tick is only emitted once an IMU
        sample after it arrives, and after an IMU dropout that can be well
        after the tick, when fixes newer than the tick have already been
        pushed. Using those would make the tick look like a blackout and
        start a spurious model chunk and blend."""
        f = next((f for f in reversed(self.fixes) if f.t <= t), None)
        if f is None or t - f.t > self.gnss_timeout_s:
            return None
        x, y = f.x, f.y
        if f.speed is not None and f.heading is not None:
            age = t - f.t
            x += f.speed * age * np.cos(f.heading)
            y += f.speed * age * np.sin(f.heading)
        return x, y, f.speed, f.heading

    def push_imu(self, t: float, accel: np.ndarray, gyro: np.ndarray) -> State:
        """One raw sample, sensor frame, SI units. Returns the high-rate state."""
        a = self.R @ np.asarray(accel, dtype=np.float64)
        g = self.R @ np.asarray(gyro, dtype=np.float64)
        finalized = []
        for t_tick, s in self.resampler.push(t, np.concatenate([a, g])):
            finalized += self.core.tick(t_tick, s, self._fix_at(t_tick))
        if finalized:
            self.ticks += finalized
            self._reanchor(finalized[-1], t)
        return self._propagate(t, a, g)

    def finish(self) -> list[State]:
        for t_tick, s in self.resampler.flush():
            self.ticks += self.core.tick(t_tick, s, self._fix_at(t_tick))
        self.ticks += self.core.flush()
        return self.ticks

    def _reanchor(self, st: State, t_now: float):
        if self.live is not None:
            shown = self._shown(t_now)
            self._offset = np.array([shown.x - st.x, shown.y - st.y])
            self._offset_t0 = t_now
        self.live = State(st.t, st.x, st.y, st.heading, st.speed, st.mode)

    def _propagate(self, t: float, a: np.ndarray, g: np.ndarray) -> State:
        mode = GNSS if self._fix_at(t) is not None else BLACKOUT
        if self.live is None:  # before the first 10 Hz tick
            return State(t, 0.0, 0.0, 0.0, 0.0, mode)
        L = self.live
        dt = max(0.0, t - L.t)
        if mode == BLACKOUT:  # raw strapdown until the next model result
            L.speed = min(self.core.v_max, max(self.core.v_min, L.speed + a[0] * dt))
        L.heading = _wrap(L.heading + g[2] * dt)
        L.x += L.speed * np.cos(L.heading) * dt
        L.y += L.speed * np.sin(L.heading) * dt
        # Host timestamps jitter and can step backwards. Never move the clock
        # back, or the next sample integrates the same interval twice.
        L.t = max(L.t, t)
        if L.mode != BLEND:  # a blend lasts until the next 10 Hz tick says otherwise
            L.mode = mode
        return self._shown(t)

    def _shown(self, t: float) -> State:
        L = self.live
        if self.live_smooth_s > 0:
            w = max(0.0, 1.0 - (t - self._offset_t0) / self.live_smooth_s)
        else:
            w = 0.0
        return State(t, L.x + w * self._offset[0], L.y + w * self._offset[1], L.heading, L.speed, L.mode)
