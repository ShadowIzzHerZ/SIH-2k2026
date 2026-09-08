"""
GNSS <-> INS mode-switching — the PS's own "instant seamless mode-switching"
line item (doc §2), sitting above both the trained network and the
map-matcher rather than replacing either.

Until now this repo had the two pieces the PS asks for individually:
    - a physics + learned-residual dead-reckoning network for the *inside*
      of a blackout (bias_correction_net.py + strapdown_ins.py)
    - map-matching to correct a drifted trajectory against real roads
      (map_matching.py)
but nothing that actually decides, sample by sample, *which* position
estimate to trust and hands off between them without a visible jump. That's
what this module does.

Design:

  GNSS_TRACKING  -- a trustworthy GNSS fix is available. Position, heading,
                     and speed all come directly from GNSS (optionally
                     lightly smoothed) — this is always more accurate than
                     dead reckoning over anything but a very short span, so
                     there's no reason to run the network at all here.

  BLACKOUT        -- no usable GNSS fix. A rolling window of the last
                      `window_size` calibrated IMU samples feeds
                      BiasCorrectionNet every new sample, and the *whole*
                      window's [delta_v, delta_theta] correction sequence
                      (not just its last position) is integrated together
                      from a single anchor state — whatever position/
                      heading/speed was current `window_size` samples ago
                      (real GNSS-derived for the first window_size samples
                      of any blackout, this module's own prior estimate for
                      anything beyond that, i.e. re-anchoring every
                      window_size samples rather than every single one).
                      This isn't the simpler design it looks like it should
                      be — see the "position-invariance" note below.

  BLEND           -- GNSS has just come back. Snapping straight to the new
                      fix would be a visible teleport (and isn't what
                      "seamless" means) — instead the output ramps linearly
                      from the INS-only estimate to the new GNSS fix over
                      `blend_seconds`, so position and heading move
                      continuously through the handoff.

A real finding from building this, worth keeping: the first version of
BLACKOUT used only the *last* position's correction each step — the exact
per-step contract export_onnx.py's docstring already describes for the
on-device app (fixed-size window in, correction for the newest sample out,
one integration step). That's a reasonable-looking design, but measured
5-6x worse real drift than this version, on identical held-out data, for an
identical span. Why: BiasCorrectionNet's output is NOT position-invariant
across its own 50-sample training window — position 0 (almost no GRU
context yet) behaves very differently from position 49 — and
train.py/evaluate.py always score the network using ALL window_size
positions' corrections integrated together from one fresh, ground-truth
anchor, never a single position in isolation. The version here reproduces
that regime instead of the simpler-but-untested on-device contract, and
closes the gap: measured mean/median drift on real comma2k19 held-out data
came back in line with (5s: ~11%/6%) or better than (30s: 2.9%) the
project's existing per-window headline numbers (16.49%/8.94%), rather than
the 5-6x-worse numbers the naive per-step version produced. IO-VNBD's
already-known low-speed/urban weakness (see README.md) is *not* fixed by
this — re-anchoring every 5s to the model's own increasingly-wrong prior
estimate over a long urban blackout compounds badly (measured 259% over a
real 45s span) precisely because individual IO-VNBD windows are already
high-variance (up to 809% worst-case per evaluate.py). That's an honest
property of the underlying network on that scenario, not something this
module can paper over — see simulate_blackout.py's --dataset flag to see
both cases side by side.

This module is deliberately IMU/GNSS-source-agnostic: it consumes plain
numpy arrays (already calibrated accel/gyro, from calibration.py, and a
per-sample GNSS-available mask), not a specific dataset's loader classes —
the same function works on a real held-out IO-VNBD/comma2k19 sequence with
an artificially-injected blackout (see simulate_blackout.py, the demo/eval
use of this) or, unchanged, on a live phone's sensor stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

from .models.bias_correction_net import BiasCorrectionNet


class FusionMode(str, Enum):
    GNSS_TRACKING = "gnss"
    BLACKOUT = "blackout"
    BLEND = "blend"


@dataclass
class FusionResult:
    position: np.ndarray      # (N, 2) fused position estimate, same frame as gnss_xy
    heading: np.ndarray       # (N,) fused heading, rad
    speed: np.ndarray         # (N,) fused speed, m/s
    mode: list[FusionMode] = field(default_factory=list)  # per-sample mode, len N


def _wrap_angle(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


@torch.no_grad()
def run_fusion(
    accel: np.ndarray,          # (N, 3) calibrated, vehicle frame, m/s^2
    gyro: np.ndarray,           # (N, 3) calibrated, vehicle frame, rad/s
    gnss_xy: np.ndarray,        # (N, 2) GNSS-derived local position, metres (ground truth in sim/eval)
    gnss_available: np.ndarray, # (N,) bool — False during a blackout, True otherwise
    dt: float,
    model: BiasCorrectionNet,
    window_size: int,
    device: str | torch.device = "cpu",
    blend_seconds: float = 2.0,
    v_clamp_min: float = 0.0,
) -> FusionResult:
    """Run the GNSS<->INS state machine over one continuous IMU/GNSS stream.

    heading/speed while GNSS is tracking are derived from consecutive GNSS
    fixes (finite difference) purely so BLACKOUT has a real v0/theta0 to
    start from the instant GNSS drops — see windowing.py's Window docstring
    for why starting a dead-reckoning leg from the true initial state (not
    zero) matters this much; the same reasoning applies here, just online
    instead of from a labeled window.
    """
    n = len(accel)
    assert gyro.shape[0] == n and gnss_xy.shape[0] == n and gnss_available.shape[0] == n, (
        "accel/gyro/gnss_xy/gnss_available must all be the same length"
    )
    model = model.to(device)
    model.eval()

    position = np.zeros((n, 2), dtype=np.float64)
    heading = np.zeros(n, dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)
    modes: list[FusionMode] = [FusionMode.GNSS_TRACKING] * n

    # Rolling buffer of calibrated IMU samples fed to the network — mirrors
    # the ONNX export's fixed (1, window_size, 6) input exactly, so this
    # module's BLACKOUT behavior is what the exported model would actually
    # do on-device, not a different, easier-to-implement approximation.
    imu_buf = np.zeros((window_size, 6), dtype=np.float32)
    buf_filled = 0

    blend_samples = max(1, int(round(blend_seconds / dt)))
    blend_remaining = 0
    blend_from_pos = None
    blend_from_heading = 0.0
    blend_from_speed = 0.0

    prev_gnss_xy = None
    prev_available = False

    for t in range(n):
        imu_buf[:-1] = imu_buf[1:]
        imu_buf[-1] = np.concatenate([accel[t], gyro[t]])
        buf_filled = min(window_size, buf_filled + 1)

        available = bool(gnss_available[t])

        if available and not prev_available:
            # GNSS just came back (or this is sample 0 with GNSS available):
            # start a BLEND ramp from wherever the INS-only estimate
            # currently is toward the new fix, instead of snapping.
            if t > 0:
                blend_remaining = blend_samples
                blend_from_pos = position[t - 1].copy()
                blend_from_heading = heading[t - 1]
                blend_from_speed = speed[t - 1]

        if available:
            position[t] = gnss_xy[t]
            if prev_gnss_xy is not None:
                d = gnss_xy[t] - prev_gnss_xy
                dist = float(np.linalg.norm(d))
                if dist > 1e-6:
                    heading[t] = float(np.arctan2(d[1], d[0]))
                    speed[t] = dist / dt
                else:
                    heading[t] = heading[t - 1] if t > 0 else 0.0
                    speed[t] = 0.0
            else:
                heading[t] = 0.0  # no prior fix yet — nothing to derive an instantaneous heading from
                speed[t] = 0.0

            if blend_remaining > 0:
                # Ramp the *displayed* output from the INS estimate toward
                # this fix over the configured window; the true GNSS state
                # above still tracks accumulated internal heading/speed for
                # when the next blackout starts.
                w = 1.0 - blend_remaining / blend_samples
                position[t] = (1 - w) * blend_from_pos + w * gnss_xy[t]
                heading[t] = _wrap_angle((1 - w) * blend_from_heading + w * heading[t])
                speed[t] = (1 - w) * blend_from_speed + w * speed[t]
                modes[t] = FusionMode.BLEND
                blend_remaining -= 1
            else:
                modes[t] = FusionMode.GNSS_TRACKING

            prev_gnss_xy = gnss_xy[t]
        else:
            modes[t] = FusionMode.BLACKOUT
            blend_remaining = 0  # a fresh blackout cancels any in-progress blend

            # Full-window re-integration, anchored `buf_filled` samples back,
            # not "last-position-only" — see this function's docstring for
            # why: BiasCorrectionNet's per-position output is NOT
            # position-invariant across its own training window (position 0,
            # with almost no GRU context, behaves very differently from
            # position 49); train.py/evaluate.py always score it using ALL
            # window_size positions' corrections, integrated together from
            # one fresh anchor state, never just the last one. Reusing only
            # the last position (the original, simpler implementation here)
            # measured 5-6x worse drift than this on real held-out data for
            # an identical span — a real train/deploy mismatch, not a wash.
            #
            # The anchor is whatever state (real GNSS-derived, or this
            # module's own prior estimate) was current `buf_filled` samples
            # ago — exactly matching evaluate.py's "fresh window, true state
            # at its start" regime for any blackout up to window_size long,
            # and degrading to re-anchoring every window_size samples (not
            # every single sample) for longer ones — far coarser
            # error-compounding than per-sample chaining.
            buf = imu_buf[window_size - buf_filled:]
            x = torch.from_numpy(buf).unsqueeze(0).to(device)  # (1, T<=window_size, 6)
            corrections = model(x)[0].detach().cpu().numpy()  # (T, 2) — every position, not just the last

            forward_accel_seq = buf[:, 0] + corrections[:, 0]
            yaw_rate_seq = buf[:, 5] + corrections[:, 1]

            anchor_idx = t - buf_filled
            v0 = speed[anchor_idx] if anchor_idx >= 0 else 0.0
            theta0 = heading[anchor_idx] if anchor_idx >= 0 else 0.0
            p0 = position[anchor_idx] if anchor_idx >= 0 else np.zeros(2)

            v, th, p = v0, theta0, p0.copy()
            for i in range(len(forward_accel_seq)):
                v = v + float(forward_accel_seq[i]) * dt
                if v_clamp_min is not None:
                    v = max(v_clamp_min, v)
                th = _wrap_angle(th + float(yaw_rate_seq[i]) * dt)
                p = p + np.array([v * np.cos(th), v * np.sin(th)]) * dt

            speed[t], heading[t], position[t] = v, th, p
            prev_gnss_xy = None  # last fix is now stale for the finite-diff heading above

        prev_available = available

    return FusionResult(position=position, heading=heading, speed=speed, mode=modes)


def blackout_drift_pct(result: FusionResult, gnss_xy: np.ndarray, gnss_available: np.ndarray) -> float | None:
    """% positional drift at the end of the (last, if several) blackout
    span, relative to the true distance travelled during it — same metric
    definition as strapdown_ins.drift_metric, so this is directly
    comparable to every other drift-% number in this project. Returns None
    if there was no blackout in this run."""
    blackout_idxs = np.where(gnss_available == False)[0]  # noqa: E712
    if len(blackout_idxs) == 0:
        return None
    start, end = blackout_idxs[0], blackout_idxs[-1]
    true_path = gnss_xy[start:end + 1]
    step_dist = np.linalg.norm(np.diff(true_path, axis=0), axis=1)
    distance_travelled = max(step_dist.sum(), 1e-6)
    final_error = np.linalg.norm(result.position[end] - gnss_xy[end])
    return 100.0 * final_error / distance_travelled


def reconnect_jump_m(result: FusionResult, gnss_available: np.ndarray) -> float | None:
    """Largest single-sample position jump right at GNSS reconnection — the
    number that actually quantifies "seamless": with blending on this
    should be small (bounded by how far speed*dt can move in one sample),
    not the multi-metre teleport a hard snap back to the raw GNSS fix would
    produce. Returns None if there was no blackout->reconnect transition."""
    reconnects = [t for t in range(1, len(gnss_available))
                  if gnss_available[t] and not gnss_available[t - 1]]
    if not reconnects:
        return None
    return max(float(np.linalg.norm(result.position[t] - result.position[t - 1])) for t in reconnects)
