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

  BLACKOUT        -- no usable GNSS fix. Processed in discrete,
                      non-overlapping `window_size`-sample chunks (5s at
                      10Hz) — one BiasCorrectionNet call per chunk, on that
                      chunk's own raw samples, and the *whole* chunk's
                      [delta_v, delta_theta] correction sequence (not just
                      its last position) integrated together from a single
                      anchor state: whatever position/heading/speed was
                      current at the end of the previous chunk (real
                      GNSS-derived for the first chunk of any blackout,
                      this module's own prior estimate for every chunk
                      after that). Deliberately NOT a fresh model call at
                      every single sample re-anchored `window_size` samples
                      back — an earlier version did exactly that and it
                      looked like it re-anchored every 5s but actually still
                      compounded every single sample against a lagging
                      reference, confirmed on real data as a smooth
                      continuous speed creep instead of the discrete
                      per-chunk steps this design intends. See the
                      "position-invariance" note below for why the whole
                      chunk's corrections are used, not just the last.

  BLEND           -- GNSS has just come back. Snapping straight to the new
                      fix would be a visible teleport (and isn't what
                      "seamless" means) — instead the output ramps linearly
                      from the INS-only estimate to the new GNSS fix over
                      `blend_seconds`, so position and heading move
                      continuously through the handoff.

Three real findings from building this, worth keeping all three fixes:

1. The first version of BLACKOUT used only the *last* position's
   correction each step — the exact per-step contract export_onnx.py's
   docstring already describes for the on-device app. Measured 5-6x worse
   real drift than the whole-chunk version here, on identical held-out
   data, for an identical span, because BiasCorrectionNet's output is NOT
   position-invariant across its own 50-sample training window (position 0,
   almost no GRU context yet, behaves very differently from position 49),
   and train.py/evaluate.py always score it using ALL window_size
   positions' corrections integrated together from one fresh, ground-truth
   anchor, never a single position alone. Fixed by reproducing that regime.

2. Seeding speed/heading from finite-differencing consecutive GNSS
   positions (a natural first instinct — the only thing directly
   available from a raw position stream) is a known-bad practice this
   project had already documented and fixed for *yaw* calibration
   (calibration.py's calibrate_sequence docstring), but it crept back in
   here for speed/heading. At real IO-VNBD low-speed driving (~3 m/s),
   confirmed it directly: two consecutive noisy position fixes produced a
   103 m/s speed spike from position quantization/noise alone — nothing to
   do with the network. Fixed by using the dataset's own GPS-chip-reported
   speed_gt/heading_gt fields (Doppler-derived, far more reliable at low
   speed) via the gnss_speed/gnss_heading parameters, falling back to
   finite-difference only when they're unavailable.

3. With neither of the above bugs, a bad chunk's speed estimate could still
   run away with nothing physically stopping it (confirmed reaching 150+
   m/s / 540+ km/h before v_clamp_max existed). Fixed with a physically-sane
   upper speed clamp (default 50 m/s / 180 km/h) — standard practice in any
   real INS.

None of the three fixes "solves" IO-VNBD's already-known low-speed/urban
weakness (see README.md) — nor should they; that's a real property of the
trained network on that scenario, not a fusion-layer bug. Measured across 8
different real 45s continuous blackouts on the same IO-VNBD trace (all
three fixes applied): 77-236% drift, mean 145%, median 109% — genuinely
worse than the per-window headline (60.92%), because chaining ~9
un-reset 5s hops compounds an already high-variance per-window model (up to
809% worst-case per evaluate.py) further than any single window shows.
comma2k19 (highway, the model's genuinely strong case) is a different
story: 5s spans average 10.7%/5.7% mean/median across 20 real segments, and
a single continuous 30s run measured 2.9% — in line with or better than the
16.49%/8.94% per-window headline, not worse. See simulate_blackout.py's
--dataset flag to reproduce either.

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
    v_clamp_max: float = 50.0,
    gnss_speed: np.ndarray | None = None,    # (N,) m/s — GPS-chip-reported speed (Doppler), NOT derived here
    gnss_heading: np.ndarray | None = None,  # (N,) rad — GPS-chip-reported course-over-ground, same convention as gnss_xy
) -> FusionResult:
    """Run the GNSS<->INS state machine over one continuous IMU/GNSS stream.

    gnss_speed/gnss_heading (strongly recommended when available — see
    below) seed speed/heading while GNSS is tracking, so BLACKOUT has a
    real v0/theta0 to start from the instant GNSS drops — see
    windowing.py's Window docstring for why starting a dead-reckoning leg
    from the true initial state (not zero) matters this much.

    If not provided, this falls back to finite-differencing consecutive
    gnss_xy fixes — which calibration.py's calibrate_sequence docstring
    already documents as unreliable ("a single 0.1s GPS position delta at
    10Hz is the same order of magnitude as consumer GPS position noise")
    and which this module's own testing confirmed the hard way: at ~3 m/s
    real speed, two consecutive noisy position fixes produced a 103 m/s
    speed spike from position quantization/noise alone, which then
    anchored an entire blackout leg and dominated its drift-% far more
    than anything about the network's own accuracy. Real GNSS chips report
    Doppler-derived speed/course directly (IO-VNBD's speed_gt/heading_gt,
    comma2k19's speed_gt/heading_gt — both already available from their
    loaders) and are far more reliable at low speed specifically, where
    position noise and real displacement are the same order of magnitude.
    Only skip these if truly unavailable (e.g. a GNSS source that only
    ever reports position) — the finite-difference fallback is a known
    hazard, not a neutral default.

    v_clamp_max (default 50 m/s = 180 km/h, generous above any real driving
    speed in either dataset) guards against a real failure mode found
    testing this on IO-VNBD's harder low-speed/urban windows: once one bad
    5s re-anchor hop overshoots speed, the *next* hop inherits that
    already-absurd value as its own anchor and integrates further from it
    — a positive feedback loop with nothing physically stopping it,
    observed running away to 150+ m/s (540+ km/h) on real data within a
    45s blackout, dominating the resulting drift-% far more than "the
    model is somewhat wrong per 5s window" would on its own. Clamping
    speed to a physically-sane envelope every step — standard practice in
    any real INS, not specific to this model — stops that runaway without
    touching the network or requiring GT anywhere it isn't already used.
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

    blend_samples = max(1, int(round(blend_seconds / dt)))
    blend_remaining = 0
    blend_from_pos = None
    blend_from_heading = 0.0
    blend_from_speed = 0.0

    prev_gnss_xy = None
    prev_available = False

    # Blackout is processed in discrete, non-overlapping window_size chunks
    # (see the loop body below), not one model call per sample — a chunk
    # already ending covers t until this index; skip re-triggering until
    # past it.
    chunk_ends_at = -1

    for t in range(n):
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
            if gnss_speed is not None and gnss_heading is not None:
                # Preferred: the GNSS chip's own Doppler-derived speed/course
                # — see this function's docstring for why the finite-diff
                # fallback below is a known hazard, not just a simpler option.
                speed[t] = float(gnss_speed[t])
                heading[t] = float(gnss_heading[t])
            elif prev_gnss_xy is not None:
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
            prev_gnss_xy = None  # last fix is now stale for the finite-diff heading above

            if t <= chunk_ends_at:
                pass  # already computed as part of an earlier chunk in this same blackout run
            else:
                # A genuine discrete, non-overlapping window_size chunk —
                # NOT a fresh model call re-anchored at every single sample.
                # An earlier version of this function did exactly that (a
                # window ending at t, recomputed every t, anchored
                # `window_size` samples back): it looked like it re-anchored
                # every 5s but actually still compounded every single
                # sample, just against a lagging reference instead of the
                # immediate previous one — confirmed on real IO-VNBD data as
                # a smooth, continuous speed creep (3 -> 50 m/s over a 45s
                # blackout) rather than the discrete per-hop jumps the
                # design was supposed to produce. This version calls the
                # model once per window_size-sample chunk, on that chunk's
                # own samples directly (matching train.py/evaluate.py's
                # actual window construction, not a sliding approximation
                # of it), and uses ALL of its per-position corrections
                # together in one integration from the *previous chunk's*
                # own final state — genuinely re-anchoring only once every
                # window_size samples.
                chunk_len = min(window_size, n - t)
                # Don't let a chunk run past where GNSS comes back — the
                # remainder gets its own (shorter) chunk on the next
                # iteration instead of overshooting into tracked territory.
                for k in range(1, chunk_len):
                    if gnss_available[t + k]:
                        chunk_len = k
                        break
                chunk_ends_at = t + chunk_len - 1

                chunk_accel = accel[t:t + chunk_len]
                chunk_gyro = gyro[t:t + chunk_len]
                x = torch.from_numpy(
                    np.concatenate([chunk_accel, chunk_gyro], axis=1).astype(np.float32)
                ).unsqueeze(0).to(device)  # (1, chunk_len, 6)
                corrections = model(x)[0].detach().cpu().numpy()  # (chunk_len, 2)

                forward_accel_seq = chunk_accel[:, 0] + corrections[:, 0]
                yaw_rate_seq = chunk_gyro[:, 2] + corrections[:, 1]

                anchor_idx = t - 1
                v0 = speed[anchor_idx] if anchor_idx >= 0 else 0.0
                theta0 = heading[anchor_idx] if anchor_idx >= 0 else 0.0
                p0 = position[anchor_idx] if anchor_idx >= 0 else np.zeros(2)

                v, th, p = v0, theta0, p0.copy()
                for i in range(chunk_len):
                    v = v + float(forward_accel_seq[i]) * dt
                    if v_clamp_min is not None:
                        v = max(v_clamp_min, v)
                    if v_clamp_max is not None:
                        v = min(v_clamp_max, v)
                    th = _wrap_angle(th + float(yaw_rate_seq[i]) * dt)
                    p = p + np.array([v * np.cos(th), v * np.sin(th)]) * dt
                    speed[t + i], heading[t + i], position[t + i] = v, th, p

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
