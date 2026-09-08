"""
Demo/eval for fusion.py on a real, held-out drive: pick a real IO-VNBD
smartphone recording, artificially black out its GNSS for a stretch in the
middle (simulating driving into a tunnel/underground parking/urban canyon),
run the GNSS<->INS mode-switcher through it using the trained
BiasCorrectionNet checkpoint, and report + plot how it did.

This is not a synthetic sanity check (that's tests/test_fusion.py) — it's
the actual demo artifact: "car is driving normally (GNSS-tracked, blue) ->
GNSS drops (red) -> the trained network dead-reckons through the gap,
map-matching-correctable -> GNSS returns and the estimate blends back in
smoothly (no jump) -> back to GNSS-tracked." Saves a trajectory plot plus a
JSON report with the blackout-specific drift-% (same metric definition as
the PS's own <10% target) and the reconnection jump size.

Run:
    python -m src.simulate_blackout --config configs/default.yaml --checkpoint checkpoints/best.pt
    python -m src.simulate_blackout --file "data/IO-VNBD/.../S-xxx.csv" --blackout_start_s 30 --blackout_duration_s 45
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display available on a dev box/CI/Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.io_vnbd_loader import ImuSequence, latlon_to_local_xy, load_sequence
from src.data.windowing import calibrate_sequence, resample_uniform
from src.fusion import FusionMode, blackout_drift_pct, reconnect_jump_m, run_fusion
from src.models.bias_correction_net import BiasCorrectionNet
from src.train import pick_device


def _find_clean_span(seq: ImuSequence, sample_rate_hz: float, lead_in_s: float, blackout_duration_s: float,
                      tail_s: float, max_step_m: float, min_blackout_distance_m: float) -> int | None:
    """Slides a candidate (lead-in + blackout + tail) window across one
    already-loaded/resampled sequence and returns the first blackout-start
    sample index that's usable, or None.

    Real single-sample GPS jumps (tens to thousands of metres — confirmed
    against IO-VNBD directly: median max-jump across a 40-file sample was
    ~154m) show up even in otherwise-fine recordings, so this checks the
    *specific span* the demo will actually use rather than requiring an
    entire (sometimes multi-hour) file to be clean, which rejected every
    IO-VNBD candidate when tried. Also requires real motion during the
    blackout span itself (min_blackout_distance_m) — same reasoning as
    windowing.py's build_windows min_distance_m filter: a parked/idle
    blackout is a degenerate demo (near-zero true distance travelled makes
    the drift-% metric meaningless)."""
    need_n = int((lead_in_s + blackout_duration_s + tail_s) * sample_rate_hz)
    if seq.lat is None or seq.lon is None or seq.n < need_n:
        return None
    slide_n = max(1, int(15.0 * sample_rate_hz))
    lead_in_n = int(lead_in_s * sample_rate_hz)
    blackout_n = int(blackout_duration_s * sample_rate_hz)

    xy = latlon_to_local_xy(seq.lat, seq.lon)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    for window_start in range(0, seq.n - need_n, slide_n):
        window_end = window_start + need_n
        if step[window_start:window_end - 1].max() > max_step_m:
            continue  # a GPS jump/glitch falls inside this candidate span
        bo_start = window_start + lead_in_n
        bo_end = bo_start + blackout_n
        blackout_dist = np.linalg.norm(np.diff(xy[bo_start:bo_end], axis=0), axis=1).sum()
        if blackout_dist < min_blackout_distance_m:
            continue  # essentially parked for the simulated blackout span
        return bo_start
    return None


def pick_demo_file(data_root: str, variant: str, file_prefix: str, column_map: dict, sample_rate_hz: float,
                    lead_in_s: float, blackout_duration_s: float, tail_s: float = 20.0,
                    min_bytes: int = 200_000, max_step_m: float = 15.0,
                    min_blackout_distance_m: float = 30.0) -> tuple[Path, ImuSequence, int]:
    """Picks a real IO-VNBD recording *and* a blackout start offset within
    it that's actually usable for a clean demo (see _find_clean_span).
    Deterministic (sorted candidates, first match, fixed slide order) so
    re-running the demo without --file reproduces the same trace/offset."""
    root = Path(data_root) / variant
    candidates = sorted(
        p for p in glob.glob(str(root / "**" / f"{file_prefix}*.csv"), recursive=True)
        if Path(p).stat().st_size > min_bytes
    )
    if not candidates:
        raise FileNotFoundError(f"No {file_prefix}*.csv over {min_bytes} bytes under {root}")

    tried_files = 0
    for p in candidates:
        tried_files += 1
        try:
            seq = load_sequence(Path(p), column_map)
            seq = resample_uniform(seq, sample_rate_hz)
        except Exception:
            continue
        bo_start = _find_clean_span(seq, sample_rate_hz, lead_in_s, blackout_duration_s, tail_s,
                                     max_step_m, min_blackout_distance_m)
        if bo_start is not None:
            return Path(p), calibrate_sequence(seq), bo_start

    raise RuntimeError(
        f"checked {tried_files} candidate files under {root}, none had a "
        f"{lead_in_s + blackout_duration_s + tail_s:.0f}s span with no GPS jump and real motion "
        f"during the blackout — pass --file (and possibly --blackout_start_s) manually."
    )


def pick_demo_segment_comma2k19(data_dir: str, sample_rate_hz: float, lead_in_s: float, blackout_duration_s: float,
                                 tail_s: float = 20.0, max_step_m: float = 15.0,
                                 min_blackout_distance_m: float = 30.0) -> tuple[str, ImuSequence, int]:
    """Same idea as pick_demo_file, over comma2k19's demo-split segments
    instead of IO-VNBD files — steady highway driving, the scenario this
    model is genuinely strong on (16.49% mean / 8.94% median test drift,
    see README.md), unlike IO-VNBD's harder low-speed/urban case. Useful to
    demo both honestly: the same mode-switcher, on the case it handles well
    and the case it doesn't."""
    from src.data.comma2k19_loader import load_all_segments  # local import: extra deps (pyarrow)

    parquet_paths = sorted(glob.glob(f"{data_dir}/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No comma2k19 parquet files under {data_dir}")
    segments = load_all_segments(parquet_paths, target_hz=sample_rate_hz)

    for seq in segments:
        bo_start = _find_clean_span(seq, sample_rate_hz, lead_in_s, blackout_duration_s, tail_s,
                                     max_step_m, min_blackout_distance_m)
        if bo_start is not None:
            return str(seq.path), calibrate_sequence(seq), bo_start

    raise RuntimeError(f"checked {len(segments)} comma2k19 segments, none had a usable "
                        f"{lead_in_s + blackout_duration_s + tail_s:.0f}s span.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dataset", choices=["iovnbd", "comma2k19"], default="iovnbd",
                         help="iovnbd: the harder low-speed/urban case (~62%% test drift baseline). "
                              "comma2k19: steady highway driving, the case this model is genuinely "
                              "strong on (~16%% mean / ~9%% median test drift, see README.md). "
                              "Ignored if --file is given.")
    parser.add_argument("--file", default=None,
                         help="Specific IO-VNBD S-*.csv to use instead of auto-picking (ignores --dataset).")
    parser.add_argument("--blackout_start_s", type=float, default=None,
                         help="Default: 60s into the usable trace.")
    parser.add_argument("--blackout_duration_s", type=float, default=45.0)
    parser.add_argument("--blend_seconds", type=float, default=2.0)
    parser.add_argument("--out_prefix", default="results/blackout_demo")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["train"]["device"])
    sample_rate_hz = cfg["data"]["sample_rate_hz"]
    dt = 1.0 / sample_rate_hz
    lead_in_s = args.blackout_start_s if args.blackout_start_s is not None else 60.0

    if args.file:
        path = Path(args.file)
        seq = load_sequence(path, cfg["data"]["column_map"])
        seq = resample_uniform(seq, sample_rate_hz)
        if seq.lat is None or seq.lon is None:
            raise ValueError(f"{path}: no lat/lon — can't simulate a GNSS blackout without a position ground truth")
        seq = calibrate_sequence(seq)
        n = seq.n
        blackout_start = int(lead_in_s / dt)
        blackout_end = blackout_start + int(args.blackout_duration_s / dt)
        if blackout_end >= n:
            raise ValueError(
                f"{path}: trace is only {n * dt:.0f}s long — not enough room for a "
                f"{lead_in_s:.0f}s start + {args.blackout_duration_s}s blackout. "
                f"Pass --blackout_start_s/--blackout_duration_s that fit, or a longer --file."
            )
    elif args.dataset == "comma2k19":
        # comma2k19's demo segments are only ~1min each — a much shorter
        # tail than IO-VNBD's multi-minute files need to fit a usable span.
        path, seq, blackout_start = pick_demo_segment_comma2k19(
            "data/comma2k19_demo/data", sample_rate_hz, lead_in_s, args.blackout_duration_s, tail_s=5.0,
        )
        path = Path(path)
        n = seq.n
        blackout_end = blackout_start + int(args.blackout_duration_s / dt)
    else:
        path, seq, blackout_start = pick_demo_file(
            cfg["data"]["root"], cfg["data"]["variant"], cfg["data"].get("file_prefix", ""),
            cfg["data"]["column_map"], sample_rate_hz, lead_in_s, args.blackout_duration_s,
        )
        n = seq.n
        blackout_end = blackout_start + int(args.blackout_duration_s / dt)
    print(f"using trace: {path} ({n / sample_rate_hz:.0f}s)")

    xy = latlon_to_local_xy(seq.lat, seq.lon)

    gnss_available = np.ones(n, dtype=bool)
    gnss_available[blackout_start:blackout_end] = False
    print(f"simulated blackout: samples [{blackout_start}, {blackout_end}) "
          f"= {blackout_start * dt:.0f}s to {blackout_end * dt:.0f}s "
          f"({args.blackout_duration_s:.0f}s) of a {n * dt:.0f}s trace")

    model = BiasCorrectionNet(
        input_channels=cfg["model"]["input_channels"], cnn_channels=cfg["model"]["cnn_channels"],
        cnn_kernel_size=cfg["model"]["cnn_kernel_size"], gru_hidden=cfg["model"]["gru_hidden"],
        gru_layers=cfg["model"]["gru_layers"], dropout=cfg["model"]["dropout"],
        output_dim=cfg["model"]["output_dim"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    result = run_fusion(
        seq.accel, seq.gyro, xy, gnss_available, dt, model,
        window_size=cfg["data"]["window_size"], device=device, blend_seconds=args.blend_seconds,
    )

    drift = blackout_drift_pct(result, xy, gnss_available)
    jump = reconnect_jump_m(result, gnss_available)
    target = cfg["eval"]["drift_target_pct"]

    report = {
        "trace": str(path),
        "trace_duration_s": n * dt,
        "blackout_start_s": blackout_start * dt,
        "blackout_duration_s": args.blackout_duration_s,
        "blackout_drift_pct": drift,
        "drift_target_pct": target,
        "under_target": bool(drift is not None and drift < target),
        "reconnect_jump_m": jump,
    }
    print(json.dumps(report, indent=2))

    Path("results").mkdir(exist_ok=True)
    out_json = f"{args.out_prefix}_report.json"
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"\nsaved report -> {out_json}")

    _plot(xy, result, gnss_available, dt, out_path=f"{args.out_prefix}.png", trace_name=path.name)
    print(f"saved plot -> {args.out_prefix}.png")


def _plot_track(ax, gnss_xy, result, gnss_available, lo, hi):
    """Draws one track (ground truth + mode-colored fused estimate) over
    sample range [lo, hi) onto the given axes."""
    ax.plot(gnss_xy[lo:hi, 0], gnss_xy[lo:hi, 1], color="#9aa5b1", linewidth=1.5, linestyle="--",
            label="ground truth (GNSS)", zorder=1)

    mode_arr = np.array([m.value for m in result.mode[lo:hi]])
    colors = {"gnss": "#2563eb", "blackout": "#dc2626", "blend": "#f59e0b"}
    labels = {"gnss": "fused — GNSS tracking", "blackout": "fused — GNSS blackout (INS only)",
              "blend": "fused — reconnect blend"}
    pos = result.position[lo:hi]
    for mode_val, color in colors.items():
        mask = mode_arr == mode_val
        if mask.any():
            ax.scatter(pos[mask, 0], pos[mask, 1], s=10, color=color, label=labels[mode_val], zorder=2)

    avail = gnss_available[lo:hi]
    blackout_idx = np.where(~avail)[0]
    if len(blackout_idx):
        ax.scatter(*gnss_xy[lo + blackout_idx[0]], marker="x", s=140, color="#dc2626", zorder=3,
                   linewidths=3, label="blackout starts")
        ax.scatter(*gnss_xy[lo + blackout_idx[-1]], marker="x", s=140, color="#16a34a", zorder=3,
                   linewidths=3, label="GNSS reconnects")
    ax.set_aspect("equal", adjustable="datalim")


def _plot(gnss_xy: np.ndarray, result, gnss_available: np.ndarray, dt: float, out_path: str, trace_name: str,
          zoom_pad_s: float = 45.0):
    """Two panels: the whole recorded drive for context (left) and a
    zoomed-in view padded around the actual blackout/reconnect handoff
    (right) — the handoff is often a small fraction of a multi-minute trip,
    so a single whole-trip view makes the interesting part illegible."""
    n = len(gnss_xy)
    blackout_idx = np.where(~gnss_available)[0]
    fig, axes = plt.subplots(1, 2 if len(blackout_idx) else 1, figsize=(15 if len(blackout_idx) else 8, 7))
    axes = np.atleast_1d(axes)

    _plot_track(axes[0], gnss_xy, result, gnss_available, 0, n)
    axes[0].set_title("full recorded drive")
    axes[0].set_xlabel("local x (m, east)")
    axes[0].set_ylabel("local y (m, north)")

    if len(blackout_idx):
        pad = int(zoom_pad_s / dt)
        lo = max(0, blackout_idx[0] - pad)
        hi = min(n, blackout_idx[-1] + pad)
        _plot_track(axes[1], gnss_xy, result, gnss_available, lo, hi)
        axes[1].set_title(f"zoomed on the handoff (±{zoom_pad_s:.0f}s)")
        axes[1].set_xlabel("local x (m, east)")
        axes[1].legend(loc="best", fontsize=8)

    fig.suptitle(f"GNSS↔INS mode-switching — {trace_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
