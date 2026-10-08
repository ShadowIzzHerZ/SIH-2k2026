"""
Checks a specific hypothesis about *why* IO-VNBD's low-speed/urban windows
drift so much worse than highway ones, raised while explaining the gap to
someone outside the project: maybe it isn't (only) that the model
underfits a harder physical regime -- maybe the *ground truth itself*
(heading_gt, a GPS course/bearing field per io_vnbd_loader.py's fuzzy
column patterns) is noisier at low speed, since GPS course-over-ground is
a Doppler/position-differencing estimate that's well known to get
unreliable near zero velocity (same hazard fusion.py's own docstring
already flags for position-differenced speed). If the label the model is
trained and evaluated against is itself noisy in exactly the regime that
scores worst, that would produce the diagnosed pattern (broad, spread-out
error, no single correlate) without any of it being a fixable modeling
gap -- a different, not mutually exclusive, explanation from the
capacity/data-mix story diagnose_drift.py already confirmed.

Method: for each raw sequence, compare heading_gt's own sample-to-sample
change against what the gyroscope (gz) independently says the heading
should have changed by over the same interval. Real turning should make
these agree closely, regardless of speed -- a physical rotation rate
measured by two different, independent sensors shouldn't disagree more at
low speed than high speed unless one of them (heading_gt, since gyro
bias/noise doesn't have a speed dependence) is genuinely noisier there.
Bins samples by concurrent speed and reports the RMS disagreement and the
correlation between the two signals per bin.

Run:
    python -m src.diagnose_label_noise --data_root data/IO-VNBD --variant "Synchronised V abd S datasets"
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from src.data.io_vnbd_loader import load_sequence


def wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/IO-VNBD")
    parser.add_argument("--variant", default="Synchronised V abd S datasets")
    parser.add_argument("--file_prefix", default="S-")
    parser.add_argument(
        "--speed_bins", type=float, nargs="+",
        default=[0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 100.0],
        help="m/s bin edges, matching FusionEngine's zuptMaxSpeed=3.0 m/s as one boundary",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(f"{args.data_root}/{args.variant}/**/{args.file_prefix}*.csv", recursive=True))
    print(f"found {len(paths)} candidate files")

    bin_edges = args.speed_bins
    n_bins = len(bin_edges) - 1
    gyro_heading_deg = [[] for _ in range(n_bins)]  # per-sample gyro-integrated delta heading, per bin
    label_heading_deg = [[] for _ in range(n_bins)]  # per-sample heading_gt delta, same samples
    n_files_used = 0

    for p in paths:
        try:
            seq = load_sequence(Path(p))
        except Exception as e:
            continue
        if seq.heading_gt is None or seq.speed_gt is None or seq.gyro is None or seq.time is None:
            continue
        if len(seq.heading_gt) < 20:
            continue

        dt = np.diff(seq.time)
        valid_dt = (dt > 0) & (dt < 2.0)  # drop gaps (real blackout stretches, per protocol)
        if valid_dt.sum() < 10:
            continue

        gz = seq.gyro[:-1, 2][valid_dt]  # yaw rate at step start, rad/s
        gyro_delta_deg = np.degrees(gz * dt[valid_dt])

        label_delta_deg = wrap_deg(seq.heading_gt[1:][valid_dt] - seq.heading_gt[:-1][valid_dt])

        speed_at_step = seq.speed_gt[:-1][valid_dt]

        finite = np.isfinite(gyro_delta_deg) & np.isfinite(label_delta_deg) & np.isfinite(speed_at_step)
        if finite.sum() < 10:
            continue

        gyro_delta_deg = gyro_delta_deg[finite]
        label_delta_deg = label_delta_deg[finite]
        speed_at_step = speed_at_step[finite]

        bin_idx = np.digitize(speed_at_step, bin_edges) - 1
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            gyro_heading_deg[b].append(gyro_delta_deg[mask])
            label_heading_deg[b].append(label_delta_deg[mask])

        n_files_used += 1

    print(f"used {n_files_used} files with usable heading_gt + speed_gt + gyro\n")

    if n_files_used == 0:
        print("No usable files found -- check --data_root/--variant/--file_prefix.")
        return

    print(f"{'speed bin (m/s)':<18}{'n samples':>12}{'RMS disagree (deg/step)':>26}{'corr(gyro,label)':>20}")
    for b in range(n_bins):
        if not gyro_heading_deg[b]:
            continue
        g = np.concatenate(gyro_heading_deg[b])
        l = np.concatenate(label_heading_deg[b])
        disagree = wrap_deg(g - l)
        rms = float(np.sqrt(np.mean(disagree ** 2)))
        corr = float(np.corrcoef(g, l)[0, 1]) if len(g) > 1 and np.std(g) > 0 and np.std(l) > 0 else float("nan")
        label = f"[{bin_edges[b]:.1f}, {bin_edges[b + 1]:.1f})"
        print(f"{label:<18}{len(g):>12}{rms:>26.2f}{corr:>20.3f}")

    print(
        "\nInterpretation: if RMS disagreement is much higher and/or correlation much\n"
        "weaker in the low-speed bins than the high-speed ones, heading_gt (not just\n"
        "the model) is genuinely noisier at low speed -- the hypothesis holds, at\n"
        "least in part. If disagreement is roughly flat across bins, the label isn't\n"
        "the story and the capacity/data-mix explanation diagnose_drift.py already\n"
        "found stands as the full explanation."
    )


if __name__ == "__main__":
    main()
