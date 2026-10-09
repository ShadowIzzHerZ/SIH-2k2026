"""
Ground-truth label audit across every dataset the model trains/tests on.

Two label bugs were found in IO-VNBD only after months of wrong conclusions
(speed column headed "Kmh" but really m/s, 3.6x too small; GPS position held
between fixes, a staircase). This script checks every dataset for the same
failure families, from the loaders' own output (the exact arrays windowing.py
trains on), so the next one is found by measurement, not by luck.

Per sequence it measures:
  speed_ratio  GPS path length / integral of speed_gt, median over 60 s blocks.
               ~1.0 = consistent. ~3.6 = speed_gt is in km/h-as-m/s (or the
               reverse, ~0.28). Anything far from 1 means speed_gt and
               position disagree about how far the vehicle went.
  held_frac    share of consecutive 10 Hz samples with IDENTICAL lat/lon
               (a stair-stepped position label; resampled-linear data is 0).
  head_err_deg median |heading_gt - direction of travel| (moving > 3 m/s,
               travel direction from position over +-1 s).
  yaw_err_deg  median |heading change - integrated gyro z| over 10 s (moving > 3 m/s):
               the independent-sensor check from diagnose_label_noise.py. 10 s because
               IO-VNBD's GPS only updates every few seconds.
  yaw_corr     correlation of those two signals over 10 s.

Run:
    python -m src.audit_labels            # everything found on disk
    python -m src.audit_labels --only iovnbd comma
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.io_vnbd_loader import compass_deg_to_xy_unit, latlon_to_local_xy, load_sequence, recover_yaw_axis
from src.data.windowing import calibrate_sequence, resample_uniform

HZ = 10.0
REPAIR_IOVNBD = False   # --repair: audit IO-VNBD after recover_yaw_axis


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def audit_sequence(seq) -> dict | None:
    if seq.lat is None or seq.lon is None or seq.speed_gt is None or seq.n < 300:
        return None
    dt = 1.0 / HZ
    xy = latlon_to_local_xy(seq.lat, seq.lon)
    speed = np.asarray(seq.speed_gt, dtype=float)
    out = {"n": seq.n, "dur_s": seq.n * dt}

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    moving_step = speed[:-1] > 1.0     # a parked car legitimately repeats its position
    out["held_frac"] = float(np.mean(step[moving_step] < 1e-9)) if moving_step.any() else 0.0

    block = int(30 * HZ)
    ratios = []
    for s in range(0, seq.n - block, block):
        path = step[s:s + block].sum()
        integ = speed[s:s + block].sum() * dt
        if path > 50 and integ > 50 and step[s:s + block].max() < 50:   # skip GPS-jump / parked blocks
            ratios.append(path / integ)
    out["speed_ratio"] = float(np.median(ratios)) if ratios else float("nan")

    if seq.heading_gt is not None:
        k = int(1.0 * HZ)
        d = xy[2 * k:] - xy[:-2 * k]                       # travel over +-1 s centred on t
        travel = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        unit = compass_deg_to_xy_unit(seq.heading_gt)
        head = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))[k:-k]
        moving = (speed[k:-k] > 3.0) & (np.linalg.norm(d, axis=1) > 6.0)
        if moving.sum() > 50:
            out["head_err_deg"] = float(np.median(np.abs(_wrap(head[moving] - travel[moving]))))
        H = int(10 * HZ)
        if seq.n > 3 * H:
            hrad = np.unwrap(np.arctan2(unit[:, 1], unit[:, 0]))
            dh = hrad[H:] - hrad[:-H]
            cg = np.cumsum(seq.gyro[:, 2].astype(float)) * dt
            dg = (cg[H:] - cg[:-H])
            ms = np.convolve(speed, np.ones(H) / H, mode="valid")[: len(dh)]
            m = ms > 3.0
            if m.sum() > 300:
                out["yaw_corr"] = float(np.corrcoef(dh[m], dg[m])[0, 1])          # heading change vs integrated gz over 10 s
                out["yaw_err_deg"] = float(np.median(np.abs(np.degrees(dh[m] - dg[m]))))
    return out


def load_sets(only: set[str] | None, cfg) -> dict[str, list]:
    sets: dict[str, list] = {}

    def want(k):
        return only is None or k in only

    if want("iovnbd"):
        seqs = []
        for p in sorted(glob.glob(f"{cfg['data']['root']}/{cfg['data']['variant']}/**/S-*.csv", recursive=True)):
            try:
                s = resample_uniform(load_sequence(Path(p), cfg["data"]["column_map"]), HZ)
                if REPAIR_IOVNBD:
                    s = recover_yaw_axis(s)
                seqs.append(calibrate_sequence(s))
            except Exception as e:
                print(f"[skip] {p}: {e}")
        sets["IO-VNBD"] = seqs
    if want("comma"):
        from src.data.comma2k19_loader import load_all_segments
        sets["comma2k19"] = [calibrate_sequence(s) for s in load_all_segments(sorted(glob.glob("data/comma2k19_demo/data/*.parquet")), HZ)]
    if want("decimeter") and Path("DECIMETER/sdc2023/sdc2023/train").exists():
        from src.data.decimeter_loader import load_all_segments
        sets["decimeter"] = [calibrate_sequence(s) for s in load_all_segments("DECIMETER/sdc2023/sdc2023/train", HZ)]
    if want("ppc") and Path("data/PPC/PPC-Dataset").exists():
        from src.data.ppc_loader import load_all_segments
        sets["PPC"] = [calibrate_sequence(s) for s in load_all_segments("data/PPC/PPC-Dataset", HZ)]
    if want("pvs") and Path("data/PVS/PVS-Dataset").exists():
        from src.data.pvs_loader import load_all_segments
        sets["PVS"] = [calibrate_sequence(s) for s in load_all_segments("data/PVS/PVS-Dataset", HZ)]
    if want("own"):
        seqs = []
        for p in sorted(glob.glob("data/own_recordings/*.csv")):
            try:
                seqs.append(calibrate_sequence(resample_uniform(load_sequence(Path(p), cfg["data"]["column_map"]), HZ)))
            except Exception as e:
                print(f"[skip] {p}: {e}")
        sets["own_recordings"] = seqs
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="iovnbd comma decimeter ppc pvs own")
    ap.add_argument("--out", default="results/label_audit.json")
    ap.add_argument("--repair", action="store_true", help="apply recover_yaw_axis to IO-VNBD first (shows the after-fix audit; files it rejects are skipped)")
    args = ap.parse_args()
    global REPAIR_IOVNBD
    REPAIR_IOVNBD = args.repair
    cfg = yaml.safe_load(open(args.config))

    sets = load_sets(set(args.only) if args.only else None, cfg)
    summary = {}
    print(f"\n{'dataset':15s} {'seqs':>5s} {'hours':>6s} {'speed_ratio':>12s} {'held%':>6s} {'head_err°':>10s} {'yaw_err_10s°':>12s} {'yaw_corr':>9s}")
    for name, seqs in sets.items():
        rows = [r for r in (audit_sequence(s) for s in seqs) if r]
        if not rows:
            print(f"{name:15s} no usable sequences"); continue

        def med(k):
            v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
            return float(np.median(v)) if v else float("nan")

        flagged = [i for i, r in enumerate(rows) if np.isfinite(r["speed_ratio"]) and not 0.8 <= r["speed_ratio"] <= 1.25]
        summary[name] = {"n_seq": len(rows), "hours": sum(r["dur_s"] for r in rows) / 3600,
                         "speed_ratio_median": med("speed_ratio"), "held_pct_median": 100 * med("held_frac"),
                         "head_err_deg_median": med("head_err_deg"), "yaw_err_deg_10s_median": med("yaw_err_deg"),
                         "yaw_corr_median": med("yaw_corr"), "n_speed_ratio_outside_0.8_1.25": len(flagged)}
        s = summary[name]
        print(f"{name:15s} {s['n_seq']:5d} {s['hours']:6.1f} {s['speed_ratio_median']:12.2f} {s['held_pct_median']:6.1f} "
              f"{s['head_err_deg_median']:10.1f} {s['yaw_err_deg_10s_median']:11.2f} {s['yaw_corr_median']:9.2f}   "
              f"(speed_ratio outside 0.8-1.25: {len(flagged)}/{len(rows)})")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
