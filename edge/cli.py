"""
Edge engine command line.

    # make a 200 Hz test drive and run it with a 20 s GNSS blackout
    python -m edge.cli synth --out edge/out/drive.csv
    python -m edge.cli run edge/out/drive.csv --blackout 30:50

    # a real 100 Hz automotive IMU (comma2k19), 15 s blackout
    python -m edge.cli from-comma2k19 data/comma2k19_demo/data/demo-00000-of-00003.parquet --segment 0 --out edge/out/c2k.csv
    python -m edge.cli run edge/out/c2k.csv --blackout 20:35

    # how fast is it on this box
    python -m edge.cli bench

`run` writes track_10hz.csv (the scored estimate), track_live.csv (one
row per IMU sample) and summary.json to --out-dir. With --blackout, GNSS is
withheld from the engine for that span but kept for scoring, and the same
run is repeated with no learned correction (plain strapdown) so the
summary shows what the model adds.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .engine import BLACKOUT, EdgeEngine, Fix, OnnxPredictor, ZeroPredictor
from .io import compass_to_math, load_csv, math_to_compass, write_csv, xy_to_latlon
from .mounting import estimate_mount, rotation_from_axes, rotation_from_rpy
from .resample import resample

DEFAULT_MODEL = "checkpoints/dead_reckoning_model.onnx"


def parse_spans(specs: list[str]) -> list[tuple[float, float]]:
    spans = []
    for s in specs or []:
        a, b = s.split(":")
        spans.append((float(a), float(b)))
    return spans


def mount_rotation(args, rec, model_hz: float):
    if args.axes:
        R_pre, info = rotation_from_axes(args.axes), {"method": "axes", "axes": args.axes}
    elif args.mount_rpy:
        r, p, y = (float(v) for v in args.mount_rpy.split(","))
        R_pre, info = rotation_from_rpy(r, p, y), {"method": "rpy", "rpy_deg": [r, p, y]}
    else:
        R_pre, info = np.eye(3), {"method": "auto"}
    if args.no_level:
        return R_pre, info
    # Level (and in auto mode, solve yaw) from 10 Hz data, using only the GNSS
    # the engine itself will see, so an injected blackout stays blind.
    t10, a10 = resample(rec.t_imu, rec.accel, model_hz, args.resample)
    spd = np.full(len(t10), np.nan)
    crs = np.full(len(t10), np.nan)
    if len(rec.t_gnss):
        spans = [(rec.t_imu[0] + a, rec.t_imu[0] + b) for a, b in parse_spans(args.blackout)]
        keep = np.ones(len(rec.t_gnss), dtype=bool)
        for a, b in spans:
            keep &= ~((rec.t_gnss >= a) & (rec.t_gnss <= b))
        tg = rec.t_gnss[keep]
        idx = np.searchsorted(tg, t10, side="right") - 1
        fresh = (idx >= 0) & (t10 - tg[np.clip(idx, 0, None)] <= args.gnss_timeout)
        spd[fresh] = rec.gnss_speed[keep][idx[fresh]]
        crs[fresh] = rec.gnss_course[keep][idx[fresh]]
    estimate_yaw = info["method"] == "auto" and args.mount_yaw is None
    R, extra = estimate_mount(a10, 1.0 / model_hz, int(args.level_seconds * model_hz), crs, spd, R_pre=R_pre,
                              estimate_yaw=estimate_yaw, min_yaw_samples=int(args.min_yaw_seconds * model_hz))
    if args.mount_yaw is not None:
        from src.calibration import yaw_rotation
        R = yaw_rotation(np.radians(args.mount_yaw)) @ R
        extra["yaw_deg"] = args.mount_yaw
        extra["yaw_source"] = "--mount-yaw"
    return R, {**info, **extra}


def run_engine(rec, predictor, R, args, model_hz, gnss_mask, predictor_highway=None):
    eng = EdgeEngine(predictor, R, model_hz=model_hz, resample=args.resample, gnss_timeout_s=args.gnss_timeout,
                     live_smooth_s=args.live_smooth, blend_seconds=args.blend_seconds,
                     v_clamp_max=args.v_max, predictor_highway=predictor_highway)
    live = []
    gi, ng = 0, len(rec.t_gnss)
    t_start = time.perf_counter()
    for i in range(len(rec.t_imu)):
        t = rec.t_imu[i]
        while gi < ng and rec.t_gnss[gi] <= t:
            if gnss_mask[gi]:
                spd = rec.gnss_speed[gi]
                crs = rec.gnss_course[gi]
                eng.push_gnss(Fix(rec.t_gnss[gi], *rec.gnss_xy[gi],
                                  None if np.isnan(spd) else float(spd),
                                  None if np.isnan(crs) else float(compass_to_math(crs))))
            gi += 1
        live.append(eng.push_imu(t, rec.accel[i], rec.gyro[i]))
    ticks = eng.finish()
    wall = time.perf_counter() - t_start
    return ticks, live, eng, wall


def score(ticks, rec, spans):
    """Drift % at the end of each blackout, against distance travelled during
    it — the metric definition fusion.blackout_drift_pct uses. Distance is
    the integral of GNSS Doppler speed when the file has it: summing hops
    between noisy fixes inflates it (1 m fix noise at 10 Hz adds ~50% at
    15 m/s), which would flatter every drift number."""
    t = np.array([s.t for s in ticks])
    est = np.array([[s.x, s.y] for s in ticks])
    true = np.stack([np.interp(t, rec.t_gnss, rec.gnss_xy[:, i]) for i in range(2)], axis=1)
    has_speed = np.isfinite(rec.gnss_speed).all() and len(rec.gnss_speed) > 1
    out = []
    for a, b in spans:
        sel = np.where((t >= a) & (t <= b))[0]
        if len(sel) < 2:
            continue
        if has_speed:
            g = (rec.t_gnss >= a) & (rec.t_gnss <= t[sel[-1]])
            dist = float(np.trapezoid(rec.gnss_speed[g], rec.t_gnss[g]))
        else:
            path = true[sel[0] - 1:sel[-1] + 1] if sel[0] > 0 else true[sel]
            dist = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        err = float(np.linalg.norm(est[sel[-1]] - true[sel[-1]]))
        jump = float(np.linalg.norm(est[sel[-1] + 1] - est[sel[-1]])) if sel[-1] + 1 < len(est) else None
        out.append({"span_s": [a, b], "distance_m": round(dist, 1), "end_error_m": round(err, 2),
                    "drift_pct": round(100 * err / max(dist, 1e-6), 2),
                    "reconnect_step_m": None if jump is None else round(jump, 2)})
    return out


def cmd_run(args):
    rec = load_csv(args.input, args.accel_unit, args.gyro_unit)
    if len(rec.t_imu) < 2:
        raise SystemExit("no IMU rows in input")
    if rec.t_imu[-1] - rec.t_imu[0] < 1.0 / args.model_hz:
        raise SystemExit(f"IMU rows span {rec.t_imu[-1] - rec.t_imu[0]:.3f} s, less than one "
                         f"{args.model_hz:g} Hz model tick")
    imu_hz = (len(rec.t_imu) - 1) / (rec.t_imu[-1] - rec.t_imu[0])
    spans = [(rec.t_imu[0] + a, rec.t_imu[0] + b) for a, b in parse_spans(args.blackout)]
    gnss_mask = np.ones(len(rec.t_gnss), dtype=bool)
    for a, b in spans:
        gnss_mask &= ~((rec.t_gnss >= a) & (rec.t_gnss <= b))

    predictor = OnnxPredictor(args.model, threads=args.threads)
    model_hz = args.model_hz
    hw = OnnxPredictor(args.model_highway, threads=args.threads) if args.model_highway else None
    R, mount = mount_rotation(args, rec, model_hz)

    ticks, live, eng, wall = run_engine(rec, predictor, R, args, model_hz, gnss_mask, hw)
    os.makedirs(args.out_dir, exist_ok=True)

    def geo(x, y):
        if rec.origin is None:
            return ["", ""]
        la, lo = xy_to_latlon(np.array([x, y]), *rec.origin)
        return [f"{la:.8f}", f"{lo:.8f}"]

    hdr = ["t", "x_m", "y_m", "lat", "lon", "heading_deg", "speed_mps", "mode"]
    write_csv(os.path.join(args.out_dir, "track_10hz.csv"), hdr,
              ([f"{s.t:.3f}", f"{s.x:.3f}", f"{s.y:.3f}", *geo(s.x, s.y), f"{math_to_compass(s.heading):.2f}",
                f"{s.speed:.3f}", s.mode] for s in ticks))
    write_csv(os.path.join(args.out_dir, "track_live.csv"), hdr,
              ([f"{s.t:.4f}", f"{s.x:.3f}", f"{s.y:.3f}", *geo(s.x, s.y), f"{math_to_compass(s.heading):.2f}",
                f"{s.speed:.3f}", s.mode] for s in live))

    t_first = ticks[0].t if ticks else np.inf
    live_xy = np.array([[s.x, s.y] for s in live if s.t > t_first])  # skip the pre-first-tick zeros
    summary = {
        "input": args.input,
        "model": args.model,
        "model_highway": args.model_highway,
        "imu_rows": int(len(rec.t_imu)),
        "imu_hz": round(float(imu_hz), 1),
        "gnss_rows": int(len(rec.t_gnss)),
        "duration_s": round(float(rec.t_imu[-1] - rec.t_imu[0]), 2),
        "resample": args.resample,
        "mount": mount,
        "model_calls": eng.core.model_calls,
        "blackout_ticks": int(sum(s.mode == BLACKOUT for s in ticks)),
        "wall_s": round(wall, 3),
        "realtime_factor": round(float(rec.t_imu[-1] - rec.t_imu[0]) / wall, 1),
        "max_live_step_m": round(float(np.linalg.norm(np.diff(live_xy, axis=0), axis=1).max()), 3) if len(live_xy) > 1 else None,
    }
    if spans and len(rec.t_gnss):
        summary["blackouts_model"] = score(ticks, rec, spans)
        base_ticks, *_ = run_engine(rec, ZeroPredictor(predictor.window_size), R, args, model_hz, gnss_mask)
        summary["blackouts_strapdown_only"] = score(base_ticks, rec, spans)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if not getattr(args, "quiet", False):
        print(json.dumps(summary, indent=2))
    return summary


def cmd_synth(args):
    from .synth import generate
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mount = rotation_from_axes(args.mount_axes) if args.mount_axes else None
    generate(args.out, duration_s=args.duration, imu_hz=args.imu_hz, gnss_hz=args.gnss_hz, grade=args.grade,
             vibration=args.vibration, mount_axes=mount, seed=args.seed)
    print(f"wrote {args.out}")


def cmd_from_comma2k19(args):
    from .comma2k19 import convert
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    info = convert(args.parquet, args.segment, args.out)
    print(json.dumps({"wrote": args.out, **info}, indent=2))


def cmd_bench(args):
    p = OnnxPredictor(args.model, threads=args.threads)
    x = np.random.default_rng(0).normal(size=(p.window_size, 6))
    x[:, 2] += 9.81
    for _ in range(20):
        p(x)
    n = args.iters
    t0 = time.perf_counter()
    for _ in range(n):
        p(x)
    per_call = (time.perf_counter() - t0) / n

    eng = EdgeEngine(ZeroPredictor(p.window_size), np.eye(3))
    m = int(args.imu_hz * 20)
    t0 = time.perf_counter()
    for i in range(m):
        eng.push_imu(i / args.imu_hz, x[i % len(x), :3], x[i % len(x), 3:])
    per_sample = (time.perf_counter() - t0) / m
    out = {
        "model": args.model,
        "model_ms_per_call": round(per_call * 1e3, 3),
        "model_calls_per_s_needed": 1 / (p.window_size / 10.0),
        "engine_us_per_imu_sample": round(per_sample * 1e6, 1),
        "imu_hz": args.imu_hz,
        "cpu_share_at_imu_hz": f"{100 * (per_sample * args.imu_hz + per_call * 10.0 / p.window_size):.1f}%",
    }
    print(json.dumps(out, indent=2))


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m edge.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process an IMU/GNSS CSV")
    r.add_argument("input")
    r.add_argument("--out-dir", default="edge/out")
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--model-highway", default=None, help="optional second ONNX model used at >= 6 m/s (duo mode)")
    r.add_argument("--accel-unit", choices=["m/s2", "g"], default="m/s2")
    r.add_argument("--gyro-unit", choices=["rad/s", "deg/s"], default="rad/s")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--axes", help='fixed mount as an axis remap, e.g. "x,-y,-z" (vehicle fwd,left,up in sensor axes)')
    g.add_argument("--mount-rpy", help="fixed mount as roll,pitch,yaw degrees")
    r.add_argument("--mount-yaw", type=float, default=None, help="auto-level but force this yaw offset (deg)")
    r.add_argument("--no-level", action="store_true", help="trust --axes / --mount-rpy exactly, skip gravity leveling")
    r.add_argument("--level-seconds", type=float, default=2.0, help="seconds at the start used for gravity leveling")
    r.add_argument("--min-yaw-seconds", type=float, default=30.0)
    r.add_argument("--resample", choices=["interp", "mean"], default="interp",
                   help="interp matches how the training data was made (default); mean is anti-aliased, see edge/resample.py")
    r.add_argument("--model-hz", type=float, default=10.0)
    r.add_argument("--blackout", action="append", metavar="START:END",
                   help="withhold GNSS for this span (seconds from file start); repeatable")
    r.add_argument("--gnss-timeout", type=float, default=1.5)
    r.add_argument("--blend-seconds", type=float, default=2.0)
    r.add_argument("--live-smooth", type=float, default=1.0)
    r.add_argument("--v-max", type=float, default=50.0)
    r.add_argument("--threads", type=int, default=1)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synth", help="write a synthetic high-rate drive CSV")
    s.add_argument("--out", default="edge/out/drive.csv")
    s.add_argument("--duration", type=float, default=100.0)
    s.add_argument("--imu-hz", type=float, default=200.0)
    s.add_argument("--gnss-hz", type=float, default=10.0)
    s.add_argument("--grade", choices=["fog", "mems"], default="fog")
    s.add_argument("--vibration", type=float, default=0.3)
    s.add_argument("--mount-axes", default=None, help="simulate a rotated install, e.g. \"y,-x,z\"")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    c = sub.add_parser("from-comma2k19", help="convert one comma2k19 segment at native IMU rate")
    c.add_argument("parquet")
    c.add_argument("--segment", type=int, default=0)
    c.add_argument("--out", default="edge/out/c2k.csv")
    c.set_defaults(func=cmd_from_comma2k19)

    b = sub.add_parser("bench", help="time the model and the engine loop")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--imu-hz", type=float, default=200.0)
    b.add_argument("--iters", type=int, default=200)
    b.add_argument("--threads", type=int, default=1)
    b.set_defaults(func=cmd_bench)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
