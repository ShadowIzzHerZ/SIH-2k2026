"""
Simulated high-rate (200 / 400 Hz) IMU streams built from real recordings.

There is no FOG or 200 Hz recording in the repo, so this is the closest
check the data allows. It takes a real drive (comma2k19 at ~100 Hz, or an
IO-VNBD phone file at 10 Hz), brings its IMU into the vehicle frame, and
re-records it as a different sensor would see it:

  - upsampled to 200 or 400 Hz on the sensor's own clock (linear
    interpolation of the real signal, plus engine/road vibration that the
    source rate cannot carry),
  - bolted in at a different orientation (z-down, yawed 90 degrees, with a
    small residual tilt the installer did not measure),
  - with the bias and noise of a MEMS or a FOG unit added on top,
  - timestamped by a host clock with jitter and a little drift, so
    timestamps are irregular and occasionally go backwards,
  - with dropped samples: isolated losses and a few bursts, one of them
    inside the GNSS blackout,
  - with a 10 Hz GNSS receiver whose fixes are not phase-locked to the IMU.

Then it runs `python -m edge.cli run` on each stream with a GNSS blackout
and reports drift for the model, for plain strapdown, and for the same
drive at the source's native rate, plus CPU time per second of data and
whether the live output really has one position per IMU sample.

This is a simulated stream, not a sensor recording. The source data
already carries its own sensor's errors (comma's MEMS, the phone's MEMS),
so the "FOG" stream is the source plus FOG-grade errors, never better than
the source. Upsampling cannot add real dynamics above the source's
Nyquist rate; the vibration term only stands in for them.

    python -m edge.stream_test --source comma2k19:data/comma2k19_demo/data/demo-00000-of-00003.parquet:0 --blackout 20:35
    python -m edge.stream_test --source "iovnbd:data/IO-VNBD/.../S-M.csv:300:120" --blackout 60:75
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time

import numpy as np

from .cli import build_parser, cmd_run
from .io import compass_to_math, load_csv, math_to_compass, write_csv, xy_to_latlon
from .mounting import estimate_mount, level, rotation_from_axes, rotation_from_rpy
from .resample import resample

# Per grade: accel turn-on bias (m/s^2), accel noise density (m/s^2/sqrt(Hz)),
# gyro turn-on bias (deg/s), gyro noise density (deg/s/sqrt(Hz)).
GRADES = {
    # automotive MEMS IMU, uncompensated turn-on bias (BMI088 / SMI230 class)
    "mems": (0.05, 0.0015, 0.1, 0.01),
    # navigation FOG (KVH 1750 class): ~1 deg/h bias, 0.012 deg/sqrt(h) ARW, 0.12 mg/sqrt(Hz) VRW
    "fog": (0.01, 0.0012, 1.0 / 3600, 0.012 / 60),
}
MOUNT_RPY = (180.0, 0.0, 90.0)  # nominal install: z down, yawed 90 degrees
TILT_DEG = (0.8, -1.5)          # roll, pitch the installer did not measure


def _source_comma2k19(parquet: str, segment: int, tmp: str) -> dict:
    from .comma2k19 import convert

    path = os.path.join(tmp, "native.csv")
    convert(parquet, segment, path)
    rec = load_csv(path)
    R_pre = rotation_from_axes("x,-y,-z")
    R, _ = level(rec.accel[rec.t_imu - rec.t_imu[0] < 2.0], R_pre)
    return {"rec": rec, "native_csv": path, "native_flags": ["--axes", "x,-y,-z"], "R_to_vehicle": R}


def _source_iovnbd(path: str, start_s: float, dur_s: float, tmp: str) -> dict:
    from pathlib import Path

    from src.data.io_vnbd_loader import load_sequence

    seq = load_sequence(Path(path))
    t = seq.time - seq.time[0]
    # Some files (S-M.csv) are recordings joined end to end, with the clock
    # restarting. Take one contiguous run of increasing time.
    i0 = int(np.argmax(t >= start_s))
    back = np.flatnonzero(np.diff(t[i0:]) < 0)
    i1 = i0 + (int(back[0]) + 1 if len(back) else len(t) - i0)
    sel = np.zeros(len(t), dtype=bool)
    sel[i0:i1] = True
    sel &= (t < start_s + dur_s) & np.isfinite(seq.lat) & np.isfinite(seq.lon)
    rows = []
    for i in np.flatnonzero(sel):
        rows.append([f"{seq.time[i] - seq.time[0] - start_s:.4f}", *seq.accel[i], *seq.gyro[i],
                     f"{seq.lat[i]:.9f}", f"{seq.lon[i]:.9f}", f"{seq.speed_gt[i]:.3f}", f"{seq.heading_gt[i]:.3f}"])
    native = os.path.join(tmp, "native.csv")
    write_csv(native, ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed", "heading"], rows)
    rec = load_csv(native)
    # Phone in a holder: frame unknown, so level and fit yaw on the whole clip.
    t10, a10 = resample(rec.t_imu, rec.accel, 10.0, "interp")
    spd = np.interp(t10, rec.t_gnss, rec.gnss_speed)
    crs = np.degrees(np.unwrap(np.radians(rec.gnss_course)))
    crs = np.interp(t10, rec.t_gnss, crs) % 360.0
    R, info = estimate_mount(a10, 0.1, len(a10), crs, spd, min_yaw_samples=100)
    return {"rec": rec, "native_csv": native, "native_flags": [], "R_to_vehicle": R, "mount_fit": info}


def load_source(spec: str, tmp: str) -> dict:
    kind, rest = spec.split(":", 1)
    if kind == "comma2k19":
        pq, seg = rest.rsplit(":", 1)
        return _source_comma2k19(pq, int(seg), tmp)
    if kind == "iovnbd":
        path, start, dur = rest.rsplit(":", 2)
        return _source_iovnbd(path, float(start), float(dur), tmp)
    raise ValueError(f"unknown source {spec!r}")


def simulate(src: dict, out_path: str, rate_hz: float = 200.0, grade: str = "mems", jitter_s: float = 0.0005,
             clock_ppm: float = 20.0, drop_frac: float = 0.01, bursts: list[tuple[float, float]] | None = None,
             vibration: float = 0.4, gnss_hz: float = 10.0, gnss_phase_s: float = 0.037, seed: int = 0) -> dict:
    """Write one simulated sensor CSV. Returns what was injected."""
    rng = np.random.default_rng(seed)
    rec, Rv = src["rec"], src["R_to_vehicle"]
    acc_v, gyr_v = rec.accel @ Rv.T, rec.gyro @ Rv.T
    order = np.argsort(rec.t_imu, kind="stable")
    ts, acc_v, gyr_v = rec.t_imu[order], acc_v[order], gyr_v[order]
    ts, first = np.unique(ts, return_index=True)
    acc_v, gyr_v = acc_v[first], gyr_v[first]

    # The sensor samples on its own regular clock.
    t_true = np.arange(ts[0], ts[-1], 1.0 / rate_hz)
    n = len(t_true)
    a = np.stack([np.interp(t_true, ts, acc_v[:, i]) for i in range(3)], axis=1)
    g = np.stack([np.interp(t_true, ts, gyr_v[:, i]) for i in range(3)], axis=1)

    # Vibration the source rate could not carry: an engine line that follows
    # speed, plus broadband road noise, only while moving.
    spd = np.interp(t_true, rec.t_gnss, rec.gnss_speed)
    moving = (spd > 0.5)[:, None]
    f_eng = 18.0 + 0.9 * spd
    phase = 2 * np.pi * np.cumsum(f_eng) / rate_hz
    a += moving * vibration * (np.sin(phase[:, None] + rng.uniform(0, 2 * np.pi, 3)) * [0.4, 0.6, 1.0]
                               + 0.5 * rng.standard_normal((n, 3)))
    g += moving * 0.1 * vibration * np.radians(1.0) * rng.standard_normal((n, 3))

    ab, an, gb, gn = GRADES[grade]
    acc_bias = ab * rng.choice([-1, 1], 3) * rng.uniform(0.5, 1.0, 3)
    gyr_bias = np.radians(gb) * rng.choice([-1, 1], 3) * rng.uniform(0.5, 1.0, 3)
    a += acc_bias + an * np.sqrt(rate_hz) * rng.standard_normal((n, 3))
    g += gyr_bias + np.radians(gn) * np.sqrt(rate_hz) * rng.standard_normal((n, 3))

    # Mount: vehicle = R_true @ sensor, so sensor = R_true.T @ vehicle.
    R_true = rotation_from_rpy(MOUNT_RPY[0] + TILT_DEG[0], MOUNT_RPY[1] + TILT_DEG[1], MOUNT_RPY[2])
    a_s, g_s = a @ R_true, g @ R_true

    # Host timestamps: clock drift plus jitter. Rows stay in arrival order,
    # so a large jitter makes timestamps go backwards now and then.
    t_rep = ts[0] + (t_true - ts[0]) * (1 + clock_ppm * 1e-6) + jitter_s * rng.standard_normal(n)

    keep = rng.random(n) >= drop_frac
    for b0, b1 in bursts or []:
        keep &= ~((t_true >= ts[0] + b0) & (t_true < ts[0] + b1))

    # 10 Hz receiver, its own phase, interpolated from the source's GNSS track.
    tg = np.arange(rec.t_gnss[0] + gnss_phase_s, rec.t_gnss[-1], 1.0 / gnss_hz)
    xy = np.stack([np.interp(tg, rec.t_gnss, rec.gnss_xy[:, i]) for i in range(2)], axis=1)
    gs = np.interp(tg, rec.t_gnss, rec.gnss_speed)
    crs_u = np.unwrap(compass_to_math(rec.gnss_course))
    gc = math_to_compass(np.interp(tg, rec.t_gnss, crs_u))
    lat, lon = xy_to_latlon(xy, *rec.origin)

    rows, gi = [], 0
    for i in np.flatnonzero(keep):
        while gi < len(tg) and tg[gi] <= t_rep[i]:
            rows.append([f"{tg[gi]:.5f}", "", "", "", "", "", "", f"{lat[gi]:.9f}", f"{lon[gi]:.9f}",
                         f"{gs[gi]:.3f}", f"{gc[gi]:.3f}"])
            gi += 1
        rows.append([f"{t_rep[i]:.5f}", *(f"{v:.6f}" for v in a_s[i]), *(f"{v:.7f}" for v in g_s[i]), "", "", "", ""])
    write_csv(out_path, ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed", "heading"], rows)
    dt_rep = np.diff(t_rep[keep])
    return {
        "rate_hz": rate_hz, "grade": grade, "imu_rows": int(keep.sum()), "dropped": int(n - keep.sum()),
        "backwards_timestamps": int((dt_rep < 0).sum()), "max_gap_s": round(float(dt_rep.max()), 3),
        "gyro_bias_deg_s": [round(float(v), 4) for v in np.degrees(gyr_bias)],
        "accel_bias_mps2": [round(float(v), 4) for v in acc_bias],
    }


def run(csv_path: str, out_dir: str, blackout: str, flags: list[str]) -> dict:
    argv = ["run", csv_path, "--out-dir", out_dir, "--blackout", blackout, *flags]
    args = build_parser().parse_args(argv)
    args.quiet = True
    cpu0 = time.process_time()
    summary = cmd_run(args)
    summary["cpu_s_total"] = round(time.process_time() - cpu0, 3)  # includes the strapdown-only rerun and leveling
    with open(os.path.join(out_dir, "track_live.csv")) as f:
        rows = list(csv.reader(f))[1:]
    live = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows]).reshape(-1, 3)
    summary["live_rows"] = len(live)
    # Fastest the live output moves between consecutive rows, as a teleport
    # check. Time steps are floored at the nominal sample period (jittered
    # timestamps can sit almost on top of each other). The reconnect blend
    # and the second after it are left out: run_fusion's blend walks from
    # the blackout estimate to GNSS in 2 s, so after a large drift it moves
    # fast by design.
    blend = np.array([r[7] == "blend" for r in rows])
    t_blend = live[blend, 0]
    near = np.zeros(len(live), dtype=bool)
    if len(t_blend):
        idx = np.searchsorted(t_blend, live[:, 0], side="right") - 1
        near = blend | ((idx >= 0) & (live[:, 0] - t_blend[np.clip(idx, 0, None)] < 1.0))
    dt = np.maximum(np.diff(live[:, 0]), 1.0 / summary["imu_hz"])
    v = np.linalg.norm(np.diff(live[:, 1:], axis=0), axis=1) / dt
    v = v[~(near[1:] | near[:-1])]
    summary["max_live_speed_mps"] = round(float(v.max()), 1) if len(v) else None
    return summary


def _row(label, s, info=None):
    m, z = s["blackouts_model"][0], s["blackouts_strapdown_only"][0]
    return {
        "case": label,
        "imu_hz": s["imu_hz"],
        "model_drift_pct": m["drift_pct"], "strapdown_drift_pct": z["drift_pct"],
        "model_end_err_m": m["end_error_m"], "strapdown_end_err_m": z["end_error_m"], "distance_m": m["distance_m"],
        "reconnect_step_m": m["reconnect_step_m"], "model_calls": s["model_calls"],
        "blackout_ticks": s["blackout_ticks"],
        "engine_ms_per_s_of_data": round(1e3 * s["wall_s"] / s["duration_s"], 2),
        "one_position_per_imu_row": s["live_rows"] == s["imu_rows"],
        "max_live_speed_mps": s["max_live_speed_mps"],
        "mount": {k: s["mount"][k] for k in ("method", "yaw_deg", "yaw_fit_quality") if k in s["mount"]},
        **({"injected": info} if info else {}),
    }


def run_source(spec: str, blackout: str, rates, grades, seed: int) -> list[dict]:
    b0, b1 = (float(v) for v in blackout.split(":"))
    # One burst before the blackout, one inside it, one after.
    bursts = [(b0 - 8.0, b0 - 7.7), (b0 + 0.4 * (b1 - b0), b0 + 0.4 * (b1 - b0) + 0.5), (b1 + 3.0, b1 + 3.2)]
    rpy = ",".join(str(v) for v in MOUNT_RPY)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        src = load_source(spec, tmp)
        s = run(src["native_csv"], os.path.join(tmp, "o_native"), blackout, src["native_flags"])
        rows.append(_row("native rate, source frame", s))
        cases = [("clean", dict(jitter_s=0.0, clock_ppm=0.0, drop_frac=0.0, bursts=None, vibration=0.0, gnss_phase_s=0.0)),
                 ("impaired", dict(bursts=bursts)),
                 ("impaired, 3 ms jitter", dict(bursts=bursts, jitter_s=0.003))]
        for rate in rates:
            for grade in grades:
                for name, kw in cases:
                    if name != "impaired" and (grade != grades[0] or rate != rates[0]):
                        continue  # the ablations once, at the first rate and grade
                    path = os.path.join(tmp, f"sim_{int(rate)}_{grade}_{len(rows)}.csv")
                    info = simulate(src, path, rate_hz=rate, grade=grade, seed=seed, **kw)
                    for mount_label, flags in [("known mount", ["--mount-rpy", rpy]), ("auto mount", [])]:
                        if mount_label == "auto mount" and name != "impaired":
                            continue
                        s = run(path, os.path.join(tmp, f"o_{len(rows)}"), blackout, flags)
                        rows.append(_row(f"{int(rate)} Hz {grade}, {name}, {mount_label}", s, info))
        v_true = float(np.nanmax(src["rec"].gnss_speed))
    for r in rows:
        r["source"] = spec
        r["max_gnss_speed_mps"] = round(v_true, 1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", required=True, help="one or more sources")
    ap.add_argument("--blackout", default="20:35")
    ap.add_argument("--rates", type=float, nargs="+", default=[200.0, 400.0])
    ap.add_argument("--grades", nargs="+", default=["mems", "fog"])
    ap.add_argument("--min-distance", type=float, default=100.0, help="leave out blackouts shorter than this (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edge/out/stream_test.json")
    a = ap.parse_args(argv)

    rows = []
    for spec in a.source:
        got = run_source(spec, a.blackout, a.rates, a.grades, a.seed)
        rows += got
        print(f"{spec}: {len(got)} runs, distance {got[0]['distance_m']} m")
    scored = [r for r in rows if r["distance_m"] >= a.min_distance]
    table = []
    for case in dict.fromkeys(r["case"] for r in scored):
        rs = [r for r in scored if r["case"] == case]
        table.append({
            "case": case, "n": len(rs),
            "model_median_pct": round(float(np.median([r["model_drift_pct"] for r in rs])), 2),
            "strapdown_median_pct": round(float(np.median([r["strapdown_drift_pct"] for r in rs])), 2),
            "model_better": int(sum(r["model_drift_pct"] < r["strapdown_drift_pct"] for r in rs)),
            "engine_ms_per_s_of_data": round(float(np.median([r["engine_ms_per_s_of_data"] for r in rs])), 2),
            "model_calls_max": max(r["model_calls"] for r in rs),
            "one_position_per_imu_row": all(r["one_position_per_imu_row"] for r in rs),
            "live_over_gnss_speed_max": round(max(r["max_live_speed_mps"] / max(r["max_gnss_speed_mps"], 1.0) for r in rs), 2),
        })
    print(f"{'case':48s} {'n':>3s} {'model':>7s} {'strap':>7s} {'better':>6s} {'ms/s':>6s} calls 1:1  live/gnss speed")
    for t in table:
        print(f"{t['case']:48s} {t['n']:3d} {t['model_median_pct']:6.2f}% {t['strapdown_median_pct']:6.2f}% "
              f"{t['model_better']:6d} {t['engine_ms_per_s_of_data']:6.2f} {t['model_calls_max']:5d} "
              f"{str(t['one_position_per_imu_row']):5s} {t['live_over_gnss_speed_max']:5.2f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"note": "simulated high-rate streams built from real recordings, not sensor recordings",
                   "blackout": a.blackout, "min_distance_m": a.min_distance, "table": table, "rows": rows}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
