"""
Two checks on what the model is trained and scored against, neither of which
needs a training run. Both look for error that is already in the inputs and
labels before any network sees them.

1. Label lag. GPS speed and course reach the log later than the IMU samples
   that caused them (fix latency, receiver filtering, held 1 Hz fixes). A late
   label gives every window a stale v0 and heading0 and supervises the
   corrections against the wrong instant. Measured per file as the shift that
   best lines up  heading change over 1 s  with  integrated gz over 1 s
   (and  speed change over 1 s  with  integrated forward accel over 1 s).
   Positive lag = the label trails the IMU.

2. Leveling after yaw recovery (IO-VNBD only). recover_yaw_axis fits gz
   against the GPS course because the gyro axes in those logs are not in the
   accelerometer's frame. calibrate_sequence then rotates the gyro by the
   accelerometer's leveling rotation anyway. This compares gz before and after
   that rotation against the GPS course over 10 s (audit_labels' yaw_err).

For both it also reports what they cost in drift: the plain-integration
position error after 30 s, as a percentage of distance, with the inputs as
trained, with gz kept from the recovery, with the labels shifted by the
measured lag, and with both. Each file's constant forward-accel offset is
removed first (and reported), because on its own it is larger than either
effect. No model is involved, so a change here is a change in the floor
every model trains against.

Run:
    python -m src.diagnose_drift_inputs                 # IO-VNBD and comma2k19, whatever is on disk
    python -m src.diagnose_drift_inputs --only iovnbd
Writes results/drift_inputs_diagnosis.json.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.io_vnbd_loader import compass_deg_to_xy_unit, latlon_to_local_xy, load_sequence, recover_yaw_axis
from src.data.windowing import calibrate_sequence, resample_uniform

HZ = 10.0
DT = 1.0 / HZ


def best_lag_s(imu_delta: np.ndarray, label_delta: np.ndarray, ok: np.ndarray,
               max_lag_s: float = 3.0, dt: float = DT, min_samples: int = 200) -> tuple[float, float, float]:
    """Shift (seconds) of label_delta relative to imu_delta that maximises
    their correlation. Positive = the label trails the IMU. Returns
    (lag_s, correlation at that lag, correlation at zero lag); NaNs when there
    is too little usable data."""
    n = len(imu_delta)
    L = int(round(max_lag_s / dt))
    corr = np.full(2 * L + 1, np.nan)
    for j, lag in enumerate(range(-L, L + 1)):
        if lag >= 0:
            a, b, m = imu_delta[:n - lag], label_delta[lag:], ok[:n - lag] & ok[lag:]
        else:
            a, b, m = imu_delta[-lag:], label_delta[:n + lag], ok[-lag:] & ok[:n + lag]
        if m.sum() >= min_samples and a[m].std() > 0 and b[m].std() > 0:
            corr[j] = np.corrcoef(a[m], b[m])[0, 1]
    if not np.isfinite(corr).any():
        return float("nan"), float("nan"), float("nan")
    i = int(np.nanargmax(corr))
    off = 0.0
    if 0 < i < 2 * L and np.isfinite(corr[i - 1]) and np.isfinite(corr[i + 1]):
        denom = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if denom < 0:
            off = 0.5 * (corr[i - 1] - corr[i + 1]) / denom      # parabola through the peak
    return (i - L + off) * dt, float(corr[i]), float(corr[L])


def shift_labels(x: np.ndarray, lag_s: float, dt: float = DT) -> np.ndarray:
    """Undo a label lag: out[t] = x[t + lag]. The tail the shift runs out of is
    NaN, so windows touching it are dropped rather than scored on made-up labels."""
    k = int(round(lag_s / dt))
    out = np.full_like(x, np.nan, dtype=float)
    if k == 0:
        out[:] = x
    elif k > 0:
        out[:-k] = x[k:]
    else:
        out[-k:] = x[:k]
    return out


def physics_drift_pct(a_fwd, gz, speed, theta, xy, horizon_s: float = 30.0, step_s: float = 10.0,
                      min_speed: float = 3.0, dt: float = DT) -> list[float]:
    """Position error after integrating a_fwd and gz for horizon_s from the
    label's own speed and heading, as % of the distance travelled. One value
    per start (every step_s) where the vehicle is moving and the track is clean."""
    H = int(round(horizon_s / dt))
    out = []
    for s in range(0, len(speed) - H - 1, int(round(step_s / dt))):
        seg = xy[s:s + H + 1]
        if not (np.isfinite(seg).all() and np.isfinite(speed[s:s + H]).all() and np.isfinite(theta[s])):
            continue
        hop = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        path = hop.sum()
        if speed[s:s + H].mean() < min_speed or path < 50.0 or hop.max() > 50.0:
            continue
        v = speed[s] + np.cumsum(a_fwd[s:s + H]) * dt
        th = theta[s] + np.cumsum(gz[s:s + H]) * dt
        p = seg[0] + np.cumsum(np.stack([v * np.cos(th), v * np.sin(th)], axis=1), axis=0) * dt
        out.append(float(np.linalg.norm(p[-1] - seg[-1]) / path * 100.0))
    return out


def yaw_agreement(gz, theta, speed, horizon_s: float = 10.0, dt: float = DT) -> tuple[float, float]:
    """(median |heading change - integrated gz| in degrees, correlation) over
    horizon_s spans while moving > 3 m/s: audit_labels' yaw_err and yaw_corr."""
    H = int(round(horizon_s / dt))
    if len(gz) <= 3 * H:
        return float("nan"), float("nan")
    dh = theta[H:] - theta[:-H]
    cg = np.concatenate([[0.0], np.cumsum(gz) * dt])      # cg[i] = turn before sample i, so dg spans the same samples as dh
    dg = cg[H:-1] - cg[:-H - 1]
    ms = np.convolve(speed, np.ones(H) / H, mode="valid")[: len(dh)]
    m = (ms > 3.0) & np.isfinite(dh) & np.isfinite(dg)
    if m.sum() < 300:
        return float("nan"), float("nan")
    return float(np.median(np.abs(np.degrees(dh[m] - dg[m])))), float(np.corrcoef(dh[m], dg[m])[0, 1])


def diagnose_sequence(seq, gz_before_leveling: np.ndarray | None = None) -> dict | None:
    """seq: calibrated (accel[:,0] forward, gyro[:,2] yaw rate). gz_before_leveling:
    the yaw rate as it was before calibrate_sequence, when that is worth comparing."""
    if seq.heading_gt is None or seq.speed_gt is None or seq.lat is None or seq.n < 600:
        return None
    speed = np.asarray(seq.speed_gt, dtype=float)
    unit = compass_deg_to_xy_unit(np.asarray(seq.heading_gt, dtype=float))
    theta = np.unwrap(np.arctan2(unit[:, 1], unit[:, 0]))
    xy = latlon_to_local_xy(seq.lat, seq.lon)
    a_fwd = seq.accel[:, 0].astype(float)
    gz = seq.gyro[:, 2].astype(float)
    # Leveling from the first 2 s leaves a constant forward-accel offset (2 deg of
    # tilt is 0.34 m/s^2) that swamps everything else after 30 s. Take the file's
    # own offset out so the drift figures below show the lag and the yaw rate.
    fin = np.isfinite(speed)
    a_bias = float((np.sum(a_fwd[fin]) * DT - (speed[fin][-1] - speed[fin][0])) / (fin.sum() * DT)) if fin.sum() > 1 else 0.0
    a_fwd = a_fwd - a_bias

    k = int(round(1.0 / DT))
    cum = lambda x: np.concatenate([[0.0], np.cumsum(x) * DT])          # noqa: E731
    moving = (speed[k:] > 3.0) & (speed[:-k] > 3.0)
    lag_h, c_h, c_h0 = best_lag_s(cum(gz)[k:-1] - cum(gz)[:-k - 1], theta[k:] - theta[:-k], moving)
    lag_v, c_v, c_v0 = best_lag_s(cum(a_fwd)[k:-1] - cum(a_fwd)[:-k - 1], speed[k:] - speed[:-k], moving)
    out = {"minutes": seq.n * DT / 60.0, "forward_accel_offset_mps2": a_bias,
           "heading_lag_s": lag_h, "heading_corr_at_lag": c_h, "heading_corr_at_zero": c_h0,
           "speed_lag_s": lag_v, "speed_corr_at_lag": c_v, "speed_corr_at_zero": c_v0}

    drift = {"as_trained": physics_drift_pct(a_fwd, gz, speed, theta, xy)}
    lag = lag_h if np.isfinite(lag_h) else 0.0
    shifted = [shift_labels(speed, lag), shift_labels(theta, lag),
               np.stack([shift_labels(xy[:, 0], lag), shift_labels(xy[:, 1], lag)], axis=1)]
    drift["labels_shifted"] = physics_drift_pct(a_fwd, gz, *shifted)
    if gz_before_leveling is not None:
        gz0 = np.asarray(gz_before_leveling, dtype=float)
        out["yaw_err_deg_10s_after_leveling"], out["yaw_corr_after_leveling"] = yaw_agreement(gz, theta, speed)
        out["yaw_err_deg_10s_before_leveling"], out["yaw_corr_before_leveling"] = yaw_agreement(gz0, theta, speed)
        drift["gz_from_recovery"] = physics_drift_pct(a_fwd, gz0, speed, theta, xy)
        drift["gz_from_recovery_and_labels_shifted"] = physics_drift_pct(a_fwd, gz0, *shifted)
    out["drift_30s_pct"] = drift
    return out


def leveling_tilt_deg(accel: np.ndarray, stationary_samples: int = 20) -> float:
    """Angle between the accelerometer's start-of-file gravity and its z axis:
    how far calibrate_sequence's leveling rotation turns the gyro."""
    g = accel[:stationary_samples].mean(axis=0)
    return float(np.degrees(np.arccos(np.clip(g[2] / np.linalg.norm(g), -1.0, 1.0))))


def load_and_diagnose(only: set[str] | None, cfg) -> dict[str, list[dict]]:
    per_set: dict[str, list[dict]] = {}
    if only is None or "iovnbd" in only:
        rows = []
        for p in sorted(glob.glob(f"{cfg['data']['root']}/{cfg['data']['variant']}/**/S-*.csv", recursive=True)):
            try:
                seq = recover_yaw_axis(resample_uniform(load_sequence(Path(p), cfg["data"]["column_map"]), HZ))
                gz0 = seq.gyro[:, 2].copy()
                tilt = leveling_tilt_deg(seq.accel)
                row = diagnose_sequence(calibrate_sequence(copy.deepcopy(seq)), gz_before_leveling=gz0)
            except Exception as e:
                print(f"[skip] {p}: {e}")
                continue
            if row is not None:
                rows.append({"file": Path(p).name, "leveling_tilt_deg": tilt, **row})
        per_set["IO-VNBD"] = rows
    if only is None or "comma" in only:
        paths = sorted(glob.glob("data/comma2k19_demo/data/*.parquet"))
        if paths:
            from src.data.comma2k19_loader import load_all_segments
            rows = []
            for i, seq in enumerate(load_all_segments(paths, HZ)):
                row = diagnose_sequence(calibrate_sequence(seq))
                if row is not None:
                    rows.append({"file": f"segment_{i}", **row})
            per_set["comma2k19"] = rows
    return per_set


def summarise(rows: list[dict]) -> dict:
    med = lambda key: float(np.nanmedian([r.get(key, np.nan) for r in rows])) if rows else float("nan")   # noqa: E731
    pct = lambda key, q: float(np.nanpercentile([r.get(key, np.nan) for r in rows], q)) if rows else float("nan")   # noqa: E731
    s = {"n_files": len(rows), "hours": sum(r["minutes"] for r in rows) / 60.0}
    for key in ("heading_lag_s", "speed_lag_s"):
        s[f"{key}_median"] = med(key)
        s[f"{key}_p10"] = pct(key, 10)
        s[f"{key}_p90"] = pct(key, 90)
    s["forward_accel_offset_abs_mps2_median"] = float(np.nanmedian([abs(r["forward_accel_offset_mps2"]) for r in rows])) if rows else float("nan")
    for key in ("heading_corr_at_lag", "heading_corr_at_zero", "speed_corr_at_lag", "speed_corr_at_zero",
                "leveling_tilt_deg", "yaw_err_deg_10s_before_leveling", "yaw_err_deg_10s_after_leveling",
                "yaw_corr_before_leveling", "yaw_corr_after_leveling"):
        if any(key in r for r in rows):
            s[f"{key}_median"] = med(key)
    if any("leveling_tilt_deg" in r for r in rows):
        s["leveling_tilt_deg_max"] = float(np.nanmax([r.get("leveling_tilt_deg", np.nan) for r in rows]))
        s["n_files_yaw_worse_after_leveling"] = int(sum(
            r.get("yaw_err_deg_10s_after_leveling", 0.0) > r.get("yaw_err_deg_10s_before_leveling", 0.0) + 0.5 for r in rows))
    variants = sorted({v for r in rows for v in r["drift_30s_pct"]})
    s["drift_30s_pct_median"] = {
        v: float(np.median(sum((r["drift_30s_pct"].get(v, []) for r in rows), []))) if any(r["drift_30s_pct"].get(v) for r in rows) else float("nan")
        for v in variants}
    s["drift_30s_windows"] = {v: int(sum(len(r["drift_30s_pct"].get(v, [])) for r in rows)) for v in variants}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="*", choices=["iovnbd", "comma"], default=None)
    ap.add_argument("--out", default="results/drift_inputs_diagnosis.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    per_set = load_and_diagnose(set(args.only) if args.only else None, cfg)
    report = {}
    for name, rows in per_set.items():
        s = summarise(rows)
        report[name] = {"summary": s,
                        "files": [{k: v for k, v in r.items() if k != "drift_30s_pct"} for r in rows]}
        print(f"\n== {name}: {s['n_files']} files, {s['hours']:.1f} h ==")
        for k, v in s.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk:40s} {vv:.2f}" if isinstance(vv, float) else f"    {kk:40s} {vv}")
            elif k not in ("n_files", "hours"):
                print(f"  {k:44s} {v:.3f}" if isinstance(v, float) else f"  {k:44s} {v}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
