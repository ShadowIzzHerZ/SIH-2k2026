"""
Score the edge engine on every comma2k19 demo segment at the IMU's native
~100 Hz rate, with one injected blackout per segment.

    python -m edge.evaluate_comma2k19 data/comma2k19_demo/data/*.parquet --blackout 20:35

For each segment it runs the learned model and plain strapdown (no
correction) under each resampling method, and reports mean / median drift.
Writes per-segment rows to --out (JSON) so the numbers can be checked.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np

from .cli import build_parser, cmd_run
from .comma2k19 import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--blackout", default="20:35")
    ap.add_argument("--model", default="checkpoints/dead_reckoning_model.onnx")
    ap.add_argument("--resample", nargs="+", default=["mean", "interp"])
    ap.add_argument("--extra", default="", help="extra `run` flags, e.g. \"--mount-yaw 0\"")
    ap.add_argument("--min-distance", type=float, default=100.0)
    ap.add_argument("--out", default="edge/out/comma2k19_eval.json")
    a = ap.parse_args()

    import pandas as pd
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for pq in a.parquets:
            n = len(pd.read_parquet(pq, columns=["segment_id"]))
            for seg in range(n):
                csv_path = os.path.join(tmp, "seg.csv")
                try:
                    info = convert(pq, seg, csv_path)
                except Exception as e:  # a few segments have missing streams
                    print(f"skip {os.path.basename(pq)}#{seg}: {e}")
                    continue
                lo, hi = (float(v) for v in a.blackout.split(":"))
                if info["duration_s"] < hi + 3:
                    print(f"skip {info['segment_id']}: only {info['duration_s']} s")
                    continue
                row = {**info, "parquet": os.path.basename(pq), "segment": seg}
                for rs in a.resample:
                    args = build_parser().parse_args(["run", csv_path, "--blackout", a.blackout, "--resample", rs,
                                                      "--model", a.model, "--out-dir", os.path.join(tmp, "o")]
                                                     + a.extra.split())
                    args.quiet = True
                    s = cmd_run(args)
                    row[f"{rs}_model"] = s["blackouts_model"][0]["drift_pct"]
                    row[f"{rs}_strapdown"] = s["blackouts_strapdown_only"][0]["drift_pct"]
                    row["yaw_deg"] = s["mount"].get("yaw_deg")
                    row["distance_m"] = s["blackouts_model"][0]["distance_m"]
                rows.append(row)
                print(json.dumps({k: row[k] for k in row if k.endswith(("model", "strapdown")) or k in ("segment_id", "yaw_deg", "distance_m")}))

    # Drift % is meaningless when the car barely moved (a stopped car makes
    # the denominator ~0 and one segment can dominate a mean), so segments
    # under --min-distance over the blackout are listed but not averaged.
    scored = [r for r in rows if r["distance_m"] >= a.min_distance]
    keys = [k for k in rows[0] if k.endswith(("_model", "_strapdown"))] if rows else []
    wins = {rs: sum(r[f"{rs}_model"] < r[f"{rs}_strapdown"] for r in scored) for rs in a.resample}
    summary = {"segments": len(rows), "scored": len(scored), "min_distance_m": a.min_distance,
               "blackout": a.blackout, "extra": a.extra, "model_beats_strapdown": wins,
               **{k: {"mean": round(float(np.mean([r[k] for r in scored])), 2),
                      "median": round(float(np.median([r[k] for r in scored])), 2)} for k in keys}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
