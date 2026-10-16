"""Collect results/*.json into results.md (median drift %, clean test drives)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ["ref_physics_orig", "ref_physics_fwdfix", "phys_zupt", "phys_bias_gyro", "phys_bias_gyro_120s",
         "phys_bias_accel", "phys_bias_both", "ref_v3chain_orig", "baseline", "calibfix", "augbias", "anchor",
         "chain6"]
COLS = [("io_5s", "IO 5 s"), ("io_30s", "IO 30 s"), ("io_60s", "IO 60 s"), ("io_30s_slow", "IO 30 s slow"),
        ("io_30s_fast", "IO 30 s fast"), ("comma_5s", "comma 5 s"), ("comma_30s", "comma 30 s")]

recs = {p.stem: json.load(open(p)) for p in (ROOT / "results").glob("*.json")}
names = [n for n in ORDER if n in recs] + sorted(n for n in recs if n not in ORDER)
lines = ["| run | what changed | " + " | ".join(c[1] for c in COLS) + " |",
         "|---|---|" + "---|" * len(COLS)]
for n in names:
    h = recs[n]["headline"]
    lines.append(f"| {n} | {recs[n]['desc']} | " + " | ".join(f"{h[c]:.1f}" for c, _ in COLS) + " |")
split = json.load(open(ROOT / "split.json"))["summary"]
md = f"""# Drift experiments: results

Median drift, % of distance travelled, on test drives that no model saw in
training (drive-level split, see split.json). Lower is better; target < 10%.
IO = IO-VNBD ({split['iovnbd']['test']['drives']} test drives, {split['iovnbd']['test']['minutes']:.0f} min),
comma = comma2k19 ({split['comma']['test']['drives']} test segments from 3 routes).
"5 s" = one window started from the true state; "30 s"/"60 s" = chained
blackouts, each 5 s chunk starting from the previous chunk's own estimate.
slow/fast = chain mean speed below/above 6 m/s.

ref_v3chain_orig was trained on the old leaky split, so some of these test
drives were in its training data: reference only, not a fair comparison.

{chr(10).join(lines)}
"""
open(ROOT / "results.md", "w").write(md)
print(md)
