"""Collect results/*.json into results.md (median drift %, clean test drives)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ["ref_physics_orig", "ref_physics_fwdfix", "phys_zupt", "phys_bias_gyro", "phys_bias_gyro_120s",
         "phys_bias_accel", "phys_bias_both", "ref_v3chain_orig", "baseline", "calibfix", "augbias", "anchor",
         "chain6", "zeroinit", "calibfix+zupt", "calibfix_s7", "augbias_s7", "aug_chain6", "augbias_long", "speedaware",
         "speedaware_s7", "duo_speedaware", "duo_speedaware_s7"]
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

## Ranking (two seeds where run; seed-to-seed spread is ~0.5 pt on IO 30 s, ~3 pt on comma 30 s)

1. **Fixed phone-to-vehicle alignment** (calibfix): clear win on both seeds. Physics alone 27.6 -> 18.2% (IO 30 s);
   trained model 16.2 -> 15.1% avg (IO 30 s), 24.6 -> 22.3% (IO 60 s), slow 58.6 -> ~41%.
2. **+ bias/noise augmentation** (augbias): small but consistent on both seeds: IO 30 s 14.65% avg, 60 s 21.1%, comma 30 s 11.7%.
3. 30 s chained fine-tune (chain6): IO about as good as augbias, but slow and comma worse; not adopted.
4. No gain / worse: stop detection (train or inference), IMU bias from the 30 s of GPS before a blackout,
   learned absolute-speed anchor (much worse), zero-initialised correction head.

Round 2 (best two-seed averages):
5. **Current speed fed to the network** (speedaware): IO 30 s 14.2%, **60 s 19.1%**, comma 5 s 1.95%; slow traffic worse (~46%).
6. **Speed-switched pair** (duo: speedaware above 6 m/s, augbias_long below): IO 30 s 14.1%, **60 s 19.1%**, slow **39.1%**,
   comma 5 s 1.95%, comma 30 s 13.1%. Best balance so far; no extra training.
7. No gain: speed from turning (centripetal observer), augbias + 30 s chains (slow worse), double training budget
   (validation improved, test unchanged except slow 37.5%).

Round 3 (seed spread turned out to be large, so only all-seed averages count; no seed is ever picked by test results):
8. **Keep the last training epoch** (lastepoch) instead of the best-validation epoch: the 5 validation drives do not
   track the test drives, and picking epochs on them produced bad seeds (60 s up to 22-26%). Last-epoch seeds are tighter.
9. **Average several seeds** (combos.py): 6 last-epoch seeds averaged (ens_lastepoch6) = IO 30 s 12.4%, **60 s 16.0%**,
   slow 37.8%, comma 30 s 11.5%. Best overall.
10. **18 recovered training drives** (prep_extra.py, train only): with last-epoch training they help long blackouts
    (no-extra seeds 14.0-14.1% at 30 s vs 12.2-13.6%); no-extra models are a bit better in slow traffic, so the pair
    duo_last6_noextra7 (with-drives above 7 m/s, no-extra below) gives slow 36.1% at 12.6 / 16.2%.
11. No gain: richer augmentation on top of the extra drives, double training budget, noisy speed input, weight-averaged
    epochs, training on the validation drives, slow-only specialist once last-epoch is used, 11-model mixed average.

Error decomposition (physics, fixed alignment, IO 30 s): 18.2% as is, 7.3% with true speed, 14.5% with true
heading, so forward-speed integration is the dominant error source.

{chr(10).join(lines)}
"""
open(ROOT / "results.md", "w").write(md)
print(md)
