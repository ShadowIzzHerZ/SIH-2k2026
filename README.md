# SIH26168 — Dead Reckoning ML

The ML piece of the SIH26168 (ISRO) build: keep vehicle position tracking
alive through GPS blackouts (tunnels, underground parking, urban canyons)
using phone IMU (accelerometer + gyroscope) and a physics-informed learned
bias-correction network, trained on [IO-VNBD](https://github.com/onyekpeu/IO-VNBD).

See `sih.md` (on the Desktop, one level up) for the full hackathon strategy
doc this project implements the ML component of.

## Results

The model, dead-reckoning drift-%, evaluated on two independent real datasets
it was trained on (checkpoint: `checkpoints/best.pt`, see [Status](#status)
for the full evidence trail behind these numbers):

| Dataset | Scenario | Mean drift | Median drift | Pass rate (<10%) |
|---|---|---|---|---|
| [comma2k19](https://huggingface.co/datasets/commaai/comma2k19) | Highway cruise (steady speed, few turns) | **16.49%** | **8.94%** | 54.53% |
| [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) | Urban stop-and-go (low speed, frequent turns) | 62.33% | 67.56% | 3.71% |

**The honest read**: this is a real, working physics+learned-residual dead-
reckoning system that meets or nearly meets the PS's own <10%-drift target
on the scenario it's actually good at — steady, higher-speed driving, where
a phone's IMU has the least ambiguous signal to work with — and is
transparently weaker on the harder case: low-speed, frequent-turn urban
driving, where per-window drift diagnostics ([src/diagnose_drift.py](src/diagnose_drift.py),
[results/drift_diagnostics.csv](results/drift_diagnostics.csv)) show broad,
spread-out underfitting rather than one fixable bug. Four different fixes
were tried against the IO-VNBD number specifically (a bigger architecture,
mixing in comma2k19 as training data, a real LR-scheduler bug fix, and
properly-normalized engineered features) — all landed in the same 60-65%
band or worse. Rather than paper over that with a cherry-picked number, the
plan going in to the demo is to lead with what's genuinely earned (the
comma2k19 result) and be upfront that urban low-speed driving is the known
hard case — see the Status section below for the full, unfiltered trail of
what was tried and what actually happened.

## Approach

Raw IMU integration ("dead reckoning") drifts within seconds because MEMS
accelerometer/gyro bias and vibration noise compound under double
integration. Rather than a black-box regressor, this is a **hybrid
physics + learned-residual** design:

```
calibrated IMU window (6ch: ax,ay,az,gx,gy,gz)
        │
        ▼
 BiasCorrectionNet (CNN + GRU)  ──►  [Δv, Δθ]  residual corrections
        │
        ▼
physics baseline (forward_accel, yaw_rate) + [Δv, Δθ]
        │
        ▼
strapdown integrator (heading + speed → 2D position, non-holonomic)
        │
        ▼
predicted trajectory ──► loss = per-step speed error + end-to-end drift %
```

The integrator is differentiable, so the network is trained directly
against the metric the PS actually grades on — % positional drift over
distance travelled — not a proxy.

| File | Role |
|---|---|
| [src/calibration.py](src/calibration.py) | Phone→vehicle frame alignment (leveling + yaw) |
| [src/models/strapdown_ins.py](src/models/strapdown_ins.py) | Differentiable dead-reckoning integrator (shared by training and on-device inference) |
| [src/models/bias_correction_net.py](src/models/bias_correction_net.py) | CNN+GRU network producing the learned residual corrections |
| [src/data/io_vnbd_loader.py](src/data/io_vnbd_loader.py) | Parses raw IO-VNBD CSVs |
| [src/data/windowing.py](src/data/windowing.py) | Builds train/val/test windowed datasets, split at the trip level |
| [src/train.py](src/train.py) | Training loop |
| [src/evaluate.py](src/evaluate.py) | Benchmarks a checkpoint against the PS's `<10%` drift target |
| [src/export_onnx.py](src/export_onnx.py) | Exports the network to ONNX for the mobile app / edge engine |
| [src/map_matching.py](src/map_matching.py) | Snaps a (predicted or raw) trajectory onto real roads via HMM map-matching (OpenStreetMap + leuvenmapmatching) — the PS's separate "map-matching / non-holonomic constraints" component |
| [src/evaluate_with_mapmatching.py](src/evaluate_with_mapmatching.py) | Measures the real improvement map-matching gives on top of the trained model's own predictions on held-out test windows |
| [src/data/comma2k19_loader.py](src/data/comma2k19_loader.py) | Loads the [comma2k19](https://huggingface.co/datasets/commaai/comma2k19) demo split — a second, independent real dataset for a generalization check |
| [src/evaluate_comma2k19.py](src/evaluate_comma2k19.py) | Runs the IO-VNBD-trained model on comma2k19 with zero retraining |
| [src/diagnose_drift.py](src/diagnose_drift.py) | Per-window drift diagnostics — correlates predicted drift-% against distance travelled, yaw rate, speed to find what's actually driving high-drift windows |
| [src/fusion.py](src/fusion.py) | GNSS↔INS mode-switching — the PS's "instant seamless mode-switching" component. State machine that runs the trained network only during a GNSS blackout and blends smoothly back on reconnect |
| [src/simulate_blackout.py](src/simulate_blackout.py) | Demo/eval for `fusion.py`: injects an artificial GNSS blackout into a real held-out drive (IO-VNBD or comma2k19), runs the mode-switcher through it, and plots + reports the result |
| [tests/test_pipeline_smoke.py](tests/test_pipeline_smoke.py) | Synthetic-data sanity check for the integrator/model/train loop |
| [tests/test_fusion.py](tests/test_fusion.py) | Synthetic sanity checks for `fusion.py`'s state machine (GNSS passthrough, blackout reconstruction, no-jump reconnect) |
| [tests/test_map_matching.py](tests/test_map_matching.py) | Validates the map-matcher against a real road segment with a known-answer synthetic-drift-recovery check (needs network access) |
| [tests/test_comma2k19_loader.py](tests/test_comma2k19_loader.py) | Sanity-checks the comma2k19 loader against the real demo split (needs it downloaded first) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset (git-lfs, ~1.6GB of CSVs):
```bash
brew install git-lfs   # if not already installed
git lfs install
git clone --depth 1 https://github.com/onyekpeu/IO-VNBD.git data/IO-VNBD
cd data/IO-VNBD && git lfs pull --include="*.csv" --exclude="*.zip"
```
(The `.zip` archives are excluded — they're the same data as the
individual CSVs, just re-bundled, and would roughly double the download.)

comma2k19 (Hugging Face, ~227MB, no login needed — see
[src/data/comma2k19_loader.py](src/data/comma2k19_loader.py) for details):
```bash
python -c "
from huggingface_hub import hf_hub_download
for i in range(3):
    hf_hub_download('commaai/comma2k19', f'data/demo-0000{i}-of-00003.parquet',
                     repo_type='dataset', local_dir='data/comma2k19_demo')
"
```

## Status

- [x] Project scaffolded, dataset pulled
- [x] Physics integrator + network architecture implemented, verified against synthetic ground truth (see smoke test)
- [x] Real IO-VNBD column mapping resolved — dataset mixes `V-*.csv` (vehicle CAN-bus, no 3-axis IMU, unusable here) and `S-*.csv` (smartphone, real accel+gyro+GPS); we train on `S-*.csv` only (`data.file_prefix` in config)
- [x] Yaw-misalignment calibration implemented, using the GPS-orientation field (not raw position deltas — those were too noisy) as ground truth heading, confidence-thresholded per recording
- [x] First real training run + drift-benchmark eval completed (6-raw-channel input):
  - **Test set: 62.76% mean drift** (down from an untrained ~80-90% baseline)
  - Re-confirmed via 4 warm-restart cycles (`train.py --resume`, ~90 more epochs total): val drift never left a tight 68-70% band, and **train drift plateaued too** (~58-61%) — not overfitting or an LR problem, the 6-raw-channel network was genuinely underfitting the task. Preserved as-is: [checkpoints/best_6ch_baseline.pt](checkpoints/best_6ch_baseline.pt), [results/eval_report_6ch_baseline.json](results/eval_report_6ch_baseline.json), [results/train_history_6ch_baseline.json](results/train_history_6ch_baseline.json).
  - **Honestly, 62.76% is well above the PS's <10% target**, and this alone won't get there — needs the other PS-listed components (map-matching, GNSS/INS fusion mode-switching) doing real work too, not just polish.
- [x] **Retried 6 added engineered input channels properly** (accel/gyro magnitude, jerk, local smoothing+roughness — `windowing.py`'s `engineer_features()`, now wired in via `--extra_features`), fixing both gaps that made the first attempt inconclusive: real per-channel z-score normalization from train-split stats, and a fair `early_stop_patience=15` (`configs/engineered_features.yaml`) instead of the original run's 8. Combined with comma2k19 mixed into training, conclusive result this time — **worse than the 6-channel baseline on the metric that matters most**: 62.75% combined / 64.28% IO-VNBD-only test drift vs. 60.92%/62.33% without the extra channels. The extra channels do help the already-easy comma2k19 case further (14.48% mean / **5.42% median**, 71.45% pass rate — better than the 16.49%/8.94%/54.5% below), just not the harder IO-VNBD one. **Reverted to the 6-channel baseline as the production checkpoint** — `checkpoints/best.pt` unchanged; `checkpoints/best_engineered.pt` + its eval reports kept as reference. Full rationale: `engineer_features()`'s docstring.
- [x] ONNX export re-verified against the current production checkpoint (6-channel + comma2k19-mixed) — matches PyTorch to 5.72e-6, see [checkpoints/dead_reckoning_model.onnx](checkpoints/dead_reckoning_model.onnx)
- [x] Map-matching implemented and validated: HMM-based (leuvenmapmatching, Newson-Krumm family) snapping onto real OpenStreetMap road graphs, **not** naive nearest-point snapping (which breaks on noisy/drifted input — no memory of the route so far). Synthetic-drift recovery test shows a real **62% error reduction** (20.1m → 7.6m mean error) on a real road segment — see [tests/test_map_matching.py](tests/test_map_matching.py).
- [x] Map-matching's improvement on the *trained model's own* predictions (not synthetic drift) — measured, and **honestly, it doesn't help here**: 20 real test windows, mean drift 58.54% → 60.56% (**-3.5%**, i.e. slightly worse), avg match rate 0.65. One window regressed badly (19.6% → 53.9%) when the matcher snapped a decent prediction onto the *wrong* nearby road. Full log: [results/mapmatching_on_real_predictions.log](results/mapmatching_on_real_predictions.log).
  - This isn't a contradiction of the synthetic-drift test above (which showed a real 62% improvement) — it's a scale mismatch. Map-matching corrects meter-to-tens-of-metres noise/offset around a *basically-correct* route (exactly what the synthetic test injected: 40m of drift on top of an otherwise-accurate path). The trained model's raw predictions are 50-80% drift — the predicted trajectory has usually left the true road corridor entirely, so the matcher (which has no ground truth to check against) confidently snaps onto *some* plausible nearby road, not necessarily the right one.
  - Practical implication: map-matching is real, working infrastructure (see the synthetic test) but isn't a rescue for a bad underlying trajectory — it's only worth applying once the dead-reckoning model's own drift is low enough that the true route is still recoverable from the prediction (rough guess: sub-~20% based on the one window here that stayed roughly flat at that level). Getting the network's own drift down remains the binding constraint, not map-matching.
- [ ] **Tried a bigger "v2" architecture** (dilated residual CNN, 64/128/256 channels + deeper unidirectional GRU, 256-hidden 3-layer, ~1.9M params) after v1 plateaued on *train* drift too (not just val, across 6 warm-restart cycles) — evidence of underfitting, not overfitting. Result on a real Colab run: **71.35% test drift, worse than v1's 62.76%**. **Reverted to v1** — `checkpoints/best.pt` is the 62.76%-drift model again. Full rationale + result: [src/models/bias_correction_net.py](src/models/bias_correction_net.py)'s module docstring. A next attempt should isolate *why* v2 underperformed (residual/dilation structure vs. raw parameter count) instead of changing several things at once.
- [x] **Cross-dataset generalization check**: ran the IO-VNBD-trained model (zero retraining) on [comma2k19](https://huggingface.co/datasets/commaai/comma2k19) — a second, independently-collected real dataset (different device, different country, highway driving). Result: **29.27% mean drift, 15.64% median** — notably *better* than IO-VNBD's own 62.76% test number, and 10% of windows already clear the PS's <10% target outright. This is a genuinely good sign: the model isn't just memorizing IO-VNBD's specific quirks, and its accuracy tracks scenario difficulty sensibly (steady highway cruise is a much easier dead-reckoning case than IO-VNBD's turning/stop-and-go urban driving — see [src/data/comma2k19_loader.py](src/data/comma2k19_loader.py) for the dataset's own caveats: demo split only, highway-only, EON dashcam device not literally a phone). Full numbers: [results/comma2k19_eval.json](results/comma2k19_eval.json). Not yet used as *training* data — this was a generalization check, not a retrain.
- [x] **Mixed comma2k19 into training** (not just the zero-shot check above) — combined IO-VNBD+comma2k19 dataset, trained from cold start then warm-restarted overnight (9+ cycles, `scripts/train_until_target_comma.sh`). Result, broken down by dataset on the same checkpoint ([results/eval_report_combined_run1.json](results/eval_report_combined_run1.json) / [results/eval_report_breakdown_run1.json](results/eval_report_breakdown_run1.json)):
  - **IO-VNBD-only test drift: 62.33%** — flat vs. the 62.76% IO-VNBD-only baseline, i.e. mixing in comma2k19 did **not** measurably help IO-VNBD itself.
  - **comma2k19-only test drift: 16.49% mean / 8.94% median** — meaningfully better than the zero-shot 29.27%/15.64% above, and the median now clears the PS's <10% target. Training on comma2k19 (not just evaluating on it) genuinely helps comma2k19 performance; it just doesn't transfer back to IO-VNBD's harder scenario.
  - Along the way, found and fixed a real bug in the warm-restart LR schedule (`src/train.py`): `CosineAnnealingLR`'s `T_max` was always the full config epoch budget (60), but `early_stop_patience=8` was cutting every resumed cycle off after only 8-16 real epochs — so LR barely moved off its peak before each cycle ended, and every restart was retracing almost the same high-LR trajectory. Fixed by sizing `T_max` to the guaranteed cycle length on resume (`early_stop_patience + 2`) so LR actually reaches a fine-tuning-scale value even in the worst case. Confirmed via the checkpoint's own history that this was a real inefficiency, not the reason for the plateau: even after the fix, **9 more warm-restart cycles all still failed to beat the pre-fix 63.26% best val drift** — this is a genuine ceiling for this architecture + data combination, not an LR artifact.
- [x] **Per-window drift diagnostics** ([src/diagnose_drift.py](src/diagnose_drift.py), [results/drift_diagnostics.csv](results/drift_diagnostics.csv)) — checked whether the 60%+ mean / 809% worst-case test drift is driven by a small number of outlier windows or one identifiable failure mode (e.g. tight turns, as earlier bullets assumed), before trying another blind architecture/data change. It's neither:
  - Removing the worst 1% of windows only drops mean drift 60.92% → 59.25%; removing the worst 5% only gets to 57.83% — the tail isn't dominating the average.
  - Correlation of drift-% with distance travelled (r=0.34), max yaw rate (r=0.22), mean yaw rate (r=0.26), and starting speed (r≈-0.04) are all weak — no single factor explains per-window error.
  - The actual pattern: median drift (66.96%) sits *above* the mean, i.e. bimodal — a cluster of easy, long, high-speed highway-like windows near 0% drift (matching comma2k19's strength) pulling the mean down, against a majority of short, low-speed, urban stop-and-go windows sitting around 60-90%.
  - **Conclusion: broad, spread-out underfitting on low-speed/urban driving specifically, not a fixable outlier bug or a single targeted failure mode.** Consistent with every other lever tried failing to move the number (warm restarts, the LR fix above, the v2 architecture, comma2k19 mixing).
- **Given the above and the Sept 10 shortlist deadline, decided to stop chasing the combined/IO-VNBD drift number further and reframe the pitch honestly around what the evidence actually supports**: the model is genuinely strong on highway-style driving (comma2k19: 16.49% mean / 8.94% median, at/near the <10% target) and openly weaker on low-speed urban stop-and-go (IO-VNBD's harder case) — a real, defensible, and honestly-earned result, rather than an unqualified claim against the PS's flat 62.76%/60.92% headline numbers.
- [x] **GNSS+INS fusion mode-switching implemented** ([src/fusion.py](src/fusion.py), demo/eval: [src/simulate_blackout.py](src/simulate_blackout.py)) — the PS's own "instant seamless mode-switching" line item, previously entirely unbuilt. A state machine (GNSS_TRACKING / BLACKOUT / BLEND) that runs the trained network only during a blackout and ramps smoothly back to GNSS on reconnect (measured reconnect jump: 0.0m — no teleport) rather than snapping.
  - Building this surfaced a real train/deploy mismatch, not a mode-switching bug: the first version fed the network a rolling IMU window and used only its *last* position's correction each step (the exact per-sample contract [src/export_onnx.py](src/export_onnx.py) already documents for the on-device app). That measured **5-6x worse drift than `evaluate.py`'s reported numbers on identical held-out data for an identical span** — because `BiasCorrectionNet`'s output isn't position-invariant across its own 50-sample training window (position 0, almost no GRU context yet, behaves very differently from position 49), and training/eval always score it using *all* window positions integrated together from one fresh ground-truth anchor, never a single position alone.
  - Fixed by processing each blackout in discrete, non-overlapping 5s chunks — the *whole* chunk's correction sequence integrated together from an anchor state at the end of the previous chunk (real GNSS state for the first chunk, the module's own prior estimate beyond that — full rationale in `fusion.py`'s docstring), reproducing the trained regime instead of a simpler-looking but untested one.
  - Two more real bugs found chasing an implausibly bad IO-VNBD number, both fixed and both real regardless of the number they moved: (1) seeding speed/heading from finite-differencing consecutive GNSS positions — a known-bad practice this project already documented and fixed for *yaw* calibration ([src/calibration.py](src/calibration.py)) but that crept back in here — confirmed producing a **103 m/s speed spike from GPS position noise alone** at real ~3 m/s IO-VNBD driving; fixed by using the dataset's own GPS-chip speed_gt/heading_gt fields instead. (2) no upper bound on integrated speed, letting one bad chunk's error run away with nothing physically stopping it (confirmed reaching **150+ m/s** before the fix); fixed with a physically-sane clamp (50 m/s / 180 km/h).
  - Real held-out results with all three fixes: **comma2k19 (highway) 30s blackout: 2.9% drift — under the PS's 10% target**; 5s spans averaged 10.7% mean / 5.7% median across 20 segments, in line with the existing 16.49%/8.94% headline numbers.
  - **Does not fix, and was never going to fix, IO-VNBD's already-known low-speed/urban weakness** — chaining ~9 un-reset 5s hops over a real continuous blackout compounds an already high-variance per-window model (up to 809% worst-case per `evaluate.py`) further than any single window shows. Measured across 8 different real 45s continuous blackouts on the same trace: **77-236% drift, mean 145%, median 109%** — genuinely worse than the 60.92% per-window headline, and not something further fusion-layer engineering fixes, only re-training would. Consistent with, not contradicting, every other IO-VNBD finding above — see `fusion.py`'s docstring for the full comparison. **For the pitch: lead with the comma2k19 result, which is genuinely strong and unaffected by this; if IO-VNBD comes up, this is the same known limitation the project's honest framing (above) was already built around, not new bad news.**
- [x] **Non-holonomic motion constraints in the matcher made explicit and verified** ([src/map_matching.py](src/map_matching.py), test: [tests/test_non_holonomic_matching.py](tests/test_non_holonomic_matching.py)). Two guarantees, checked separately against a small synthetic road (no network needed, unlike `test_map_matching.py`'s real-OSM drift-recovery test):
  - **Wrong-way travel on a one-way street is structurally impossible**, not just discouraged — `osmnx_graph_to_inmem_map` only ever adds the directed edge(s) osmnx itself resolved, so a one-way street's reverse direction simply doesn't exist as an edge to match onto. Was already true, but only asserted in a comment before; now checked directly against the real function.
  - **A single noisy backward-looking observation shouldn't teleport the matched path backward** — leuvenmapmatching's `avoid_goingback` transition penalty was already the library's implicit default (undocumented in this repo, silently inherited), now passed explicitly so it can't silently change/disable under us. Measured on a synthetic one-way road with an injected 18m backward jitter: matched path holds flat (0.0m backward movement) instead of following the jitter back — confirmed the explicit setting is actually taking effect, not just present in code. Re-ran the existing real-OSM drift-recovery test (`test_map_matching.py`) too — unaffected: still 20.1m → 7.6m mean error, 100% match rate.
  - Remaining gap, explicitly not claimed as solved: this is a *soft* penalty (halves transition probability, not a hard ban) — a strongly-evidenced backward observation can still occasionally win. No test asserts the matcher enforces one-way directionality under GPS noise *pressuring* it the wrong way (only that the edge itself doesn't exist), since that would need a synthetic case with a parallel wrong-way road nearby to be meaningful.
- [ ] Own campus recordings (see `data/own_recordings/`) — not yet collected; could help close the gap given IO-VNBD's mounting/session variety is a real source of error
- [x] **Android app started** (see [app/](app/)) — native Kotlin port of the calibration/integration/mode-switching pipeline, running the exported ONNX model on-device via ONNX Runtime Mobile. Builds a real, signed debug APK (`./gradlew assembleDebug`, ~24MB, arm64 only). Not yet tested on a physical device; map-matching isn't wired in yet. See `app/README.md` for the full picture.

## Running

```bash
# 1. confirm dataset columns (one-time)
python src/data/inspect_dataset.py

# 2. train (cold start)
python -m src.train --config configs/default.yaml

# 2b. or resume/warm-restart from the current best checkpoint instead of
# random init — measures the loaded weights' real val drift first so
# checkpoints/best.pt only gets overwritten if this run actually beats it.
# Backs up the previous checkpoints/best.pt -> best_prev.pt and
# results/train_history.json -> train_history_prev.json before it starts.
python -m src.train --config configs/default.yaml --resume checkpoints/best.pt

# 3. evaluate against the PS's <10% drift benchmark
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt

# 4. export for the mobile app
python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best.pt

# sanity check any time you change the integrator/model/loss
python -m tests.test_pipeline_smoke

# sanity check any time you change fusion.py's state machine
python -m tests.test_fusion

# demo/eval the GNSS<->INS mode-switcher on a real held-out drive
python -m src.simulate_blackout --dataset comma2k19 --blackout_start_s 10 --blackout_duration_s 30
python -m src.simulate_blackout --dataset iovnbd --blackout_duration_s 45

# validate the map-matcher against a real road segment (needs network access)
python -m tests.test_map_matching

# measure map-matching's real improvement on the trained model's predictions
python -m src.evaluate_with_mapmatching --config configs/default.yaml --checkpoint checkpoints/best.pt
```

### Training on Colab

[notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) is generated from
the real `src/` files by [scripts/build_colab_notebook.py](scripts/build_colab_notebook.py)
(re-run that script and re-upload any time `src/*.py` or the config change,
so the notebook never drifts from what's actually in the repo). Upload it to
[colab.research.google.com](https://colab.research.google.com), set
Runtime → GPU, and run the cells top to bottom.

It resumes rather than cold-starts: it fetches `checkpoints/best.pt` from
Google Drive (a previous Colab run) or, failing that, from this GitHub repo,
then loops `src.train --resume ...` (a cosine-annealed warm restart each
cycle) + `src.evaluate`, checkpointing to Drive after every cycle so nothing
is lost on disconnect. It keeps going until the PS's <10% drift target is
hit or the Colab runtime itself ends — there's a very high cycle cap as a
pure safety backstop, not a real limit.
