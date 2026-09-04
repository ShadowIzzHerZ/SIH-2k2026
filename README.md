# SIH26168 — Dead Reckoning ML

The ML piece of the SIH26168 (ISRO) build: keep vehicle position tracking
alive through GPS blackouts (tunnels, underground parking, urban canyons)
using phone IMU (accelerometer + gyroscope) and a physics-informed learned
bias-correction network, trained on [IO-VNBD](https://github.com/onyekpeu/IO-VNBD).

See `sih.md` (on the Desktop, one level up) for the full hackathon strategy
doc this project implements the ML component of.

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
| [tests/test_pipeline_smoke.py](tests/test_pipeline_smoke.py) | Synthetic-data sanity check for the integrator/model/train loop |
| [tests/test_map_matching.py](tests/test_map_matching.py) | Validates the map-matcher against a real road segment with a known-answer synthetic-drift-recovery check (needs network access) |

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

## Status

- [x] Project scaffolded, dataset pulled
- [x] Physics integrator + network architecture implemented, verified against synthetic ground truth (see smoke test)
- [x] Real IO-VNBD column mapping resolved — dataset mixes `V-*.csv` (vehicle CAN-bus, no 3-axis IMU, unusable here) and `S-*.csv` (smartphone, real accel+gyro+GPS); we train on `S-*.csv` only (`data.file_prefix` in config)
- [x] Yaw-misalignment calibration implemented, using the GPS-orientation field (not raw position deltas — those were too noisy) as ground truth heading, confidence-thresholded per recording
- [x] First real training run + drift-benchmark eval completed (6-raw-channel input):
  - **Test set: 62.76% mean drift** (down from an untrained ~80-90% baseline)
  - Re-confirmed via 4 warm-restart cycles (`train.py --resume`, ~90 more epochs total): val drift never left a tight 68-70% band, and **train drift plateaued too** (~58-61%) — not overfitting or an LR problem, the 6-raw-channel network was genuinely underfitting the task. Preserved as-is: [checkpoints/best_6ch_baseline.pt](checkpoints/best_6ch_baseline.pt), [results/eval_report_6ch_baseline.json](results/eval_report_6ch_baseline.json), [results/train_history_6ch_baseline.json](results/train_history_6ch_baseline.json).
  - **Honestly, 62.76% is well above the PS's <10% target**, and this alone won't get there — needs the other PS-listed components (map-matching, GNSS/INS fusion mode-switching) doing real work too, not just polish.
- [ ] **Tried 6 added engineered input channels** (accel/gyro magnitude, jerk, local smoothing+roughness — see `windowing.py`'s `engineer_features()`, currently defined but not wired in) instead of just 6 raw axes. First cold-start cycle reached 71.73% val drift at epoch 2 then early-stopped at epoch 10 — noisier and no better than the 6-channel baseline at a comparable point, but on only 10 epochs vs. the 57 the baseline took to reach its best, so inconclusive rather than a disproof. **Reverted to the 6-channel baseline for now** given the deadline — `checkpoints/best.pt` is back to the 62.76%-drift model. See `engineer_features()`'s docstring for what a real retry would need (higher first-cycle patience, per-channel normalization).
- [x] ONNX export re-verified against the reverted checkpoint — matches PyTorch to 4.05e-6, see [checkpoints/dead_reckoning_model.onnx](checkpoints/dead_reckoning_model.onnx)
- [x] Map-matching implemented and validated: HMM-based (leuvenmapmatching, Newson-Krumm family) snapping onto real OpenStreetMap road graphs, **not** naive nearest-point snapping (which breaks on noisy/drifted input — no memory of the route so far). Synthetic-drift recovery test shows a real **62% error reduction** (20.1m → 7.6m mean error) on a real road segment — see [tests/test_map_matching.py](tests/test_map_matching.py).
- [ ] Map-matching's improvement on the *trained model's own* predictions (not synthetic drift) — [src/evaluate_with_mapmatching.py](src/evaluate_with_mapmatching.py) is built and runs, but the free OSM Overpass API rate-limits repeated automated queries hard, so a full-confidence run is still pending — rerun with fewer/cached windows or a paid/self-hosted Overpass instance for a reliable number.
- [ ] Non-holonomic motion constraints not yet enforced inside the matcher itself (relies on the road graph's own directionality for one-way streets, but no explicit "can't teleport backward along a one-way" cost yet)
- [ ] GNSS+INS fusion mode-switching (seamless handoff between GPS-available and blackout) — not started, sits above both the network and the matcher
- [ ] Own campus recordings (see `data/own_recordings/`) — not yet collected; could help close the gap given IO-VNBD's mounting/session variety is a real source of error

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
