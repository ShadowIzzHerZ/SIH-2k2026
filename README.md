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
| [tests/test_pipeline_smoke.py](tests/test_pipeline_smoke.py) | Synthetic-data sanity check — run this after touching integrator/model/train code |

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
- [ ] **`configs/default.yaml`'s `data.column_map` needs the real IO-VNBD column names** — run `python src/data/inspect_dataset.py` once the dataset finishes downloading, paste the printed headers in. Loader has fuzzy-match fallback so it may already work, but confirm before trusting results.
- [ ] First real training run + drift-benchmark eval
- [ ] Calibration's yaw-misalignment estimation is stubbed at 0 — wire up `calibration.estimate_yaw_misalignment` using a GPS-heading segment once column names are confirmed
- [ ] ONNX export smoke-tested on a random checkpoint; needs re-verification post-training
- [ ] Map-matching / OSM non-holonomic snapping (separate component per the PS, not yet started — sits downstream of this model's raw trajectory output)

## Running

```bash
# 1. confirm dataset columns (one-time)
python src/data/inspect_dataset.py

# 2. train
python -m src.train --config configs/default.yaml

# 3. evaluate against the PS's <10% drift benchmark
python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt

# 4. export for the mobile app
python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best.pt

# sanity check any time you change the integrator/model/loss
python -m tests.test_pipeline_smoke
```
