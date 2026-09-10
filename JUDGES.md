# JUDGES.md — a map of this repo, part by part

This is a navigational overview: what every top-level part of this
submission is and does, so you can find your way around the actual code
rather than take a pitch's word for it. For the persuasive "how to talk
about this" version, see [docs/understanding.md](docs/understanding.md)
— it has the real measured numbers, the bugs found and fixed, and
pre-answered judge questions. For full technical depth on any one piece,
each subfolder has its own README (linked throughout below).

**Problem statement:** SIH26168 — "AI-ML based Intelligent Dead
Reckoning system for seamless navigation," Indian Space Research
Organisation (ISRO), Smart Vehicles theme. Phone GPS drops out in
tunnels, underground parking, dense city streets, forests — most Indian
vehicles have no factory INS backup, so navigation just stops. The ask:
use the phone's own accelerometer + gyroscope (IMU) to keep estimating
position through the blackout, with AI, at 10Hz, drifting under 10% of
distance travelled — plus phone-to-vehicle calibration, GNSS↔INS fusion
with instant mode-switching, and map-matching with non-holonomic
constraints. See [docs/sih.md](docs/sih.md) for the full PS text and
[docs/understanding.md](docs/understanding.md) for the grading criteria
spelled out.

---

## The shape of this submission: three independent parts

This repo actually contains **two separate, alternative implementations**
of dead reckoning, plus the training/evaluation pipeline behind one of
them. They are **not wired together** — worth knowing before you go
looking for how they connect, because they don't:

| Part | What it is | Tech |
|---|---|---|
| **[src/](#1-src--the-ml-pipeline-python-training--evaluation)** | Trains and validates the physics+learned-residual network | Python, PyTorch |
| **[app/](#2-app--the-android-app-the-primary-live-demo)** | Runs that trained network live, fully on-device, in a real Android app | Kotlin, ONNX Runtime Mobile |
| **[backend/](#3-backend--a-second-independent-implementation-classical-not-ml)** | A separate, classical (non-ML) implementation by a teammate — a 9-DOF Extended Kalman Filter server | Python, FastAPI |

`app/` is the one meant to be handed to a judge and held in your hand —
it's fully self-contained (runs the trained model with ONNX Runtime
Mobile, no network calls for the dead-reckoning pipeline itself). It does
**not** talk to `backend/`; `backend/` expects its own mobile client
streaming raw IMU over WebSocket, which nothing in this repo currently
implements. Know this going in so you don't get asked "so the app talks
to this backend?" and have to improvise an answer.

---

## 1. `src/` — the ML pipeline (Python, training + evaluation)

The core deliverable: a **physics-informed learned-residual network** —
not a black box. A differentiable strapdown integrator does the exact
physics; a small neural network only learns to correct the sensor's own
bias/noise, trained end-to-end against the actual grading metric (%
positional drift), not a proxy.

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

| File | What it does |
|---|---|
| [src/calibration.py](src/calibration.py) | Phone→vehicle frame alignment: leveling (gravity tells you which way is up) then yaw alignment (compares horizontal-accel direction against GPS heading during a moving stretch) |
| [src/models/strapdown_ins.py](src/models/strapdown_ins.py) | The differentiable physics integrator — 2D, non-holonomic (a car can't move sideways), shared verbatim by training and the on-device Kotlin port so there's zero train/inference mismatch |
| [src/models/bias_correction_net.py](src/models/bias_correction_net.py) | The "AI filter" the PS asks for — small CNN+GRU, outputs `[Δv, Δθ]` per timestep. A bigger v2 architecture was tried and reverted (measurably worse: 71.35% vs 62.76% drift) — see its docstring |
| [src/data/io_vnbd_loader.py](src/data/io_vnbd_loader.py) | Parses the PS's official [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) dataset (40hrs/1,300km vehicle + 58hrs/4,400km smartphone, UK/Nigeria/France), fuzzy-matching real header variants |
| [src/data/windowing.py](src/data/windowing.py) | Builds train/val/test windows, split at the *trip/file* level (not window level) to avoid leakage; also mixes in `data/comma2k19_demo/` and `data/own_recordings/` when asked |
| [src/data/comma2k19_loader.py](src/data/comma2k19_loader.py) | Loads [comma2k19](https://huggingface.co/datasets/commaai/comma2k19) — a second, independent real dataset (US highway, different device/country) used as a generalization check, never trained on by default |
| [src/train.py](src/train.py) | The training loop — `python -m src.train --resume checkpoints/best.pt` |
| [src/evaluate.py](src/evaluate.py) | Benchmarks a checkpoint against the PS's own `<10%` drift target |
| [src/evaluate_comma2k19.py](src/evaluate_comma2k19.py) | Runs the IO-VNBD-trained model on comma2k19 with zero retraining — the generalization check |
| [src/diagnose_drift.py](src/diagnose_drift.py) | Per-window drift diagnostics — why urban low-speed driving is harder (broad underfitting, not one fixable bug) |
| [src/fusion.py](src/fusion.py) | The GNSS↔INS mode-switching state machine (GNSS_TRACKING / BLACKOUT / BLEND) — the PS's "instant seamless mode-switching" component, ported live to `app/`'s `FusionEngine.kt` |
| [src/simulate_blackout.py](src/simulate_blackout.py) | Injects a real blackout into a real held-out drive and runs the fusion state machine through it — the offline counterpart to the app's "Simulate GNSS blackout" toggle |
| [src/map_matching.py](src/map_matching.py) | Real HMM (Hidden Markov Model) map-matcher — snaps a trajectory onto actual OSM roads, with non-holonomic constraints (a one-way street's reverse direction isn't even an edge to match onto) |
| [src/evaluate_with_mapmatching.py](src/evaluate_with_mapmatching.py) | Measures map-matching's real improvement on the trained model's own predictions |
| [src/export_onnx.py](src/export_onnx.py) | Exports the trained network to ONNX for the Android app — fixed input shape on purpose (see its docstring) |

**Real, measured results** (checkpoint `checkpoints/best.pt`, full evidence
trail in [results/README.md](results/README.md)):

| Dataset | Scenario | Mean drift | Median drift | Pass rate (<10%) |
|---|---|---|---|---|
| comma2k19 | Highway cruise, steady speed | **16.49%** | **8.94%** | 54.53% |
| IO-VNBD | Urban stop-and-go, low speed | 62.33% | 67.56% | 3.71% |

Meets/nearly meets the PS's target on highway-style driving; honestly
weaker on harder low-speed urban driving — diagnosed, not hidden (four
different fixes tried, all landed in the same ~60-65% band — see
`src/diagnose_drift.py` and §6 of `docs/understanding.md`).

Map-matching cut error **62%** on a real road segment with injected
drift (20.1m → 7.6m mean error). The fusion layer measured **2.9% drift**
over a real 30-second continuous highway blackout, with a **0.0m**
reconnect jump (no visible teleport when GPS returns).

**Try it:**
```bash
python -m src.evaluate --config configs/default.yaml   # real drift numbers, printed
python -m src.simulate_blackout                         # fusion state machine over real data
```

---

## 2. `app/` — the Android app (the primary live demo)

The PS explicitly requires a **working mobile app doing live on-device
inference**, not a notebook. This is it: every piece above (calibration,
physics integrator, the trained network, fusion mode-switching,
map-matching) ported to native Kotlin and running for real on a phone —
ONNX Runtime Mobile for inference, no network call anywhere in the
dead-reckoning pipeline itself (only the map's background tiles need
internet).

| File | What it does |
|---|---|
| [Calibration.kt](app/app/src/main/java/com/sih26168/deadreckoning/Calibration.kt) | Direct Kotlin port of `src/calibration.py`'s math — leveling + yaw rotation, plus the compass-bearing↔internal-heading conversion (a real bug found and fixed this session — see its doc) |
| [CalibrationManager.kt](app/app/src/main/java/com/sih26168/deadreckoning/CalibrationManager.kt) | Drives calibration live: leveling verifies the phone actually held steady before committing, yaw alignment waits for its estimate to settle across checkpoints — both degrade honestly ("⚠ rough calibration") rather than hang or silently lie |
| [SensorReader.kt](app/app/src/main/java/com/sih26168/deadreckoning/SensorReader.kt) | Raw accelerometer/gyroscope via `SensorManager` |
| [LocationReader.kt](app/app/src/main/java/com/sih26168/deadreckoning/LocationReader.kt) | Raw GPS via plain `LocationManager` — no Google Play Services, no API key |
| [BiasCorrectionModel.kt](app/app/src/main/java/com/sih26168/deadreckoning/BiasCorrectionModel.kt) | ONNX Runtime Mobile wrapper running the exported trained network |
| [FusionEngine.kt](app/app/src/main/java/com/sih26168/deadreckoning/FusionEngine.kt) | The live GNSS↔INS state machine — same design as `src/fusion.py`, including ZUPT (zero-velocity update) for when the phone is genuinely at rest |
| [RoadMapView.kt](app/app/src/main/java/com/sih26168/deadreckoning/RoadMapView.kt) | Real OpenStreetMap-tile map (osmdroid) — mode-colored trail, a blinking "live" marker, translucent glass status cards |
| [MapMatcher.kt](app/app/src/main/java/com/sih26168/deadreckoning/MapMatcher.kt) + [RoadGraph.kt](app/app/src/main/java/com/sih26168/deadreckoning/RoadGraph.kt) | Live greedy sequential map-matcher against a bundled pre-fetched OSM road extract — same non-holonomic/continuity reasoning as `src/map_matching.py`, simplified for online (can't see the future) matching |
| [ReplayDataSource.kt](app/app/src/main/java/com/sih26168/deadreckoning/ReplayDataSource.kt) | Replays a real, previously-validated recorded drive bundled as an asset — for demoing indoors with no GPS reception |
| [DevRecorder.kt](app/app/src/main/java/com/sih26168/deadreckoning/DevRecorder.kt) | Hidden Developer Mode: records real sensor+GPS to a CSV that drops straight into `src/data/io_vnbd_loader.py` with zero config, for collecting real training data on a walk/drive |
| [MainActivity.kt](app/app/src/main/java/com/sih26168/deadreckoning/MainActivity.kt) | Wires it all together in a 10Hz tick loop — matches the PS's 10Hz requirement exactly |

**What you'll actually see on the screen:** a real OSM map, a blinking
"you are here" marker, a mode-colored trail (blue = GPS-tracked, red =
blackout/INS-only, orange = reconnect blend, green = map-matched), a
glass status card with live speed/heading and a calibration progress bar,
a **"Simulate GNSS blackout"** toggle (for demoing without physically
driving into a tunnel), a **"Replay real recorded drive"** toggle (for
demoing indoors with zero GPS), and a hidden Developer Mode (long-press
the title) with a data-recording tool and a testing-only calibration
skip.

**Try it:** see [app/README.md](app/README.md) for the build/install
steps — needs a real device (GPS/motion sensors don't behave meaningfully
in an emulator).

---

## 3. `backend/` — a second, independent implementation (classical, not ML)

Brought into this repo so the whole team's submission lives in one place.
Built separately by a teammate, and a genuinely **different technical
approach** from `src/`/`app/` — a classical **9-DOF Extended Kalman
Filter** with Zero-Velocity Updates and pedestrian step-counting,
deliberately **without any ML**. FastAPI, WebSocket ingestion at 10Hz,
JWT auth, NTP clock sync, a starvation-free priority queue (live packets
never wait behind backlog recovery), Redis/in-memory caching,
TimescaleDB/SQLite persistence, and a built-in dashboard.

This is a **server**, not something that runs on the phone — a mobile
client would stream raw IMU samples over WebSocket to it and receive
fused position back. Nothing in this repo currently plays that client
role (`app/` runs its own pipeline fully on-device instead). Full
protocol/endpoint details, the 3 operational modes (Stationary/ZUPT,
Moving/step-counting, Adaptive Storage), and run instructions are in
[backend/README.md](backend/README.md).

**Try it:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
python tests/simulate_device.py   # simulates a phone streaming to it
```

---

## Everything else, briefly

| Path | What it is |
|---|---|
| [data/](data) | `IO-VNBD/` (the PS's official dataset, gitignored — git-lfs, pulled separately) · `comma2k19_demo/` (gitignored generalization-check data) · `own_recordings/` (real phone recordings collected via the app's Developer Mode, mixed into training) |
| [checkpoints/](checkpoints) | Trained model weights + the ONNX export the app ships — see [checkpoints/README.md](checkpoints/README.md) for which one is production vs. kept-for-reference |
| [results/](results) | Every real eval report/training curve, nothing hand-edited — see [results/README.md](results/README.md) for the full index |
| [configs/](configs) | `default.yaml` — the one real config (dataset paths, model architecture, training hyperparameters, the PS's own drift/example-check targets) |
| [tests/](tests) | Deterministic regression tests — several exist specifically because a real bug was found and fixed, then a test written to prove the fix is load-bearing (reverted the fix, confirmed the test fails, restored it) |
| [scripts/](scripts) | `train_until_target*.sh` (unattended training loops), `build_colab_notebook.py` (generates a Colab notebook for GPU training) |
| [docs/sih.md](docs/sih.md) | The original strategy doc — why this PS was picked, the exact official problem text, deliverable requirements |
| [docs/understanding.md](docs/understanding.md) | **Read this next** — the full judge-facing pitch: every component explained with a "say it like this" line, the honest results story, every real bug found and fixed with evidence, known limitations, and pre-answered hard questions |
| [notebooks/](notebooks) | The Colab training notebook (GPU training away from a local machine) |

---

## The 20-second version

Physics does the exact math; a small trained network only corrects the
phone's own sensor bias, trained end-to-end against the real grading
metric. It meets the PS's target on highway-style driving, is honestly
weaker (and openly diagnosed) on harder low-speed urban driving, and
every other PS-named piece — calibration, GNSS↔INS fusion with a
measured 0.0m reconnect jump, and HMM map-matching with real
non-holonomic constraints — is built, tested against real data, and
running live on a real Android phone right now, not in a notebook.
