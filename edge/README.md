# Edge engine (non-phone IMU path)

The problem statement asks for a second deliverable next to the Android app:
an edge-deployable engine that works with non-phone IMUs, with a ~200 Hz
position update rate on a FOG-IMU (docs/sih.md §2 and §4). This is it: a
Python service around the same exported ONNX model and the same GNSS/INS
state machine as `src/fusion.py`, taking a raw IMU stream at any rate.

Runtime needs only `numpy` and `onnxruntime` (plus `pandas`/`pyarrow` for the
comma2k19 converter). No PyTorch.

## Try it

```bash
# synthetic 200 Hz FOG drive, 20 s GNSS blackout
python -m edge.cli synth --out edge/out/drive.csv
python -m edge.cli run edge/out/drive.csv --blackout 30:50

# real automotive IMU at its native ~100 Hz (comma2k19), 15 s blackout
python -m edge.cli from-comma2k19 data/comma2k19_demo/data/demo-00000-of-00003.parquet --segment 0
python -m edge.cli run edge/out/c2k.csv --blackout 20:35 --axes x,-y,-z

# speed on this machine
python -m edge.cli bench

# checks (exact parity with src/fusion.py runs when torch is installed)
python -m edge.test_edge
```

`run` writes `track_10hz.csv` (the scored estimate), `track_live.csv` (one
position per IMU sample) and `summary.json`. With `--blackout`, GNSS is hidden
from the engine for that span but kept for scoring, and the same run is
repeated with no learned correction, so the summary shows what the model adds.

## Input

One CSV, one row per sensor event (`edge/io.py` has the details):

```
t,ax,ay,az,gx,gy,gz,lat,lon,speed,heading
```

IMU rows and GNSS rows can be separate or combined, at any rates. A gap in
GNSS rows is a blackout. Units: `--accel-unit m/s2|g`, `--gyro-unit rad/s|deg/s`.
`speed` (Doppler, m/s) and `heading` (compass course, degrees) are optional but
`src/fusion.py` explains why they matter for seeding a blackout.

## How it works

| File | What it does |
|---|---|
| `engine.py` | `ModelRateFusion`: the `run_fusion` state machine made causal (chunks resolve when their last sample arrives). `EdgeEngine`: mounting rotation, resampling, and a per-IMU-sample output that re-anchors to each 10 Hz result and fades out the difference instead of jumping. |
| `resample.py` | 200 Hz to 10 Hz, streaming. `interp` (default) matches how the training data was made; `mean` is anti-aliased for sensors with strong vibration. |
| `mounting.py` | Sensor to vehicle frame: `--axes x,-y,-z` or `--mount-rpy r,p,y` for a known install, or auto (gravity leveling plus a GNSS yaw fit). |
| `cli.py` | `run`, `synth`, `from-comma2k19`, `bench`. |
| `evaluate_comma2k19.py` | Scores every comma2k19 demo segment at its native IMU rate. |

A short final chunk (GNSS back mid-chunk) is handled like the Android
`FusionEngine.kt`: the fixed-shape ONNX graph gets the last 50 real samples and
only the newest corrections are used. Full chunks match `run_fusion` exactly;
`edge/test_edge.py` checks this against `src/fusion.py` directly.

## Results

**Speed.** On a 4-core cloud x86 box: 0.6 ms per model call (one every 5 s of
blackout), 13 µs of Python per IMU sample, about 0.3% of one core at 200 Hz.
A 100 s, 200 Hz file runs about 370x faster than real time.

**Real 100 Hz IMU, comma2k19, 64 segments.** One injected blackout per
segment, segments where the car moved under 100 m (50 m for 5 s) left out of
the averages. Drift is end error over distance travelled, the project's usual
metric. Raw rows are in `benchmarks/`.

| Blackout | Sensor frame | Model median | Strapdown-only median | Model better on |
|---|---|---|---|---|
| 15 s | correct (`--axes x,-y,-z` + leveling) | 14.0% | 13.6% | 32 of 58 |
| 15 s | auto mount | 14.3% | 13.5% | 32 of 58 |
| 5 s | correct | 9.2% | 6.1% | 15 of 55 |
| 15 s | as in training (see below) | 13.6% | 18.9% | 34 of 58 |
| 5 s | as in training | 8.0% | 6.1% | 17 of 55 |

What this says, plainly:

1. The engine works end to end on a real high-rate IMU, but the current
   `checkpoints/dead_reckoning_model.onnx` (exported 2026-09-05 from the
   then-production `best.pt`) does not beat plain strapdown integration once
   the sensor frame is right. The model lands near 14% either way; strapdown
   only looks worse when its forward axis is scrambled.
2. The comma2k19 training data was leveled in a scrambled frame. comma's
   device is z-down, and `src.calibration.leveling_rotation` turns that
   ~180 degrees about a horizontal axis set by the small tilt, so each
   segment's "forward" channel pointed somewhere different (between
   -179 and +178 degrees off forward across the 64 demo segments). `calibrate_sequence` then skipped the yaw step for
   these one-minute segments. `mounting.level` flips z-down sensors first
   to avoid this. That is a question for the model work in `src/`, not
   something this directory changes.
3. `src.calibration.estimate_yaw_misalignment` subtracts a sensor-frame angle
   from a world-frame course, so it returns roughly the average compass
   heading plus the mount offset. `mounting.estimate_mount` uses a
   vehicle-frame fit instead and recovers a 90 degree install on synthetic data
   (fit quality 0.94). On steady highway driving it often has too little
   signal and falls back to zero yaw; prefer `--axes` when the install is known.
4. The synthetic drive is not a model benchmark. The network was trained on
   real vehicle data and does badly on clean simulated signals (32% vs 2% for
   strapdown); it is there to check units, mounting and plumbing.

When a new model is exported, rerun:

```bash
python -m edge.evaluate_comma2k19 data/comma2k19_demo/data/*.parquet --blackout 20:35 --extra "--axes x,-y,-z" --model <new.onnx>
```

## Not done yet

- No real FOG/200 Hz recording has been run; comma2k19's ~100 Hz MEMS is the
  closest real data in the repo. `synth` covers 200 Hz plumbing only.
- No live serial/UDP reader. `EdgeEngine.push_imu` / `push_gnss` are the
  streaming API; a reader for a specific sensor protocol is a few lines on top.
- No C++ port. Python is fast enough here (above) and the PS allows either.
