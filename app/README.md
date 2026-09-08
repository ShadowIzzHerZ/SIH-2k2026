# SIH26168 — Dead Reckoning Android App

Native Kotlin app that runs the trained pipeline live, on-device: real
accelerometer/gyroscope + GNSS, the exact same calibration
(`src/calibration.py`), integration (`src/models/strapdown_ins.py`), and
GNSS↔INS mode-switching (`src/fusion.py`, the fixed version — see its
docstring for the bugs found and fixed getting there) logic ported to
Kotlin, running the exported ONNX model (`checkpoints/dead_reckoning_model.onnx`)
via ONNX Runtime Mobile.

This is the on-device counterpart to `src/simulate_blackout.py` — same
state machine, same discrete 5s-chunk BLACKOUT design, same reconnect
BLEND — but driven by live sensor/location ticks instead of a recorded
trace, with a **"Simulate GNSS blackout" toggle** for demoing on stage
without needing to physically walk into a real tunnel on cue.

## Architecture

| File | Role |
|---|---|
| [app/src/main/java/.../Calibration.kt](app/src/main/java/com/sih26168/deadreckoning/Calibration.kt) | Direct Kotlin port of `src/calibration.py` — leveling + yaw rotation math |
| [app/src/main/java/.../CalibrationManager.kt](app/src/main/java/com/sih26168/deadreckoning/CalibrationManager.kt) | Drives calibration live on startup (stationary leveling, then yaw lock-in during an initial confident/moving stretch) |
| [app/src/main/java/.../SensorReader.kt](app/src/main/java/com/sih26168/deadreckoning/SensorReader.kt) | Wraps SensorManager for raw accel/gyro |
| [app/src/main/java/.../LocationReader.kt](app/src/main/java/com/sih26168/deadreckoning/LocationReader.kt) | Wraps LocationManager — no Play Services/API key needed. Uses `Location.getSpeed()`/`getBearing()` (the platform's own Doppler-derived values), not position-differencing — see `fusion.py`'s docstring for why that distinction mattered a lot in testing |
| [app/src/main/java/.../BiasCorrectionModel.kt](app/src/main/java/com/sih26168/deadreckoning/BiasCorrectionModel.kt) | ONNX Runtime wrapper for the exported network |
| [app/src/main/java/.../FusionEngine.kt](app/src/main/java/com/sih26168/deadreckoning/FusionEngine.kt) | The live GNSS↔INS state machine — same design as the fixed `src/fusion.py` |
| [app/src/main/java/.../TrajectoryView.kt](app/src/main/java/com/sih26168/deadreckoning/TrajectoryView.kt) | Live trajectory trail, colored by mode — same visual language as `results/blackout_demo_*.png` |
| [app/src/main/java/.../MainActivity.kt](app/src/main/java/com/sih26168/deadreckoning/MainActivity.kt) | Wires it all together, 10Hz tick loop |

The app module's `syncModel` Gradle task copies `checkpoints/dead_reckoning_model.onnx`
into `assets/` at build time — it is never hand-copied/duplicated, so the
app always bundles whatever the training pipeline last exported. Re-run
`src/export_onnx.py` and rebuild if the checkpoint changes.

## Building

Requires: JDK 17, Android SDK (platform 34, build-tools 34.0.0). On this
machine both were installed under `~/Library/Android/sdk` and
`/opt/homebrew/opt/openjdk@17` specifically for this project — `local.properties`
(gitignored) points at the SDK path.

```bash
cd app
export JAVA_HOME=/opt/homebrew/opt/openjdk@17   # or wherever your JDK 17 is
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Restricted to `arm64-v8a` only (every realistic demo/judging device) —
bundling onnxruntime-android's other ABIs roughly quadruples APK size for
architectures nothing here will run on.

## Running

Needs a **real device** — GPS and motion sensors don't behave meaningfully
in an emulator. Install via `adb install app-debug.apk` or copy the APK to
the phone and tap to install (allow "install from unknown sources").

On first launch:
1. Grant location permission.
2. Hold the phone still for ~2s — leveling (roll/pitch from gravity).
3. Drive/walk in a roughly straight line with GPS available for a few
   seconds — yaw alignment (matches the phone's heading to the vehicle's).
4. Once both complete, the trajectory view starts drawing: blue while
   GNSS-tracked, red during a blackout (real or the simulate toggle),
   orange during the reconnect blend.

## Known limitations / not yet in the app

- **Map-matching** (`src/map_matching.py`) is not wired in — the app shows
  the raw fused trajectory, not snapped to roads. Would need an offline
  OSM extract bundled for the demo venue (network map-matching mid-demo
  isn't reliable) — a real next step, not started.
- Not tested on a physical device yet — built and verified structurally
  (correct manifest/permissions/assets, valid signed debug APK) but the
  live sensor/calibration/UI flow needs a real run to confirm.
- No persistence — closing the app loses the current trajectory.
