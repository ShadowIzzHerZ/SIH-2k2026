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
without needing to physically walk into a real tunnel on cue. A
**"Replay real recorded drive" toggle** swaps the live sensor/GPS data
source for a real, previously-validated comma2k19 segment bundled as an
asset (the one that measured 2.9% drift in the offline evaluation) — for
demoing indoors with no GPS reception and no room to drive, using real
data instead of idealized synthetic motion. It's also how the app's own
bugs were actually found and fixed on a real device (see FusionEngine.kt's
docstring).

Map-matching (`RoadGraph.kt` + `MapMatcher.kt`) is wired in too — a real,
working matcher against a pre-fetched OSM road extract for the replay
route's area, snapping the live fused position onto roads (green overlay)
with the same non-holonomic/continuity reasoning as `src/map_matching.py`
(not naive nearest-point snapping). Verified live: during a simulated
blackout, the raw fused trail (red) visibly drifts off the road while the
map-matched trail (green) stays snapped to it — the same before/after
improvement `src/evaluate_with_mapmatching.py` reports offline, live here
instead.

## Architecture

| File | Role |
|---|---|
| [app/src/main/java/.../Calibration.kt](app/src/main/java/com/sih26168/deadreckoning/Calibration.kt) | Direct Kotlin port of `src/calibration.py` — leveling + yaw rotation math |
| [app/src/main/java/.../CalibrationManager.kt](app/src/main/java/com/sih26168/deadreckoning/CalibrationManager.kt) | Drives calibration live on startup (stationary leveling, then yaw lock-in during an initial confident/moving stretch) |
| [app/src/main/java/.../SensorReader.kt](app/src/main/java/com/sih26168/deadreckoning/SensorReader.kt) | Wraps SensorManager for raw accel/gyro |
| [app/src/main/java/.../LocationReader.kt](app/src/main/java/com/sih26168/deadreckoning/LocationReader.kt) | Wraps LocationManager — no Play Services/API key needed. Uses `Location.getSpeed()`/`getBearing()` (the platform's own Doppler-derived values), not position-differencing — see `fusion.py`'s docstring for why that distinction mattered a lot in testing |
| [app/src/main/java/.../BiasCorrectionModel.kt](app/src/main/java/com/sih26168/deadreckoning/BiasCorrectionModel.kt) | ONNX Runtime wrapper for the exported network |
| [app/src/main/java/.../FusionEngine.kt](app/src/main/java/com/sih26168/deadreckoning/FusionEngine.kt) | The live GNSS↔INS state machine — same design as the fixed `src/fusion.py` |
| [app/src/main/java/.../TrajectoryView.kt](app/src/main/java/com/sih26168/deadreckoning/TrajectoryView.kt) | Live trajectory trail, colored by mode — same visual language as `results/blackout_demo_*.png` — plus a green map-matched overlay |
| [app/src/main/java/.../ReplayDataSource.kt](app/src/main/java/com/sih26168/deadreckoning/ReplayDataSource.kt) | Reads the bundled real-drive replay asset (`assets/replay_drive.json`) |
| [app/src/main/java/.../RoadGraph.kt](app/src/main/java/com/sih26168/deadreckoning/RoadGraph.kt) | Loads the bundled pre-fetched OSM road extract (`assets/road_graph.json`) |
| [app/src/main/java/.../MapMatcher.kt](app/src/main/java/com/sih26168/deadreckoning/MapMatcher.kt) | Live sequential map-matcher — snaps the fused position onto RoadGraph with the same non-holonomic/continuity reasoning as `src/map_matching.py`, simplified for online (not whole-path) matching |
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

- **The bundled road graph only covers the replay route's area** (a ~1.8km
  radius around it, fetched offline via `src/map_matching.py`'s own
  `download_road_graph`). Live driving anywhere else won't find a nearby
  road to match onto (`MapMatcher.match` returns null — the raw fused
  trajectory still renders, just no green overlay). For a real demo venue,
  re-fetch a road_graph.json for that specific area the same way (see the
  inline snippet in RoadGraph.kt's usage, or ask to regenerate it).
- **MapMatcher is a simplified greedy sequential matcher**, not a full
  HMM/Viterbi port of `map_matching.py`'s DistanceMatcher — see
  MapMatcher.kt's docstring for why (online matching can't score whole
  future paths the way the offline evaluator can) and what's actually
  implemented (distance + continuity/non-holonomic scoring, one point at a
  time).
- No persistence — closing the app loses the current trajectory.
- Real (non-replay) live driving hasn't been tested — verified live only
  via the replay data source so far (real GPS/IMU wiring is the same code
  path, just unexercised end-to-end outdoors).
