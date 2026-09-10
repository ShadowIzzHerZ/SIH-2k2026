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

The UI shows a **real OpenStreetMap-tile map** (`RoadMapView.kt`, via
osmdroid — no API key, no Google Play Services) instead of an abstract
scatter plot: a colored "you are here" marker, mode-colored trail
polylines, the green map-matched overlay, a live-follow camera that pauses
the instant you pan (recenter FAB brings it back), all in floating cards
over the map — the same free-OSM-tiles UX approach as the Zen/DevStorm-2026
project's Leaflet map, ported to native Android. Verified live with both
the replay data and **real live GPS** (walking indoors/outdoors — real
street names rendered correctly, e.g. "Law Gate Rd", "NH44").

## Architecture

| File | Role |
|---|---|
| [app/src/main/java/.../Calibration.kt](app/src/main/java/com/sih26168/deadreckoning/Calibration.kt) | Direct Kotlin port of `src/calibration.py` — leveling + yaw rotation math |
| [app/src/main/java/.../CalibrationManager.kt](app/src/main/java/com/sih26168/deadreckoning/CalibrationManager.kt) | Drives calibration live on startup (stationary leveling, then yaw lock-in during an initial confident/moving stretch) |
| [app/src/main/java/.../SensorReader.kt](app/src/main/java/com/sih26168/deadreckoning/SensorReader.kt) | Wraps SensorManager for raw accel/gyro |
| [app/src/main/java/.../LocationReader.kt](app/src/main/java/com/sih26168/deadreckoning/LocationReader.kt) | Wraps LocationManager — no Play Services/API key needed. Uses `Location.getSpeed()`/`getBearing()` (the platform's own Doppler-derived values), not position-differencing — see `fusion.py`'s docstring for why that distinction mattered a lot in testing |
| [app/src/main/java/.../BiasCorrectionModel.kt](app/src/main/java/com/sih26168/deadreckoning/BiasCorrectionModel.kt) | ONNX Runtime wrapper for the exported network |
| [app/src/main/java/.../FusionEngine.kt](app/src/main/java/com/sih26168/deadreckoning/FusionEngine.kt) | The live GNSS↔INS state machine — same design as the fixed `src/fusion.py` |
| [app/src/main/java/.../RoadMapView.kt](app/src/main/java/com/sih26168/deadreckoning/RoadMapView.kt) | Real OSM-tile map (osmdroid) — "you are here" marker, mode-colored trail polylines, green map-matched overlay, follow/recenter camera |
| [app/src/main/java/.../ReplayDataSource.kt](app/src/main/java/com/sih26168/deadreckoning/ReplayDataSource.kt) | Reads a bundled real-drive replay asset — MainActivity loads two, `assets/replay_demo1.json`/`replay_demo2.json` (picked via the demoSelector radio buttons, shown once "Replay" is on), both converted from real DevRecorder captures in Jalandhar (`data/own_recordings/openroute_20260909_{1908,1821}.csv`) |
| [app/src/main/java/.../RoadGraph.kt](app/src/main/java/com/sih26168/deadreckoning/RoadGraph.kt) | Loads the bundled pre-fetched OSM road extract (`assets/road_graph.json`) |
| [app/src/main/java/.../MapMatcher.kt](app/src/main/java/com/sih26168/deadreckoning/MapMatcher.kt) | Live sequential map-matcher — snaps the fused position onto RoadGraph with the same non-holonomic/continuity reasoning as `src/map_matching.py`, simplified for online (not whole-path) matching |
| [app/src/main/java/.../DevRecorder.kt](app/src/main/java/com/sih26168/deadreckoning/DevRecorder.kt) | Developer mode's data logger — see below |
| [app/src/main/java/.../MainActivity.kt](app/src/main/java/com/sih26168/deadreckoning/MainActivity.kt) | Wires it all together, 10Hz tick loop |

### Developer mode — collecting real training data on a walk

Long-press the title to reveal a hidden "Developer mode" section
(persists across restarts; long-press again to hide — not something a
judge stumbles into). Its one control, **"Record training data"**, logs
real raw accel/gyro + GPS to a CSV at 10Hz — literal canonical column
names (`accel_x`, `gyro_z`, `lat`, `speed_gt`, ...) that
`src/data/io_vnbd_loader.py`'s fuzzy-matcher already recognizes with zero
config, verified end-to-end: a phone-recorded file loads cleanly through
the real, unmodified loader. A missing GPS fix is logged as blank fields,
not skipped — a real blackout stretch inside a recording is expected and
useful, per `data/own_recordings/README.md`'s own protocol. Blocked
while "Replay" is on (recording replay data would just duplicate
existing training data). Files land in the app's external-files
`recordings/` folder — grab them with `adb pull`, or tap **"Share last
recording"** for the normal Android share sheet (WhatsApp/Drive/email —
verified live) if there's no computer handy.

This is explicitly **data collection, not on-device training** — ONNX
Runtime Mobile (this app's inference engine) has no backprop, and
building a real training loop in Kotlin isn't something to rush before a
hackathon deadline, nor would it run the same already-validated training
code the project's real numbers come from. The actual workflow: record
on a walk/drive, pull the CSV into `data/own_recordings/`, then really
retrain with the existing, validated pipeline:
```bash
python -m src.train --config configs/default.yaml --resume checkpoints/best.pt
```

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
4. Once both complete, the map starts drawing: a blue marker/trail while
   GNSS-tracked, red during a blackout (real or the simulate toggle),
   orange during the reconnect blend, green for the map-matched overlay.
   The camera follows automatically; pan away to look around and tap the
   bottom-right button to snap back.

## Known limitations

- **The bundled road graph covers two areas** — the comma2k19 replay
  route (California) and wherever real live testing happened (currently
  Jalandhar, Punjab — both fetched offline via `src/map_matching.py`'s own
  `download_road_graph` and merged into one `road_graph.json`, ~460KB).
  Live driving/testing anywhere else won't find a nearby road to match
  onto (`MapMatcher.match` returns null — the raw fused trajectory still
  renders, just no green overlay). For a real demo venue, fetch and merge
  in a road_graph.json for that specific area the same way (see git
  history for the merge script, or ask to regenerate it).
- **MapMatcher is a simplified greedy sequential matcher**, not a full
  HMM/Viterbi port of `map_matching.py`'s DistanceMatcher — see
  MapMatcher.kt's docstring for why (online matching can't score whole
  future paths the way the offline evaluator can) and what's actually
  implemented (distance + continuity/non-holonomic + heading-consistency
  scoring, one point at a time). The heading term was added after a real
  bug found on a real device: at a junction/roundabout, several short
  edges can sit within maxDistM with similar distance+continuity scores,
  and without checking whether a candidate edge's own direction is
  anywhere close to the vehicle's current heading, the matcher would snap
  onto a perpendicular or backward-looping edge just because it was a few
  metres closer — visibly "going backwards" relative to the vehicle's
  actual direction of travel.
- **The map needs network access** to fetch OSM tiles (dead-reckoning
  itself stays fully offline — only the visual background needs a
  connection). A real dev-time gotcha, fixed but worth knowing: stale
  osmdroid `SharedPreferences` surviving repeated `adb install -r` cycles
  during development somehow left the online tile downloader out of the
  provider chain entirely (confirmed via `Configuration.isDebugTileProviders`
  — only offline/cache providers ever appeared). A clean uninstall +
  reinstall fixed it outright; a genuine first-time user install never
  hits this since it never has stale prefs to begin with — but if tiles
  ever show as a gray checkerboard on a dev device, uninstall and
  reinstall clean before assuming it's a real bug.
- No persistence — closing the app loses the current trajectory.
- Verified live with both the replay data source and real live GPS
  (walking, not driving) — a real driving test hasn't been done yet.
