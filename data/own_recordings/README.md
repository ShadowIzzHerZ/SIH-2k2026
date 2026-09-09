# Own recordings — drop CSVs here

This is where your team's own sensor-logging recordings go (see the
campus data-collection protocol discussed in the project chat / can be
added to `docs/sih.md` §7.5). This is separate from `data/IO-VNBD/` — that's
the public benchmark dataset; this folder is your own real-world runs on
your actual demo phone/vehicle/route.

## Naming convention
Use a name that tells you what the run was without opening it:

```
own_recordings/
├── calib_YYYYMMDD_HHMM.csv          # stationary calibration clip
├── openroute_YYYYMMDD_HHMM.csv      # normal driving/riding, GPS on throughout
├── blackout_<location>_YYYYMMDD_HHMM.csv   # e.g. blackout_basementA_20260903_0715.csv
```

## What each file needs
Whatever sensor-logging app you used (Sensor Logger, PhyPhox, SensorLog),
export as CSV with, at minimum:
- timestamp
- accelerometer x/y/z
- gyroscope x/y/z
- GPS lat/lon (even if it drops out mid-file during a blackout run — that
  gap is expected and useful, don't try to fill it in)

Exact column names don't matter — `src/data/io_vnbd_loader.py` fuzzy-matches
common naming variants (`accX`, `acc_x`, `AccelerometerX`, etc.). If a file
fails to load, run:

```bash
python src/data/inspect_dataset.py
```
(update its `DATA_ROOT` or point a quick script at this folder) to see what
columns it actually found, and add an explicit mapping in
`configs/default.yaml` → `data.column_map` if the fuzzy match misses.

## Before a blackout run
Always record a ~15s stationary `calib_*` clip immediately before it, same
phone mount, engine/wheels off — `calibration.py` uses this to work out the
phone's mounting angle relative to the vehicle.

## Status
- [ ] No recordings yet — this is set up and ready for the first drop
