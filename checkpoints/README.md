# checkpoints/ — index

See [../results/README.md](../results/README.md) for each checkpoint's
actual eval numbers, and the main [README.md](../README.md)'s Status
section for the full narrative of why each one exists.

| File | Role |
|---|---|
| **`best_v3chain.pt`** | **Best checkpoint** (2026-10-16 re-check). Same 6-channel network and ONNX interface as `best.pt`, trained with the corrected IO-VNBD labels and repaired gyro yaw axis, then fine-tuned on chained 5 s chunks. Median drift 3.4% (5 s) / 13.1% (30 s) on comma2k19 highway, 22.0% / 38.8% on an unseen IO-VNBD drive. Not yet exported into the app. Export with `python -m src.export_onnx --checkpoint checkpoints/best_v3chain.pt`. |
| `best.pt` | What the Android app ships today (its ONNX export is `dead_reckoning_model.onnx`). 6-raw-channel network, trained on IO-VNBD + comma2k19 before the label fixes. 8.9% / 20.8% on highway, 52.5% / 139.4% on the unseen IO-VNBD drive. |
| `best_cleanlabels.pt`, `best_chainfix.pt`, `best_v3.pt` | Reference only — intermediate runs from the label-fix work. `best_cleanlabels.pt` is good on single 5 s windows but falls apart on chained blackouts (86-91% over 30 s). |
| `dead_reckoning_model.onnx` | ONNX export of `best.pt` — the exact file `app/`'s `syncModel` Gradle task copies into the app's assets at build time. Re-run `src/export_onnx.py` and rebuild the app if `best.pt` changes; never hand-edit this file. |
| `best_6ch_baseline.pt` | Reference only — the first trained checkpoint (IO-VNBD-only, before comma2k19 was mixed into training). Kept so the "mixing comma2k19 in didn't help IO-VNBD itself" comparison in the main README is reproducible, not just asserted. |
| `best_engineered.pt` | Reference only — a variant trained with 6 extra engineered input channels. Measurably worse on IO-VNBD, so **not** shipped; kept as evidence for why it was reverted. |
| `best_decimeter.pt` | Reference only — trained with the Google Smartphone Decimeter Challenge dataset mixed in alongside IO-VNBD (`--decimeter_dir`). The best result in the project on its own test set (6.86% mean / 2.77% median drift) but measurably worse on IO-VNBD (74.13%, worse than every other variant here) — so **not** shipped; kept as evidence. |

`best_prev.pt` (gitignored, not listed above) is a local safety backup
`src.train --resume` writes automatically before it overwrites `best.pt`
— convenience for whoever's running training, not a deliverable.
