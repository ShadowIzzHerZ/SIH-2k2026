# checkpoints/ — index

See [../results/README.md](../results/README.md) for each checkpoint's
actual eval numbers, and the main [README.md](../README.md)'s Status
section for the full narrative of why each one exists.

| File | Role |
|---|---|
| **`best.pt`** | **Production checkpoint** — what `src/export_onnx.py` exports and the Android app ships. 6-raw-channel network, trained on IO-VNBD + comma2k19 combined. |
| `dead_reckoning_model.onnx` | ONNX export of `best.pt` — the exact file `app/`'s `syncModel` Gradle task copies into the app's assets at build time. Re-run `src/export_onnx.py` and rebuild the app if `best.pt` changes; never hand-edit this file. |
| `best_6ch_baseline.pt` | Reference only — the first trained checkpoint (IO-VNBD-only, before comma2k19 was mixed into training). Kept so the "mixing comma2k19 in didn't help IO-VNBD itself" comparison in the main README is reproducible, not just asserted. |
| `best_engineered.pt` | Reference only — a variant trained with 6 extra engineered input channels. Measurably worse on IO-VNBD, so **not** shipped; kept as evidence for why it was reverted. |

`best_prev.pt` (gitignored, not listed above) is a local safety backup
`src.train --resume` writes automatically before it overwrites `best.pt`
— convenience for whoever's running training, not a deliverable.
