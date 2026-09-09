# results/ — index

Every file here is a real, generated report — nothing hand-edited. The
main [README.md](../README.md)'s Status section is the full narrative
(what was tried, why, what happened); this table is just so the
directory itself doesn't read as an undifferentiated pile of JSON when
you're browsing it directly.

## Current production checkpoint (`checkpoints/best.pt`)

The 6-raw-channel network, trained on IO-VNBD + comma2k19 combined (the
one the app ships). "Current" files have no run suffix; the `_run1`
duplicates alongside them are kept because the main README cites those
exact filenames as the evidence trail for that specific training run —
same numbers, both kept rather than one silently deleted out from under
a citation.

| File | What it is | Headline number |
|---|---|---|
| [eval_report.json](eval_report.json) | Combined test-set drift (IO-VNBD + comma2k19 windows together) | 60.92% mean / 66.96% median |
| [eval_report_breakdown.json](eval_report_breakdown.json) | Same checkpoint, split by dataset | IO-VNBD 62.33% mean · comma2k19 **16.49% mean / 8.94% median** |
| [eval_report_combined_run1.json](eval_report_combined_run1.json) | = `eval_report.json`, kept as the specific artifact the README's "mixed comma2k19 into training" bullet cites | (identical) |
| [eval_report_breakdown_run1.json](eval_report_breakdown_run1.json) | = `eval_report_breakdown.json`, same reason | (identical) |
| [train_history.json](train_history.json) | Per-epoch train/val loss and drift for this checkpoint | — |
| [comma2k19_eval.json](comma2k19_eval.json) | Zero-shot comma2k19 check *before* it was mixed into training (IO-VNBD-only checkpoint, evaluated with no retraining) | 29.27% mean / 15.64% median |

## Superseded / reference checkpoints (kept for the record, not shipped)

| File | What it is | Headline number | Why it's here |
|---|---|---|---|
| [eval_report_6ch_baseline.json](eval_report_6ch_baseline.json) | The first trained checkpoint, IO-VNBD-only, before comma2k19 was mixed in | 62.76% mean | `checkpoints/best_6ch_baseline.pt`'s eval — the "before" in the mixing-comma2k19 comparison |
| [train_history_6ch_baseline.json](train_history_6ch_baseline.json) | Its training curve | — | Shows the val-drift plateau that motivated everything tried afterward |
| [eval_report_engineered.json](eval_report_engineered.json) | A variant with 6 extra engineered input channels (accel/gyro magnitude, jerk, roughness) | 62.75% mean (worse than baseline) | `checkpoints/best_engineered.pt`'s eval — evidence for reverting the extra-features experiment |
| [eval_report_breakdown_engineered.json](eval_report_breakdown_engineered.json) | Same variant, split by dataset | IO-VNBD 64.28% (worse) · comma2k19 14.48%/5.42% (better) | Shows *why* it was reverted despite helping comma2k19 — IO-VNBD is the harder, binding case |
| [train_history_engineered.json](train_history_engineered.json) | Its training curve | — | — |

## GNSS↔INS fusion demo reports (`src/simulate_blackout.py`)

Not model eval — these run the *fusion state machine* (`src/fusion.py`)
over a real held-out drive with an injected blackout, the on-device
app's exact counterpart.

| File | Scenario | Result |
|---|---|---|
| [blackout_demo_comma30_report.json](blackout_demo_comma30_report.json) | comma2k19 highway, 30s continuous blackout | **2.9% drift** — under the 10% target, 0.0m reconnect jump |
| [blackout_demo_iovnbd45_report.json](blackout_demo_iovnbd45_report.json) | IO-VNBD urban, 45s continuous blackout | 234% drift — the known low-speed/urban weakness, chained over a longer continuous span |

## Map-matching

| File | What it is |
|---|---|
| [mapmatching_on_real_predictions.log](mapmatching_on_real_predictions.log) | Full run log for measuring map-matching's effect on the trained model's *own* (not synthetic) predictions — see main README for the honest result (it doesn't help on high-drift input; the synthetic-drift recovery test in `tests/test_map_matching.py` is the one showing the real 62% improvement, since that test doesn't write a file here) |

## Not tracked in git (regenerable, not deliverables)

`results/*.png` and `results/*.csv` are gitignored — plots and the full
per-window `drift_diagnostics.csv` are large/regenerable, not needed to
evaluate the project from a fresh clone. Re-run the relevant `src/*.py`
script to regenerate them.
