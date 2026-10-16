# Drift experiments: results

Median drift, % of distance travelled, on test drives that no model saw in
training (drive-level split, see split.json). Lower is better; target < 10%.
IO = IO-VNBD (6 test drives, 149 min),
comma = comma2k19 (9 test segments from 3 routes).
"5 s" = one window started from the true state; "30 s"/"60 s" = chained
blackouts, each 5 s chunk starting from the previous chunk's own estimate.
slow/fast = chain mean speed below/above 6 m/s.

ref_v3chain_orig was trained on the old leaky split, so some of these test
drives were in its training data: reference only, not a fair comparison.

| run | what changed | IO 5 s | IO 30 s | IO 60 s | IO 30 s slow | IO 30 s fast | comma 5 s | comma 30 s |
|---|---|---|---|---|---|---|---|---|
| ref_physics_orig | physics only, orig calibration | 10.4 | 27.6 | 41.0 | 273.0 | 23.5 | 7.2 | 30.6 |
| ref_physics_fwdfix | physics only, fwdfix calibration | 7.9 | 18.2 | 26.0 | 63.5 | 16.1 | 2.5 | 12.4 |
| phys_zupt | physics only, fixed calibration, phys_zupt | 7.9 | 17.9 | 25.8 | 53.7 | 16.1 | 2.5 | 12.4 |
| phys_bias_gyro | physics only, fixed calibration, phys_bias_gyro | 7.9 | 20.3 | 30.8 | 64.9 | 18.2 | 2.5 | 11.6 |
| phys_bias_gyro_120s | physics only, fixed calibration, phys_bias_gyro_120s | 7.9 | 18.7 | 27.1 | 63.9 | 16.8 | 2.5 | 12.4 |
| phys_bias_accel | physics only, fixed calibration, phys_bias_accel | 7.9 | 21.6 | 35.4 | 67.6 | 18.6 | 2.5 | 13.9 |
| phys_bias_both | physics only, fixed calibration, phys_bias_both | 7.9 | 23.7 | 39.6 | 69.5 | 20.9 | 2.5 | 13.5 |
| ref_v3chain_orig | existing best_v3chain.pt (old leaky split), orig calibration | 7.1 | 13.6 | 18.6 | 45.8 | 12.0 | 2.7 | 9.8 |
| baseline | v3 recipe (heading + end-heading losses, chained fine-tune K=3), original calibration, clean split | 6.6 | 16.2 | 24.6 | 58.6 | 14.2 | 2.9 | 15.4 |
| calibfix | baseline + fixed phone->vehicle calibration (whole-drive leveling, forward axis fitted to GPS accel) | 6.5 | 15.3 | 22.7 | 38.6 | 13.8 | 2.4 | 11.9 |
| augbias | calibfix + random IMU bias/noise augmentation in training | 6.5 | 14.9 | 21.6 | 38.8 | 13.3 | 2.5 | 11.8 |
| calibfix+zupt | calibfix checkpoint + stop detection at inference | 6.5 | 15.3 | 22.7 | 39.1 | 13.8 | 2.4 | 11.9 |
| ref_v3chain_fwdfix | existing best_v3chain.pt (old leaky split), fwdfix calibration | 6.8 | 14.8 | 19.7 | 64.1 | 12.8 | 2.8 | 11.5 |
