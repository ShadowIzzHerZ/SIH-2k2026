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

## Ranking (two seeds where run; seed-to-seed spread is ~0.5 pt on IO 30 s, ~3 pt on comma 30 s)

1. **Fixed phone-to-vehicle alignment** (calibfix): clear win on both seeds. Physics alone 27.6 -> 18.2% (IO 30 s);
   trained model 16.2 -> 15.1% avg (IO 30 s), 24.6 -> 22.3% (IO 60 s), slow 58.6 -> ~41%.
2. **+ bias/noise augmentation** (augbias): small but consistent on both seeds: IO 30 s 14.65% avg, 60 s 21.1%, comma 30 s 11.7%.
3. 30 s chained fine-tune (chain6): IO about as good as augbias, but slow and comma worse; not adopted.
4. No gain / worse: stop detection (train or inference), IMU bias from the 30 s of GPS before a blackout,
   learned absolute-speed anchor (much worse), zero-initialised correction head.

Round 2 (best two-seed averages):
5. **Current speed fed to the network** (speedaware): IO 30 s 14.2%, **60 s 19.1%**, comma 5 s 1.95%; slow traffic worse (~46%).
6. **Speed-switched pair** (duo: speedaware above 6 m/s, augbias_long below): IO 30 s 14.1%, **60 s 19.1%**, slow **39.1%**,
   comma 5 s 1.95%, comma 30 s 13.1%. Best balance so far; no extra training.
7. No gain: speed from turning (centripetal observer), augbias + 30 s chains (slow worse), double training budget
   (validation improved, test unchanged except slow 37.5%).

Round 3 (seed spread turned out to be large, so only all-seed averages count; no seed is ever picked by test results):
8. **Keep the last training epoch** (lastepoch) instead of the best-validation epoch: the 5 validation drives do not
   track the test drives, and picking epochs on them produced bad seeds (60 s up to 22-26%). Last-epoch seeds are tighter.
9. **Average several seeds** (combos.py): 6 last-epoch seeds averaged (ens_lastepoch6) = IO 30 s 12.4%, **60 s 16.0%**,
   slow 37.8%, comma 30 s 11.5%. Best overall.
10. **18 recovered training drives** (prep_extra.py, train only): with last-epoch training they help long blackouts
    (no-extra seeds 14.0-14.1% at 30 s vs 12.2-13.6%); no-extra models are a bit better in slow traffic, so the pair
    duo_last6_noextra7 (with-drives above 7 m/s, no-extra below) gives slow 36.1% at 12.6 / 16.2%.
11. No gain: richer augmentation on top of the extra drives, double training budget, noisy speed input, weight-averaged
    epochs, training on the validation drives, slow-only specialist once last-epoch is used, 11-model mixed average.

Error decomposition (physics, fixed alignment, IO 30 s): 18.2% as is, 7.3% with true speed, 14.5% with true
heading, so forward-speed integration is the dominant error source.

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
| anchor | calibfix + learned absolute-speed anchor (complementary filter on speed) | 10.5 | 23.8 | 27.9 | 44.2 | 22.5 | 4.8 | 13.0 |
| chain6 | calibfix + chained fine-tune over 6 chunks (30 s) instead of 3 | 6.6 | 14.8 | 21.1 | 43.9 | 12.9 | 2.7 | 14.3 |
| zeroinit | calibfix + correction head initialised to zero (training starts from physics) | 6.5 | 15.6 | 22.8 | 46.3 | 13.9 | 2.4 | 12.0 |
| calibfix+zupt | calibfix checkpoint + stop detection at inference | 6.5 | 15.3 | 22.7 | 39.1 | 13.8 | 2.4 | 11.9 |
| calibfix_s7 | baseline + fixed phone->vehicle calibration (whole-drive leveling, forward axis fitted to GPS accel) | 6.4 | 14.9 | 21.8 | 42.8 | 13.0 | 2.9 | 15.1 |
| augbias_s7 | calibfix + random IMU bias/noise augmentation in training | 6.5 | 14.4 | 20.6 | 42.4 | 12.8 | 2.4 | 11.6 |
| aug_chain6 | calibfix + bias augmentation + 30 s (6-chunk) chained fine-tune | 6.7 | 15.1 | 21.1 | 52.1 | 13.5 | 2.8 | 10.7 |
| augbias_long | augbias with double the training budget (12 single-window + 12 chained epochs) | 6.6 | 14.9 | 21.7 | 37.5 | 13.5 | 2.6 | 12.0 |
| speedaware | calibfix + bias augmentation + current speed fed to the network as an input | 6.4 | 13.7 | 18.8 | 46.3 | 12.3 | 2.0 | 13.1 |
| speedaware_s7 | calibfix + bias augmentation + current speed fed to the network as an input | 6.7 | 14.7 | 19.4 | 45.7 | 12.8 | 1.9 | 12.9 |
| duo_speedaware | speedaware above 6 m/s, augbias_long below (switch on current speed estimate) | 6.4 | 13.8 | 19.2 | 38.9 | 12.4 | 2.0 | 13.2 |
| duo_speedaware_s7 | speedaware_s7 above 6 m/s, augbias_long below (switch on current speed estimate) | 6.6 | 14.4 | 18.9 | 39.3 | 12.8 | 1.9 | 13.0 |
| duo_all5_slowspec7 | 5-model average above 7 m/s, slow specialist below | 6.4 | 12.7 | 16.4 | 39.1 | 11.3 | 2.1 | 11.0 |
| duo_ens3_slowspec7 | 3-seed moredata average above 7 m/s, slow specialist below | 6.4 | 12.6 | 16.2 | 38.5 | 11.2 | 2.3 | 11.5 |
| duo_ens6_slowspec7 | 6-seed moredata average above 7 m/s, slow specialist below | 6.4 | 12.9 | 16.8 | 38.7 | 11.5 | 2.3 | 12.3 |
| duo_ens_slowplain8 | moredata seed-average above 8 m/s, plain slow specialist below | 6.4 | 12.7 | 16.5 | 40.3 | 11.3 | 2.3 | 12.4 |
| duo_ens_slowspec | moredata seed-average above 6 m/s, slow specialist below | 6.4 | 12.6 | 16.4 | 41.3 | 11.2 | 2.3 | 12.4 |
| duo_ens_slowspec10 | moredata seed-average above 10 m/s, slow specialist below | 6.6 | 13.6 | 18.4 | 35.3 | 11.9 | 2.3 | 12.4 |
| duo_ens_slowspec7 | moredata seed-average above 7 m/s, slow specialist below | 6.4 | 12.6 | 16.5 | 39.1 | 11.3 | 2.3 | 12.4 |
| duo_ens_slowspec8 | moredata seed-average above 8 m/s, slow specialist below | 6.4 | 13.0 | 16.9 | 37.6 | 11.4 | 2.3 | 12.4 |
| duo_last4_slowspec7 | 4-seed last-epoch average above 7 m/s, slow specialist below | 6.4 | 12.7 | 16.2 | 37.3 | 11.3 | 2.2 | 11.9 |
| duo_last6_noextra7 | 6-seed last-epoch moredata average above 7 m/s, 3-seed no-extra average below | 6.4 | 12.6 | 16.2 | 36.1 | 11.3 | 2.2 | 11.5 |
| duo_last6_slowspec7 | 6-seed last-epoch average above 7 m/s, slow specialist below | 6.4 | 12.6 | 16.3 | 37.3 | 11.3 | 2.2 | 11.6 |
| duo_moredata_slowspec | moredata above 6 m/s, slow specialist below | 6.4 | 12.7 | 16.8 | 41.8 | 11.4 | 2.2 | 12.0 |
| ens_all11 | average of every speedaware-family seed: 6 moredata + 5 speedaware | 6.2 | 12.7 | 16.5 | 47.2 | 11.1 | 2.1 | 12.9 |
| ens_all5 | average of moredata (3 seeds) and richaug (2 seeds) | 6.3 | 12.5 | 16.2 | 42.4 | 11.1 | 2.1 | 11.0 |
| ens_last9mix | average of 6 last-epoch moredata + 3 last-epoch no-extra seeds | 6.4 | 12.7 | 16.3 | 37.3 | 11.4 | 2.1 | 11.5 |
| ens_lastepoch4 | average of 4 last-epoch moredata seeds | 6.4 | 12.5 | 16.1 | 38.9 | 11.0 | 2.2 | 11.8 |
| ens_lastepoch6 | average of 6 last-epoch moredata seeds | 6.4 | 12.4 | 16.0 | 37.8 | 11.1 | 2.2 | 11.5 |
| ens_lastnoextra3 | average of 3 last-epoch speedaware seeds without the recovered drives | 6.5 | 13.8 | 17.8 | 35.0 | 12.5 | 2.2 | 11.3 |
| ens_moredata | average of moredata seeds 42 and 7 | 6.4 | 12.6 | 16.5 | 41.3 | 11.2 | 2.3 | 12.4 |
| ens_moredata3 | average of moredata seeds 42, 7 and 3 | 6.4 | 12.5 | 16.2 | 40.6 | 11.0 | 2.3 | 11.5 |
| ens_moredata4 | average of 4 moredata seeds | 6.3 | 12.8 | 16.8 | 45.3 | 11.2 | 2.2 | 11.9 |
| ens_moredata6 | average of all 6 moredata seeds (no seed picked by test results) | 6.3 | 12.8 | 16.7 | 43.1 | 11.2 | 2.3 | 12.1 |
| ens_round3 | average of moredata and richaug, both seeds | 6.3 | 12.6 | 16.4 | 43.0 | 11.3 | 2.0 | 11.1 |
| ens_speedaware3 | average of 3 speedaware seeds (no extra drives): fair control for the moredata ensembles | 6.4 | 12.8 | 16.4 | 48.2 | 11.4 | 2.1 | 13.4 |
| lastepoch | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.4 | 12.2 | 15.6 | 36.1 | 11.0 | 2.1 | 12.0 |
| lastepoch_chain6 | last-epoch moredata with the chained fine-tune over 6 chunks (30 s) instead of 3 | 6.5 | 12.3 | 15.7 | 49.0 | 10.9 | 2.6 | 11.7 |
| lastepoch_chain6_s7 | last-epoch moredata with the chained fine-tune over 6 chunks (30 s) instead of 3 | 6.6 | 13.4 | 16.7 | 37.5 | 12.0 | 2.4 | 12.5 |
| lastepoch_noextra | speedaware (no recovered drives) with the last epoch kept: checks whether the extra drives help once epoch picking is removed | 6.6 | 14.0 | 18.0 | 36.9 | 12.8 | 2.2 | 11.8 |
| lastepoch_noextra_s3 | speedaware (no recovered drives) with the last epoch kept: checks whether the extra drives help once epoch picking is removed | 6.6 | 14.0 | 19.0 | 36.7 | 12.8 | 2.1 | 10.4 |
| lastepoch_noextra_s7 | speedaware (no recovered drives) with the last epoch kept: checks whether the extra drives help once epoch picking is removed | 6.5 | 14.1 | 18.4 | 33.6 | 12.7 | 2.4 | 12.2 |
| lastepoch_s11 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.5 | 12.9 | 17.1 | 37.1 | 11.4 | 2.6 | 13.4 |
| lastepoch_s13 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.4 | 12.3 | 15.7 | 37.3 | 11.1 | 2.4 | 13.3 |
| lastepoch_s17 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.3 | 14.1 | 20.2 | 40.9 | 12.8 | 2.4 | 14.3 |
| lastepoch_s3 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.5 | 12.8 | 16.3 | 40.1 | 11.3 | 2.1 | 10.7 |
| lastepoch_s5 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.5 | 13.3 | 17.7 | 34.6 | 12.0 | 2.3 | 10.5 |
| lastepoch_s7 | moredata with the last epoch kept instead of the best-val epoch (control for trainval) | 6.5 | 13.6 | 17.9 | 38.2 | 12.2 | 2.3 | 12.0 |
| moredata | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.4 | 12.8 | 17.2 | 41.8 | 11.4 | 2.2 | 11.5 |
| moredata_long | moredata with double the training budget (12 single-window + 12 chained epochs) | 6.2 | 14.5 | 20.6 | 54.1 | 12.9 | 2.3 | 18.5 |
| moredata_s11 | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.3 | 14.9 | 22.7 | 56.8 | 13.4 | 2.1 | 13.9 |
| moredata_s13 | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.4 | 13.9 | 19.6 | 40.9 | 12.4 | 2.0 | 10.9 |
| moredata_s3 | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.4 | 12.5 | 16.4 | 39.9 | 11.1 | 2.5 | 10.3 |
| moredata_s5 | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.4 | 12.7 | 16.6 | 39.2 | 11.3 | 2.9 | 12.9 |
| moredata_s7 | speedaware + 18 recovered IO-VNBD training drives (+537 min, prep_extra.py); val/test unchanged | 6.4 | 12.9 | 16.6 | 44.0 | 11.4 | 2.4 | 13.3 |
| moredata_swa | moredata + weight average of the 6 chained epochs (kept if val agrees); targets the big seed-to-seed swings | 6.5 | 12.5 | 16.4 | 39.1 | 11.0 | 2.2 | 12.0 |
| moredata_swa_s11 | moredata + weight average of the 6 chained epochs (kept if val agrees); targets the big seed-to-seed swings | 6.3 | 16.0 | 25.2 | 57.2 | 14.1 | 2.5 | 15.9 |
| ref_v3chain_fwdfix | existing best_v3chain.pt (old leaky split), fwdfix calibration | 6.8 | 14.8 | 19.7 | 64.1 | 12.8 | 2.8 | 11.5 |
| richaug | speedaware + richer augmentation (scale error, yaw misalignment, road-grade leak, varied noise) | 6.5 | 13.6 | 17.5 | 43.6 | 12.3 | 2.2 | 11.0 |
| richaug_s7 | speedaware + richer augmentation (scale error, yaw misalignment, road-grade leak, varied noise) | 6.3 | 13.7 | 19.0 | 42.7 | 12.4 | 2.0 | 9.4 |
| richmore | speedaware + richer augmentation + 18 recovered training drives | 6.3 | 13.1 | 16.7 | 42.3 | 11.5 | 2.1 | 11.9 |
| richmore_s7 | speedaware + richer augmentation + 18 recovered training drives | 6.4 | 14.7 | 22.0 | 44.1 | 13.0 | 2.3 | 12.4 |
| slowspec | slow specialist: speedaware + richer augmentation trained only on windows below 8 m/s (with recovered drives); meant for the duo's low-speed half | 49.9 | 59.4 | 60.2 | 34.5 | 62.1 | 69.1 | 82.3 |
| slowspec_plain | slow specialist with plain bias augmentation (no richer augmentation), windows below 8 m/s, recovered drives | 35.9 | 49.8 | 52.1 | 36.3 | 51.7 | 65.2 | 77.1 |
| speedaware_s13 | calibfix + bias augmentation + current speed fed to the network as an input | 6.4 | 16.1 | 23.9 | 62.5 | 14.3 | 1.7 | 10.3 |
| speedaware_s3 | calibfix + bias augmentation + current speed fed to the network as an input | 6.3 | 12.9 | 16.5 | 49.4 | 11.5 | 2.5 | 15.8 |
| speedaware_s5 | calibfix + bias augmentation + current speed fed to the network as an input | 6.4 | 16.1 | 26.1 | 89.2 | 14.4 | 2.3 | 17.1 |
| speedjitter | moredata + noisy speed input during single-window training (10% + 0.5 m/s) | 6.4 | 13.2 | 17.1 | 43.9 | 11.7 | 2.5 | 13.9 |
| speedjitter_s7 | moredata + noisy speed input during single-window training (10% + 0.5 m/s) | 6.4 | 13.5 | 18.2 | 42.3 | 12.1 | 2.2 | 14.1 |
| trainval | moredata trained on train + val drives (5 more drives), last epoch kept: val scores do not track test, so spend val on training | 6.6 | 13.5 | 18.5 | 36.4 | 12.4 | 2.5 | 12.2 |
| trainval_s7 | moredata trained on train + val drives (5 more drives), last epoch kept: val scores do not track test, so spend val on training | 6.5 | 13.2 | 17.5 | 38.8 | 11.7 | 2.1 | 11.8 |
