#!/bin/bash
# after queue16: third weight-averaging seed, then the 3-model average of those
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0-6].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
[ -f $R/hist10_speedaware_swa_s3.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_swa --seed 3 > $L/hist10_speedaware_swa_s3.log 2>&1
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_swa ens=hist10_speedaware_swa_s7 ens=hist10_speedaware_swa_s3 > $L/swaens3.log 2>&1
echo Q17_DONE >> $L/queue17.done
