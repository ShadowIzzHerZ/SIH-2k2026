#!/bin/bash
# after queue6: 3rd seed of 15 s context + speed input, then its seed ensemble
cd "$(dirname "$0")/../.."
while pgrep -f "queue[2-6].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
[ -f $R/hist10_speedaware_s3.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware --seed 3 > $L/hist10_speedaware_s3.log 2>&1
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware ens=hist10_speedaware_s7 ens=hist10_speedaware_s3 > $L/histspeedens.log 2>&1
echo Q7_DONE >> $L/queue7.done
