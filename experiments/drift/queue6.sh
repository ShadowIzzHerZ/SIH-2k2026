#!/bin/bash
# after queue5: ensemble of all 15 s-context models, then a 3rd seed of the combo
cd "$(dirname "$0")/../.."
while pgrep -f "queue[2-5].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
python3 -m experiments.drift.run_ensembles seedens hist=100 ens=hist10 ens=hist10_s7 ens=hist10_s3 ens=hist10_endweight ens=hist10_endweight_s7 > $L/histens5.log 2>&1
[ -f $R/hist10_endweight_s3.json ] || python3 -m experiments.drift.run_exp2 hist10_endweight --seed 3 > $L/hist10_endweight_s3.log 2>&1
echo Q6_DONE >> $L/queue6.done
