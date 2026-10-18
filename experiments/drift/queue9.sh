#!/bin/bash
# after queue8: two more seeds of the best model and a 5-model average
cd "$(dirname "$0")/../.."
while pgrep -f "queue[2-8].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 11 13; do
  [ -f $R/hist10_speedaware_s$s.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware --seed $s > $L/hist10_speedaware_s$s.log 2>&1
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware ens=hist10_speedaware_s7 ens=hist10_speedaware_s3 ens=hist10_speedaware_s11 ens=hist10_speedaware_s13 > $L/histspeedens5.log 2>&1
echo Q9_DONE >> $L/queue9.done
