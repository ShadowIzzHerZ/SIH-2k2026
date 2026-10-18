#!/bin/bash
# after queue10/11: best model with last-epoch + extra training drives, 3 seeds, then their average
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[01].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7 3; do
  tag=hist10_speedaware_lx; [ $s != 42 ] && tag=hist10_speedaware_lx_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_lx --seed $s > $L/$tag.log 2>&1
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx ens=hist10_speedaware_lx_s7 ens=hist10_speedaware_lx_s3 > $L/histspeedens_lx.log 2>&1
echo Q12_DONE >> $L/queue12.done
