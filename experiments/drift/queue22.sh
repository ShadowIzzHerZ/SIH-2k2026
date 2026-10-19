#!/bin/bash
# after queue20: 15 s history + speed input + the drift-experiments recipe (keep_last, extra drives, 6-chunk chains)
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|20).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7 3; do
  tag=hist10_speedaware_lx_chain6; [ $s != 42 ] && tag=hist10_speedaware_lx_chain6_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_lx_chain6 --seed $s > $L/$tag.log 2>&1
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx_chain6 ens=hist10_speedaware_lx_chain6_s7 ens=hist10_speedaware_lx_chain6_s3 > $L/lxchain6ens3.log 2>&1
echo Q22_DONE >> $L/queue22.done
