#!/bin/bash
# after queue21/22: two more seeds of the new best recipe and the 5-model average
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-2]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 11 13; do
  tag=hist10_speedaware_lx_chain6_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_lx_chain6 --seed $s > $L/$tag.log 2>&1
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx_chain6 ens=hist10_speedaware_lx_chain6_s7 ens=hist10_speedaware_lx_chain6_s3 ens=hist10_speedaware_lx_chain6_s11 ens=hist10_speedaware_lx_chain6_s13 > $L/lxchain6ens5.log 2>&1
echo Q23_DONE >> $L/queue23.done
