#!/bin/bash
# after queue24: new best recipe with the chain loss weighted toward 30 s
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-4]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7 3; do
  tag=hist10_speedaware_lx_chain6_ew; [ $s != 42 ] && tag=hist10_speedaware_lx_chain6_ew_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_lx_chain6_ew --seed $s > $L/$tag.log 2>&1
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx_chain6_ew ens=hist10_speedaware_lx_chain6_ew_s7 ens=hist10_speedaware_lx_chain6_ew_s3 > $L/lxchain6ewens3.log 2>&1
echo Q25_DONE >> $L/queue25.done
