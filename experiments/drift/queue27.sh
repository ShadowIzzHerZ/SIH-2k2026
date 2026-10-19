#!/bin/bash
# after queue26: new best recipe with 20 s of context
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-6]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist15_speedaware_lx_chain6; [ $s != 42 ] && tag=hist15_speedaware_lx_chain6_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist15_speedaware_lx_chain6 --seed $s > $L/$tag.log 2>&1
done
echo Q27_DONE >> $L/queue27.done
