#!/bin/bash
# after queue14: weight averaging (SWA) over the chained epochs of the best recipe
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0234].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware_swa; [ $s != 42 ] && tag=hist10_speedaware_swa_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_swa --seed $s > $L/$tag.log 2>&1
done
echo Q15_DONE >> $L/queue15.done
