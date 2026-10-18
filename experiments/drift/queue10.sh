#!/bin/bash
# after queue8: best model + auxiliary speed head (2 seeds)
cd "$(dirname "$0")/../.."
while pgrep -f "queue[2-9].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaux; [ $s != 42 ] && tag=hist10_speedaux_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaux --seed $s > $L/$tag.log 2>&1
done
echo Q10_DONE >> $L/queue10.done
