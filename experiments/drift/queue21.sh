#!/bin/bash
# after queue20: slow windows 3x as often in single-window training
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0-7].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware_slow3; [ $s != 42 ] && tag=hist10_speedaware_slow3_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_slow3 --seed $s > $L/$tag.log 2>&1
done
echo Q21_DONE >> $L/queue21.done
