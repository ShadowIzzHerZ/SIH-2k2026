#!/bin/bash
# after queue4: longer context + current speed as input
cd "$(dirname "$0")/../.."
while pgrep -f "queue2.sh|queue3.sh|queue4.sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware; [ $s != 42 ] && tag=hist10_speedaware_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware --seed $s > $L/$tag.log 2>&1
done
echo Q5_DONE >> $L/queue5.done
