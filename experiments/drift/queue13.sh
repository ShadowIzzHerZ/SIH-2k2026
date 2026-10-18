#!/bin/bash
# after queue10 and queue12 (one training job at a time: two at once thrash the 4 cores)
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[02].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedbig; [ $s != 42 ] && tag=hist10_speedbig_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedbig --seed $s > $L/$tag.log 2>&1
done
echo Q13_DONE >> $L/queue13.done
