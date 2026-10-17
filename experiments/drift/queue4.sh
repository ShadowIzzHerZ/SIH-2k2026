#!/bin/bash
# after queue3: combine the two winners (longer context + 30 s weighting)
cd "$(dirname "$0")/../.."
while pgrep -f "queue2.sh|queue3.sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_endweight; [ $s != 42 ] && tag=hist10_endweight_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_endweight --seed $s > $L/$tag.log 2>&1
done
echo Q4_DONE >> $L/queue4.done
