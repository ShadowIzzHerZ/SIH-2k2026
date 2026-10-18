#!/bin/bash
# after queue19: distil the 3-model average into one model (one third of the phone compute)
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0-9].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware_distill; [ $s != 42 ] && tag=hist10_speedaware_distill_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_distill --seed $s > $L/$tag.log 2>&1
done
echo Q20_DONE >> $L/queue20.done
