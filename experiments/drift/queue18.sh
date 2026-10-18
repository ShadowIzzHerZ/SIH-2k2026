#!/bin/bash
# after queue17: longer chained fine-tune (10 epochs instead of 6)
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0-7].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware_ch10; [ $s != 42 ] && tag=hist10_speedaware_ch10_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_ch10 --seed $s > $L/$tag.log 2>&1
done
echo Q18_DONE >> $L/queue18.done
