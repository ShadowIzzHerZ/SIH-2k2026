#!/bin/bash
# after queue7: how much history the speed-aware context model needs (5 s and 20 s of history)
cd "$(dirname "$0")/../.."
while pgrep -f "queue[2-7].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for e in hist5_speedaware hist15_speedaware; do
  for s in 42 7; do
    tag=$e; [ $s != 42 ] && tag=${e}_s$s
    [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 $e --seed $s > $L/$tag.log 2>&1
  done
done
echo Q8_DONE >> $L/queue8.done
