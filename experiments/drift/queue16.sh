#!/bin/bash
# after queue15: random mounting-rotation augmentation in training
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[02345].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
for s in 42 7; do
  tag=hist10_speedaware_rotaug; [ $s != 42 ] && tag=hist10_speedaware_rotaug_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist10_speedaware_rotaug --seed $s > $L/$tag.log 2>&1
done
echo Q16_DONE >> $L/queue16.done
