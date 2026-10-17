#!/bin/bash
# Runs after queue4.sh: one experiment at a time.
cd "$(dirname "$0")/../.."
while pgrep -f "queue[34][.]sh" > /dev/null; do sleep 30; done
for job in "slowspec" "richmore_long --stage1 12 --stage2 12"; do
  name=$(echo $job | sed 's/ --seed /_s/; s/ --stage.*//')
  [ -f experiments/drift/results/$name.json ] && continue
  python -m experiments.drift.run_exp $job > logs/$name.log 2>&1
done
