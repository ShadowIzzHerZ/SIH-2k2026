#!/bin/bash
# Round 3 queue: runs one experiment at a time after the current one finishes.
cd "$(dirname "$0")/../.."
while ps aux | grep -v grep | grep -q "run_exp richaug"; do sleep 30; done
for job in "moredata" "moredata --seed 7" "richaug --seed 7"; do
  name=$(echo $job | sed 's/ --seed /_s/')
  [ -f experiments/drift/results/$name.json ] && continue
  python -m experiments.drift.run_exp $job > logs/$name.log 2>&1
done
