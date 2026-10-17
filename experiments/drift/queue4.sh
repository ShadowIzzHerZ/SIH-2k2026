#!/bin/bash
# Runs after queue3.sh: one experiment at a time.
cd "$(dirname "$0")/../.."
while pgrep -f "queue3[.]sh" > /dev/null; do sleep 30; done
for job in "richmore" "richmore --seed 7"; do
  name=$(echo $job | sed 's/ --seed /_s/')
  [ -f experiments/drift/results/$name.json ] && continue
  python -m experiments.drift.run_exp $job > logs/$name.log 2>&1
done
