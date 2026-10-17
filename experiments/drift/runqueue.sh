#!/bin/bash
# Runs experiments/drift/queue.txt top to bottom, one at a time, skipping any
# run that already has a result. Re-reads the file after every run, so new
# lines can be appended while it runs. Safe to restart after a reboot.
cd "$(dirname "$0")/../.."
while true; do
  job=""
  while read -r line; do
    [ -z "$line" ] && continue
    name=$(echo $line | sed 's/ --seed /_s/; s/ --stage.*//')
    [ -f experiments/drift/results/$name.json ] && continue
    job="$line"; break
  done < experiments/drift/queue.txt
  [ -z "$job" ] && { sleep 60; continue; }
  python -m experiments.drift.run_exp $job > logs/$name.log 2>&1 || sleep 60
done
