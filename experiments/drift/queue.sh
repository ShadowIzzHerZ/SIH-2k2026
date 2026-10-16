#!/bin/bash
# run experiments one after another (one at a time: the box has 4 cores)
cd "$(dirname "$0")/../.."
while pgrep -f "run_exp baseline" >/dev/null; do sleep 15; done
for e in "$@"; do
  python3 -m experiments.drift.run_exp $e > experiments/drift/logs/${e// /_}.log 2>&1
done
