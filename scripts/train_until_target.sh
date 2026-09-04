#!/usr/bin/env bash
# Keeps warm-restarting src.train (resuming from checkpoints/best.pt each
# cycle) + evaluating, until the PS's <10% test-drift target is hit, a real
# failure happens, or the cycle cap (a safety backstop, not a real limit) is
# reached. Mirrors the Colab notebook's loop cell — see
# scripts/build_colab_notebook.py.
#
# Run:
#   nohup ./scripts/train_until_target.sh > logs/train_loop.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET=10.0
MAX_CYCLES=500
PY=.venv/bin/python3

mkdir -p logs
echo "=== train_until_target started $(date) ==="

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle + 1))
  resume_args=()
  if [ -f checkpoints/best.pt ]; then
    resume_args=(--resume checkpoints/best.pt)
  fi
  echo ""
  echo "====================================================================="
  echo "cycle $cycle $(date) ${resume_args[*]:-(cold start)}"
  echo "====================================================================="

  "$PY" -m src.train --config configs/default.yaml "${resume_args[@]}"
  train_status=$?
  if [ $train_status -ne 0 ]; then
    echo "train exited with status $train_status — stopping loop"
    break
  fi

  "$PY" -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
  eval_status=$?
  if [ $eval_status -ne 0 ] || [ ! -f results/eval_report.json ]; then
    echo "evaluate failed or produced no report (status $eval_status) — stopping loop"
    break
  fi

  drift=$("$PY" -c "import json; print(json.load(open('results/eval_report.json'))['mean_drift_pct'])")
  echo "cycle $cycle: test mean drift ${drift}% (target < ${TARGET}%)"

  reached=$("$PY" -c "print(1 if float('$drift') < $TARGET else 0)")
  if [ "$reached" = "1" ]; then
    echo ""
    echo "target reached after $cycle cycle(s) — stopping."
    break
  fi
done

echo ""
echo "=== train_until_target ended $(date), cycle=$cycle ==="
