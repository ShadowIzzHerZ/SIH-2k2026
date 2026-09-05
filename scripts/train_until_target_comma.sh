#!/usr/bin/env bash
# Same warm-restart loop as train_until_target.sh, but with comma2k19 mixed
# into every train/evaluate call (matches the currently-running combined
# IO-VNBD+comma2k19 experiment). Kept as a separate script rather than a
# flag on the original so a plain `train_until_target.sh` run never
# accidentally pulls in comma2k19 by surprise.
#
# Run:
#   nohup ./scripts/train_until_target_comma.sh > logs/train_loop_comma.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET=10.0
MAX_CYCLES=500
PY=.venv/bin/python3
COMMA_DIR="data/comma2k19_demo/data"

mkdir -p logs
echo "=== train_until_target_comma started $(date) ==="

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle + 1))
  resume_arg=""
  label="(cold start)"
  if [ -f checkpoints/best.pt ]; then
    resume_arg="--resume checkpoints/best.pt"
    label="$resume_arg"
  fi
  echo ""
  echo "====================================================================="
  echo "cycle $cycle $(date) $label"
  echo "====================================================================="

  "$PY" -u -m src.train --config configs/default.yaml --comma2k19_dir "$COMMA_DIR" $resume_arg
  train_status=$?
  if [ $train_status -ne 0 ]; then
    echo "train exited with status $train_status — stopping loop"
    break
  fi

  "$PY" -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt --comma2k19_dir "$COMMA_DIR"
  eval_status=$?
  if [ $eval_status -ne 0 ] || [ ! -f results/eval_report.json ]; then
    echo "evaluate failed or produced no report (status $eval_status) — stopping loop"
    break
  fi

  drift=$("$PY" -c "import json; print(json.load(open('results/eval_report.json'))['mean_drift_pct'])")
  echo "cycle $cycle: combined test mean drift ${drift}% (target < ${TARGET}%)"

  reached=$("$PY" -c "print(1 if float('$drift') < $TARGET else 0)")
  if [ "$reached" = "1" ]; then
    echo ""
    echo "target reached after $cycle cycle(s) — stopping."
    break
  fi
done

echo ""
echo "=== train_until_target_comma ended $(date), cycle=$cycle ==="
