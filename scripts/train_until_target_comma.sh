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
# Python: $PYTHON if set, else the repo's .venv when there is one, else
# python3 on PATH (Colab, CI, a conda env).
if [ -n "${PYTHON:-}" ]; then
  PY=$PYTHON
elif [ -x .venv/bin/python3 ]; then
  PY=.venv/bin/python3
else
  PY=python3
fi

# The first cycle warm-starts from RESUME_FROM (default: best_v3chain.pt,
# the checkpoint the app ships). Results go to checkpoints/best_<RUN_NAME>.pt
# (default best_loop_comma.pt) and results/eval_report_<RUN_NAME>.json, and every
# later cycle resumes from that file. Writing to a separate name keeps
# src.train from overwriting best_v3chain.pt or the best.pt reference model.
# RUN_NAME= (empty) gives the old behaviour: train into and resume from
# checkpoints/best.pt. TRAIN_ARGS is passed through to src.train, e.g.
# TRAIN_ARGS="--chain_train_chunks 3" to keep the chained-chunk objective
# best_v3chain.pt was fine-tuned with.
RESUME_FROM=${RESUME_FROM:-checkpoints/best_v3chain.pt}
RUN_NAME=${RUN_NAME-loop_comma}
TRAIN_ARGS=${TRAIN_ARGS:-}
if [ -n "$RUN_NAME" ]; then
  OUT=checkpoints/best_${RUN_NAME}.pt
  name_arg="--run_name $RUN_NAME"
  REPORT=results/eval_report_${RUN_NAME}.json
else
  OUT=checkpoints/best.pt
  name_arg=""
  REPORT=results/eval_report.json
fi
COMMA_DIR="data/comma2k19_demo/data"

mkdir -p logs
echo "=== train_until_target_comma started $(date) ==="

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle + 1))
  resume_arg=""
  label="(cold start)"
  if [ -f "$OUT" ]; then
    resume_arg="--resume $OUT"
  elif [ -f "$RESUME_FROM" ]; then
    resume_arg="--resume $RESUME_FROM"
  fi
  if [ -n "$resume_arg" ]; then
    label="$resume_arg"
  fi
  echo ""
  echo "====================================================================="
  echo "cycle $cycle $(date) $label"
  echo "====================================================================="

  "$PY" -u -m src.train --config configs/default.yaml --comma2k19_dir "$COMMA_DIR" $resume_arg $name_arg $TRAIN_ARGS
  train_status=$?
  if [ $train_status -ne 0 ]; then
    echo "train exited with status $train_status — stopping loop"
    break
  fi

  # src.train only writes $OUT when it beats the checkpoint it resumed
  # from, so after a cycle with no improvement score the starting point.
  ckpt=$OUT
  [ -f "$ckpt" ] || ckpt=$RESUME_FROM
  "$PY" -m src.evaluate --config configs/default.yaml --checkpoint "$ckpt" --comma2k19_dir "$COMMA_DIR" $name_arg
  eval_status=$?
  if [ $eval_status -ne 0 ] || [ ! -f "$REPORT" ]; then
    echo "evaluate failed or produced no report (status $eval_status) — stopping loop"
    break
  fi

  drift=$("$PY" -c "import json; print(json.load(open('$REPORT'))['mean_drift_pct'])")
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
