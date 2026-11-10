#!/usr/bin/env bash
# Keeps warm-restarting src.train (resuming from the best checkpoint so far
# each cycle, see RESUME_FROM / RUN_NAME below) + evaluating, until the PS's <10% test-drift target is hit, a real
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
# (default best_loop.pt) and results/eval_report_<RUN_NAME>.json, and every
# later cycle resumes from that file. Writing to a separate name keeps
# src.train from overwriting best_v3chain.pt or the best.pt reference model.
# RUN_NAME= (empty) gives the old behaviour: train into and resume from
# checkpoints/best.pt. TRAIN_ARGS is passed through to src.train, e.g.
# TRAIN_ARGS="--chain_train_chunks 3" to keep the chained-chunk objective
# best_v3chain.pt was fine-tuned with.
RESUME_FROM=${RESUME_FROM:-checkpoints/best_v3chain.pt}
RUN_NAME=${RUN_NAME-loop}
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

mkdir -p logs
echo "=== train_until_target started $(date) ==="

cycle=0
while [ "$cycle" -lt "$MAX_CYCLES" ]; do
  cycle=$((cycle + 1))
  # A plain string (not a bash array) deliberately — macOS's default
  # /bin/bash is 3.2, where `"${arr[@]}"` on an empty array errors under
  # `set -u`. Safe to word-split unquoted here: it's always either empty or
  # "--resume <checkpoint path>", so keep checkpoint paths free of spaces.
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

  "$PY" -u -m src.train --config configs/default.yaml $resume_arg $name_arg $TRAIN_ARGS
  train_status=$?
  if [ $train_status -ne 0 ]; then
    echo "train exited with status $train_status — stopping loop"
    break
  fi

  # src.train only writes $OUT when it beats the checkpoint it resumed
  # from, so after a cycle with no improvement score the starting point.
  ckpt=$OUT
  [ -f "$ckpt" ] || ckpt=$RESUME_FROM
  "$PY" -m src.evaluate --config configs/default.yaml --checkpoint "$ckpt" $name_arg
  eval_status=$?
  if [ $eval_status -ne 0 ] || [ ! -f "$REPORT" ]; then
    echo "evaluate failed or produced no report (status $eval_status) — stopping loop"
    break
  fi

  drift=$("$PY" -c "import json; print(json.load(open('$REPORT'))['mean_drift_pct'])")
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
