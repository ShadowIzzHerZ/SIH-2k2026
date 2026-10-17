"""
scripts/train_until_target*.sh: which Python runs, which checkpoint each
cycle resumes from, and where results go. Runs the real scripts in a scratch
copy of the repo layout with a stand-in "python" that logs its arguments
and fakes a passing eval report, so the loop stops after one cycle.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_PY = """#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$*" in
  *src.evaluate*)
    out=results/eval_report.json
    prev=""
    for a in "$@"; do
      [ "$prev" = "--run_name" ] && out="results/eval_report_$a.json"
      prev=$a
    done
    echo '{"mean_drift_pct": 5.0}' > "$out" ;;
  -c*) exec "{real}" "$@" ;;
esac
exit 0
"""


def _run(tmp_path, script, env_extra, checkpoints=("best_v3chain.pt", "best.pt")):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / script, tmp_path / "scripts" / script)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "results").mkdir()
    for c in checkpoints:
        (tmp_path / "checkpoints" / c).write_bytes(b"")
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY.replace("{real}", sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    env = {**os.environ, "PYTHON": str(fake), "FAKE_LOG": str(log), **env_extra}
    r = subprocess.run(["bash", str(tmp_path / "scripts" / script)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "target reached after 1 cycle" in r.stdout, r.stdout
    calls = [line for line in log.read_text().splitlines() if "-m src." in line]
    train = next(c for c in calls if "src.train" in c)
    evaluate = next(c for c in calls if "src.evaluate" in c)
    return train, evaluate


@pytest.mark.parametrize("script", ["train_until_target.sh", "train_until_target_comma.sh"])
def test_defaults_resume_shipped_model_into_separate_name(tmp_path, script):
    train, evaluate = _run(tmp_path, script, {})
    assert "--resume checkpoints/best_v3chain.pt" in train, train
    assert "--run_name loop" in train, train
    # nothing was trained into best_loop*.pt (fake train), so score the start point
    assert "--checkpoint checkpoints/best_v3chain.pt" in evaluate, evaluate
    assert "--run_name loop" in evaluate, evaluate


def test_resume_from_and_run_name_overrides(tmp_path):
    train, evaluate = _run(tmp_path, "train_until_target.sh",
                           {"RESUME_FROM": "checkpoints/best.pt", "RUN_NAME": "exp1",
                            "TRAIN_ARGS": "--chain_train_chunks 3"})
    assert "--resume checkpoints/best.pt" in train and "--run_name exp1" in train, train
    assert "--chain_train_chunks 3" in train, train
    assert "--run_name exp1" in evaluate, evaluate


def test_continues_from_own_output(tmp_path):
    train, evaluate = _run(tmp_path, "train_until_target.sh", {},
                           checkpoints=("best_v3chain.pt", "best_loop.pt"))
    assert "--resume checkpoints/best_loop.pt" in train, train
    assert "--checkpoint checkpoints/best_loop.pt" in evaluate, evaluate


def test_empty_run_name_keeps_old_best_pt_loop(tmp_path):
    train, evaluate = _run(tmp_path, "train_until_target.sh", {"RUN_NAME": ""})
    assert "--resume checkpoints/best.pt" in train and "--run_name" not in train, train
    assert "--checkpoint checkpoints/best.pt" in evaluate, evaluate
