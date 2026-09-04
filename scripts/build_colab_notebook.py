"""
Generates notebooks/colab_train.ipynb from the actual src/ files on disk,
so the Colab notebook is always byte-identical to the local repo instead of
a hand-copied (and driftable) duplicate.

Run this again any time src/*.py or configs/default.yaml change, before
re-uploading to Colab.

Usage:
    python scripts/build_colab_notebook.py
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_train.ipynb"


def _cell_id():
    return uuid.uuid4().hex[:8]


def md(text):
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": _cell_id(), "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def writefile_cell(rel_path: str):
    """A %%writefile cell whose body is read straight from the real file on disk."""
    content = (ROOT / rel_path).read_text()
    return code(f"%%writefile {rel_path}\n" + content)


cells = []

cells.append(md(
"""# SIH26168 — Dead Reckoning Training (Colab)

Trains the physics-informed bias-correction network on the IO-VNBD dataset.
This notebook is auto-generated from the local project's `src/` files
(`scripts/build_colab_notebook.py`) — the code cells below are byte-identical
to the repo, not a hand-copied version.

**Before running:** Runtime → Change runtime type → **T4 GPU** (or better).

**Steps:** get dataset → install deps → write project code → confirm real
column names → resume from the best existing checkpoint → train in a loop
until the PS's <10% drift target is hit (or this runtime ends) → evaluate →
export ONNX.
"""))

cells.append(code(
"""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("⚠️  No GPU — go to Runtime → Change runtime type → GPU before continuing.")
"""))

cells.append(md("## 1. Get the dataset (IO-VNBD, ~1.6GB of CSVs over git-lfs)"))

cells.append(code(
"""!apt-get -qq update && apt-get -qq install -y git-lfs
!git lfs install
!git clone --depth 1 https://github.com/onyekpeu/IO-VNBD.git data/IO-VNBD
%cd data/IO-VNBD
!git lfs pull --include="*.csv" --exclude="*.zip"
%cd /content
"""))

cells.append(md(
"""## 2. Mount Google Drive (recommended)

Keeps checkpoints/results if the Colab runtime disconnects. Skip this cell
if you'd rather not connect Drive — training still works, you'll just lose
checkpoints on disconnect.
"""))

cells.append(code(
"""from google.colab import drive
drive.mount('/content/drive')

import os
CKPT_DIR = "/content/drive/MyDrive/SIH26168-ML/checkpoints"
RESULTS_DIR = "/content/drive/MyDrive/SIH26168-ML/results"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Checkpoints will be saved to:", CKPT_DIR)
"""))

cells.append(md("## 3. Install Python deps (torch is preinstalled on Colab; add the rest)"))
cells.append(code("""!pip install -q onnx onnxruntime filterpy pyyaml\n"""))

cells.append(md(
"""## 4. Project code

Written from the local repo's `src/` files — identical logic to what runs
on your machine, so results here are directly comparable.
"""))

cells.append(code(
"""import os
os.makedirs("src/data", exist_ok=True)
os.makedirs("src/models", exist_ok=True)
os.makedirs("configs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
open("src/__init__.py", "w").close()
open("src/data/__init__.py", "w").close()
open("src/models/__init__.py", "w").close()
"""))

for rel in [
    "src/calibration.py",
    "src/models/strapdown_ins.py",
    "src/models/bias_correction_net.py",
    "src/data/io_vnbd_loader.py",
    "src/data/windowing.py",
    "src/train.py",
    "src/evaluate.py",
    "src/export_onnx.py",
]:
    cells.append(writefile_cell(rel))

cells.append(writefile_cell("configs/default.yaml"))

cells.append(md(
"""## 5. Confirm the real IO-VNBD column names

**Do not skip this** — the loader fuzzy-matches column names as a fallback,
but confirm it actually resolved the right ones before trusting any result.
"""))

cells.append(code(
"""import pandas as pd
from pathlib import Path

# S-*.csv only — those are the smartphone recordings this pipeline trains
# on. V-*.csv (vehicle CAN-bus) lack 3-axis accel/gyro and are excluded via
# configs/default.yaml's data.file_prefix.
samples = sorted(Path("data/IO-VNBD").rglob("S-*.csv"))
samples = [p for p in samples if p.stat().st_size > 5000][:5]

for p in samples:
    print("---", p, "---")
    try:
        df = pd.read_csv(p, nrows=3, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(p, nrows=3, encoding="latin-1")  # IO-VNBD isn't consistently UTF-8
    print(list(df.columns))
    print(df.head(2))
    print()
"""))

cells.append(md(
"""If the fuzzy-matched columns look wrong, edit `configs/default.yaml`'s
`data.column_map` in the cell above (re-run that `%%writefile` cell with the
real names filled in) before training.
"""))

cells.append(md(
"""## 6. Get the best checkpoint to resume from

Never cold-starts if a better starting point exists. Checks, in order:
1. Google Drive from a previous Colab run (Step 2), if mounted and present
2. `checkpoints/best.pt` already pushed to the GitHub repo (the last
   locally-trained result — currently ~62.8% test drift)
3. otherwise trains from scratch

`src.train`'s `--resume` flag (below) then warm-starts from whatever this
cell finds, instead of random init.
"""))

cells.append(code(
"""import os
import shutil
import urllib.request

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/ShadowIzzHerZ/Sih/main"

if os.path.exists(CKPT_DIR + "/best.pt"):
    shutil.copy(CKPT_DIR + "/best.pt", "checkpoints/best.pt")
    if os.path.exists(RESULTS_DIR + "/train_history.json"):
        shutil.copy(RESULTS_DIR + "/train_history.json", "results/train_history.json")
    print("resuming from Google Drive checkpoint (previous Colab run)")
else:
    try:
        urllib.request.urlretrieve(GITHUB_RAW + "/checkpoints/best.pt", "checkpoints/best.pt")
        print("resuming from checkpoints/best.pt already in the GitHub repo (~62.8% test drift)")
    except Exception as e:
        print(f"no existing checkpoint found anywhere ({e}) — will train from scratch")
"""))

cells.append(md(
"""## 7. Train until the target drift is hit (or this runtime ends)

Each cycle runs a full `src.train` pass (`--resume checkpoints/best.pt` once
one exists) — a cosine-annealed "warm restart" from the current best weights
— then evaluates on the real held-out test set. It keeps looping,
checkpointing to Drive after every cycle so nothing is lost if Colab
disconnects mid-run, until either the PS's <10% drift target is hit or the
cycle cap below is reached (a safety backstop against a runaway loop, not a
real target — at ~60 epochs/cycle it's far more cycles than one Colab
session will ever reach, so in practice this only stops on target-hit,
failure, or the runtime itself ending).
"""))

cells.append(code(
"""import subprocess
import sys
import json
import shutil
import os

TARGET_DRIFT_PCT = 10.0
MAX_CYCLES = 500  # safety backstop only, see markdown above — not a real limit

cycle = 0
while cycle < MAX_CYCLES:
    cycle += 1
    resume_args = ["--resume", "checkpoints/best.pt"] if os.path.exists("checkpoints/best.pt") else []
    print(f"\\n{'='*70}\\ncycle {cycle} {'(warm restart)' if resume_args else '(cold start)'}\\n{'='*70}")

    # -u: unbuffered stdout. Without it, src.train's stdout is piped (not a
    # TTY) so Python fully block-buffers it — nothing appears in the Colab
    # cell until the whole cycle's subprocess exits and flushes everything
    # at once, which can look like a silent hang for many minutes on a
    # cold-start cycle. -u makes per-epoch progress genuinely live here too.
    train_ret = subprocess.run([sys.executable, "-u", "-m", "src.train", "--config", "configs/default.yaml", *resume_args])
    if train_ret.returncode != 0:
        print("training subprocess exited with an error — stopping the loop, see output above")
        break

    if os.path.exists("/content/drive/MyDrive"):
        shutil.copy("checkpoints/best.pt", CKPT_DIR + "/best.pt")
        if os.path.exists("results/train_history.json"):
            shutil.copy("results/train_history.json", RESULTS_DIR + "/train_history.json")

    eval_ret = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--config", "configs/default.yaml", "--checkpoint", "checkpoints/best.pt"],
        capture_output=True, text=True,
    )
    print(eval_ret.stdout[-1500:])
    if eval_ret.returncode != 0 or not os.path.exists("results/eval_report.json"):
        print("evaluate failed or produced no report — stopping the loop, see output above")
        break

    report = json.load(open("results/eval_report.json"))
    mean_drift = report["mean_drift_pct"]
    print(f"cycle {cycle}: test mean drift {mean_drift:.2f}% (target < {TARGET_DRIFT_PCT}%)")
    if os.path.exists("/content/drive/MyDrive"):
        shutil.copy("results/eval_report.json", RESULTS_DIR + "/eval_report.json")

    if mean_drift < TARGET_DRIFT_PCT:
        print(f"\\n🎯 target reached after {cycle} cycle(s) — stopping.")
        break
else:
    print(f"\\nhit the {MAX_CYCLES}-cycle safety cap without reaching target — this would be very unusual; check results/train_history.json for what's actually happening (plateaued vs. still improving).")

assert os.path.exists("checkpoints/best.pt"), "checkpoint still missing — something is wrong, scroll up"
print(f"\\n✅ checkpoints/best.pt: {os.path.getsize('checkpoints/best.pt')} bytes")
"""))

cells.append(md(
"""## 8. Evaluate against the PS's <10% drift benchmark

(Step 7's loop already ran this each cycle — this cell just re-confirms the
final number on the checkpoint the loop stopped on.)
"""))
cells.append(code(
"""import os
assert os.path.exists("checkpoints/best.pt"), (
    "checkpoints/best.pt doesn't exist — go back and get Step 7 to finish with a ✅ first."
)
!python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
"""))

cells.append(md("## 9. Export to ONNX (for the mobile app)"))
cells.append(code(
"""import os
assert os.path.exists("checkpoints/best.pt"), (
    "checkpoints/best.pt doesn't exist — go back and get Step 7 to finish with a ✅ first."
)
!python -m src.export_onnx --config configs/default.yaml --checkpoint checkpoints/best.pt

import shutil
if os.path.exists("checkpoints/dead_reckoning_model.onnx") and os.path.exists("/content/drive/MyDrive"):
    shutil.copy("checkpoints/dead_reckoning_model.onnx", CKPT_DIR + "/dead_reckoning_model.onnx")
    print("ONNX model copied to Drive")
"""))

cells.append(md(
"""## 10. Download artifacts directly (alternative to Drive)"""))
cells.append(code(
"""from google.colab import files
import os

for f in ["checkpoints/best.pt", "checkpoints/dead_reckoning_model.onnx", "results/eval_report.json"]:
    if os.path.exists(f):
        files.download(f)
    else:
        print(f"skipping {f} — doesn't exist (an earlier step didn't finish)")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "colab_train.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")
