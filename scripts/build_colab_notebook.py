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
    "src/chain_eval.py",  # src.train and src.evaluate import it
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
"""## 6. Get the checkpoint to resume from

Same scheme as `scripts/train_until_target.sh`: the loop warm-starts from
`checkpoints/best_v3chain.pt` (the checkpoint the app ships, see
`checkpoints/README.md`) and writes its own results to
`checkpoints/best_loop.pt` / `results/eval_report_loop.json`, so it can never
overwrite `best_v3chain.pt` or the older `best.pt`. This cell gets, in order:
1. `best_loop.pt` from Google Drive, if a previous Colab run left one (Step 2)
2. `best_v3chain.pt` from the GitHub repo (the starting point)
3. if neither is reachable, Step 7 trains from scratch

`src.train`'s `--resume` flag (below) then warm-starts from whatever this
cell finds, instead of random init.
"""))

cells.append(code(
"""import os
import shutil
import urllib.request

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/ShadowIzzHerZ/SIH-2k2026/main"
RESUME_FROM = "checkpoints/best_v3chain.pt"  # the shipped model
RUN_NAME = "loop"
OUT = f"checkpoints/best_{RUN_NAME}.pt"     # what the loop trains into
REPORT = f"results/eval_report_{RUN_NAME}.json"

drive_ok = os.path.exists("/content/drive/MyDrive")
if drive_ok and os.path.exists(CKPT_DIR + f"/best_{RUN_NAME}.pt"):
    shutil.copy(CKPT_DIR + f"/best_{RUN_NAME}.pt", OUT)
    if os.path.exists(RESULTS_DIR + f"/train_history_{RUN_NAME}.json"):
        shutil.copy(RESULTS_DIR + f"/train_history_{RUN_NAME}.json", f"results/train_history_{RUN_NAME}.json")
    print(f"resuming from {OUT} on Google Drive (previous Colab run)")
try:
    urllib.request.urlretrieve(GITHUB_RAW + "/" + RESUME_FROM, RESUME_FROM)
    print(f"downloaded {RESUME_FROM} from the GitHub repo")
except Exception as e:
    print(f"could not download {RESUME_FROM} ({e})")
if not os.path.exists(OUT) and not os.path.exists(RESUME_FROM):
    print("no checkpoint found anywhere — Step 7 will train from scratch")
"""))

cells.append(md(
"""## 7. Train until the target drift is hit (or this runtime ends)

Each cycle runs a full `src.train` pass with `--run_name loop` (resuming from
`best_loop.pt` once one exists, otherwise from `best_v3chain.pt`) — a
cosine-annealed "warm restart" from the current best weights — then
evaluates on the real held-out test set. `src.train` only writes
`best_loop.pt` when it beats the weights it started from. It keeps looping,
checkpointing to Drive after every cycle so nothing is lost if Colab
disconnects mid-run, until either the PS's <10% drift target is hit or the
cycle cap below is reached (a safety backstop against a runaway loop, not a
real target — at ~60 epochs/cycle it's far more cycles than one Colab
session will ever reach, so in practice this only stops on target-hit,
failure, or the runtime itself ending).

`TRAIN_ARGS` is passed through to `src.train`, e.g.
`["--chain_train_chunks", "3"]` to keep the chained-chunk objective
`best_v3chain.pt` was fine-tuned with.
"""))

cells.append(code(
"""import subprocess
import sys
import json
import shutil
import os

TARGET_DRIFT_PCT = 10.0
MAX_CYCLES = 500  # safety backstop only, see markdown above — not a real limit
TRAIN_ARGS = []


def current_checkpoint():
    # best_loop.pt once this loop has improved on its starting point, else
    # the starting point itself (None = nothing yet, cold start)
    for p in (OUT, RESUME_FROM):
        if os.path.exists(p):
            return p
    return None


cycle = 0
while cycle < MAX_CYCLES:
    cycle += 1
    start = current_checkpoint()
    resume_args = ["--resume", start] if start else []
    print(f"\\n{'='*70}\\ncycle {cycle} {('(warm restart from ' + start + ')') if start else '(cold start)'}\\n{'='*70}")

    # -u: unbuffered stdout. Without it, src.train's stdout is piped (not a
    # TTY) so Python fully block-buffers it — nothing appears in the Colab
    # cell until the whole cycle's subprocess exits and flushes everything
    # at once, which can look like a silent hang for many minutes on a
    # cold-start cycle. -u makes per-epoch progress genuinely live here too.
    train_ret = subprocess.run([sys.executable, "-u", "-m", "src.train", "--config", "configs/default.yaml",
                                *resume_args, "--run_name", RUN_NAME, *TRAIN_ARGS])
    if train_ret.returncode != 0:
        print("training subprocess exited with an error — stopping the loop, see output above")
        break

    if drive_ok and os.path.exists(OUT):
        shutil.copy(OUT, CKPT_DIR + f"/best_{RUN_NAME}.pt")
        if os.path.exists(f"results/train_history_{RUN_NAME}.json"):
            shutil.copy(f"results/train_history_{RUN_NAME}.json", RESULTS_DIR + f"/train_history_{RUN_NAME}.json")

    ckpt = current_checkpoint()
    eval_ret = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--config", "configs/default.yaml", "--checkpoint", ckpt,
         "--run_name", RUN_NAME],
        capture_output=True, text=True,
    )
    print(eval_ret.stdout[-1500:])
    if eval_ret.returncode != 0 or not os.path.exists(REPORT):
        print("evaluate failed or produced no report — stopping the loop, see output above")
        break

    report = json.load(open(REPORT))
    mean_drift = report["mean_drift_pct"]
    print(f"cycle {cycle}: {ckpt} test mean drift {mean_drift:.2f}% (target < {TARGET_DRIFT_PCT}%)")
    if drive_ok:
        shutil.copy(REPORT, RESULTS_DIR + f"/eval_report_{RUN_NAME}.json")

    if mean_drift < TARGET_DRIFT_PCT:
        print(f"\\n🎯 target reached after {cycle} cycle(s) — stopping.")
        break
else:
    print(f"\\nhit the {MAX_CYCLES}-cycle safety cap without reaching target — this would be very unusual; check results/train_history_{RUN_NAME}.json for what's actually happening (plateaued vs. still improving).")

ckpt = current_checkpoint()
assert ckpt, "no checkpoint at all — something is wrong, scroll up"
print(f"\\n✅ {ckpt}: {os.path.getsize(ckpt)} bytes"
      + ("" if ckpt == OUT else f" (no cycle beat {RESUME_FROM}, so there is no {OUT})"))
"""))

cells.append(md(
"""## 8. Evaluate against the PS's <10% drift benchmark

(Step 7's loop already ran this each cycle — this cell just re-confirms the
final number on the checkpoint the loop stopped on: `best_loop.pt`, or
`best_v3chain.pt` if no cycle improved on it.)
"""))
cells.append(code(
"""import os
ckpt = OUT if os.path.exists(OUT) else RESUME_FROM
assert os.path.exists(ckpt), f"{ckpt} doesn't exist — go back and get Step 7 to finish with a ✅ first."
!python -m src.evaluate --config configs/default.yaml --checkpoint {ckpt} --run_name {RUN_NAME}
"""))

cells.append(md(
"""## 9. Export to ONNX (for the mobile app)

Exports `best_loop.pt` to `checkpoints/dead_reckoning_model_loop.onnx`, not
to the config's `checkpoints/dead_reckoning_model.onnx`: that name is the
model the app ships, and copying a Colab export over it by accident would
replace it with whatever this run produced. Compare
`results/eval_report_loop.json` against `results/README.md` first, then copy
it over the shipped file by hand if it really is better. If no cycle beat
`best_v3chain.pt`, there is nothing new to export.
"""))
cells.append(code(
"""import os
import shutil
import yaml

ONNX_OUT = f"checkpoints/dead_reckoning_model_{RUN_NAME}.onnx"
if not os.path.exists(OUT):
    print(f"{OUT} doesn't exist (no cycle beat {RESUME_FROM}) — nothing new to export; "
          f"the shipped ONNX is already {RESUME_FROM}.")
else:
    export_cfg = yaml.safe_load(open("configs/default.yaml"))
    export_cfg["export"]["onnx_path"] = ONNX_OUT
    with open(f"configs/export_{RUN_NAME}.yaml", "w") as f:
        yaml.safe_dump(export_cfg, f)
    !python -m src.export_onnx --config configs/export_{RUN_NAME}.yaml --checkpoint {OUT}
    if os.path.exists(ONNX_OUT) and drive_ok:
        shutil.copy(ONNX_OUT, CKPT_DIR + f"/dead_reckoning_model_{RUN_NAME}.onnx")
        print("ONNX model copied to Drive")
"""))

cells.append(md(
"""## 10. Download artifacts directly (alternative to Drive)"""))
cells.append(code(
"""from google.colab import files
import os

for f in [OUT, f"checkpoints/dead_reckoning_model_{RUN_NAME}.onnx", REPORT]:
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
