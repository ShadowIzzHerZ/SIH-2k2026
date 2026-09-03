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
column names → train → evaluate against the PS's <10% drift target → export ONNX.
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
"""## 6. Train

Runs inline (not as a background `!python` subprocess) so nothing about a
failure can be silently scrolled past: it prints the train/val window counts
up front, refuses to proceed if either is zero (the #1 cause of a missing
checkpoint later — it means the column_map isn't matching the real CSV
headers from Step 5), and ends with an explicit ✅ confirming the checkpoint
file actually exists on disk.
"""))

cells.append(code(
"""import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.windowing import load_dataset_splits
from src.models.bias_correction_net import BiasCorrectionNet
from src.train import pick_device, forward_pass, compute_loss

cfg = yaml.safe_load(open("configs/default.yaml"))
device = pick_device(cfg["train"]["device"])
print("device:", device)

splits = load_dataset_splits(
    data_root=cfg["data"]["root"],
    variant=cfg["data"]["variant"],
    column_map=cfg["data"]["column_map"],
    sample_rate_hz=cfg["data"]["sample_rate_hz"],
    window_size=cfg["data"]["window_size"],
    window_stride=cfg["data"]["window_stride"],
    train_split=cfg["data"]["train_split"],
    val_split=cfg["data"]["val_split"],
    file_prefix=cfg["data"].get("file_prefix", ""),
)

n_train, n_val = len(splits["train"]), len(splits["val"])
print(f"\\ntrain windows: {n_train} | val windows: {n_val}")

if n_train == 0 or n_val == 0:
    raise RuntimeError(
        "STOPPING: zero usable windows. This means load_sequence() couldn't resolve "
        "the real accel/gyro/lat/lon columns for any file — the column_map in "
        "configs/default.yaml doesn't match the actual CSV headers from Step 5. "
        "Fix column_map (edit + re-run the '%%writefile configs/default.yaml' cell "
        "above) before re-running this cell."
    )

train_loader = DataLoader(splits["train"], batch_size=cfg["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(splits["val"], batch_size=cfg["train"]["batch_size"], shuffle=False)

model = BiasCorrectionNet(
    input_channels=cfg["model"]["input_channels"],
    cnn_channels=cfg["model"]["cnn_channels"],
    cnn_kernel_size=cfg["model"]["cnn_kernel_size"],
    gru_hidden=cfg["model"]["gru_hidden"],
    gru_layers=cfg["model"]["gru_layers"],
    dropout=cfg["model"]["dropout"],
    output_dim=cfg["model"]["output_dim"],
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
dt = 1.0 / cfg["data"]["sample_rate_hz"]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

best_val_drift = float("inf")
for epoch in range(cfg["train"]["epochs"]):
    model.train()
    for batch in train_loader:
        opt.zero_grad()
        speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
        loss, _ = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    val_drift_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            speed_pred, pos_pred, speed_gt, pos_gt = forward_pass(model, batch, dt, device)
            _, metrics = compute_loss(speed_pred, pos_pred, speed_gt, pos_gt, cfg["train"]["loss_weights"])
            val_drift_sum += metrics["drift_pct"]
            n_batches += 1
    val_drift = val_drift_sum / max(1, n_batches)
    print(f"epoch {epoch:03d} | val drift {val_drift:.2f}%")

    if val_drift < best_val_drift:
        best_val_drift = val_drift
        torch.save(model.state_dict(), "checkpoints/best.pt")
        print(f"  -> saved checkpoint (val drift {val_drift:.2f}%)")

assert os.path.exists("checkpoints/best.pt"), "checkpoint still missing after training loop — something is wrong"
print(f"\\n✅ done. best val drift: {best_val_drift:.2f}% | checkpoints/best.pt: {os.path.getsize('checkpoints/best.pt')} bytes")
"""))

cells.append(md(
"""## 7. Copy checkpoint + history to Drive (if mounted)"""))
cells.append(code(
"""import shutil, os

assert os.path.exists("checkpoints/best.pt"), (
    "checkpoints/best.pt doesn't exist — Step 6 hasn't completed successfully yet. "
    "Scroll up, fix whatever it printed, and re-run Step 6 until you see the ✅ line."
)

if os.path.exists("/content/drive/MyDrive"):
    shutil.copy("checkpoints/best.pt", CKPT_DIR + "/best.pt")
    if os.path.exists("results/train_history.json"):
        shutil.copy("results/train_history.json", RESULTS_DIR + "/train_history.json")
    print("copied to Drive")
else:
    print("Drive not mounted — checkpoint only in this session's /content/checkpoints")
"""))

cells.append(md(
"""## 8. Evaluate against the PS's <10% drift benchmark"""))
cells.append(code(
"""import os
assert os.path.exists("checkpoints/best.pt"), (
    "checkpoints/best.pt doesn't exist — go back and get Step 6 to finish with a ✅ first."
)
!python -m src.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
"""))

cells.append(md("## 9. Export to ONNX (for the mobile app)"))
cells.append(code(
"""import os
assert os.path.exists("checkpoints/best.pt"), (
    "checkpoints/best.pt doesn't exist — go back and get Step 6 to finish with a ✅ first."
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
