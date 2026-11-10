"""
src.train --resume promises to back up the previous best checkpoint and
train history (*_prev) before overwriting them. When the resume checkpoint
fails to load (missing, or an input_channels/architecture change), train.py
falls back to a cold start, and that path skipped the backup: the cold
start then overwrote checkpoints/best.pt and results/train_history.json with
nothing kept. Runs train.main() for one epoch on a tiny synthetic dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import src.train as train
from src.data.windowing import IOVNBDWindowDataset, Window

ROOT = Path(__file__).resolve().parents[1]


def _windows(n=8, T=50, dt=0.1):
    out = []
    t = np.arange(T) * dt
    for i in range(n):
        pos = np.stack([10.0 * t, np.zeros(T)], axis=1).astype(np.float32)
        accel = np.tile([0.0, 0.0, 9.81], (T, 1)).astype(np.float32)
        out.append(Window(accel, np.zeros((T, 3), np.float32), np.full(T, 10.0, np.float32), pos,
                          10.0, 0.0, 31.0, 75.0, dt, src="r", start=i * T))
    return out


def _run(tmp_path, monkeypatch, argv):
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    cfg["train"].update(epochs=1, batch_size=4, device="cpu")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    splits = {s: IOVNBDWindowDataset(_windows()) for s in ("train", "val", "test")}
    monkeypatch.setattr(train, "load_combined_dataset_splits", lambda **kw: splits)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["src.train", "--config", str(cfg_path), "--chain_chunks", "0", *argv])
    train.main()


def test_failed_resume_still_backs_up(tmp_path, monkeypatch):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "results").mkdir()
    old_ckpt = tmp_path / "checkpoints" / "best.pt"
    torch.save({"not": torch.zeros(1)}, old_ckpt)          # won't load into BiasCorrectionNet
    old_bytes = old_ckpt.read_bytes()
    old_hist = [{"epoch": 41, "train": {}, "val": {"drift_pct": 7.0}, "score": 7.0}]
    (tmp_path / "results" / "train_history.json").write_text(json.dumps(old_hist))

    _run(tmp_path, monkeypatch, ["--resume", "checkpoints/best.pt"])

    assert old_ckpt.read_bytes() != old_bytes                     # the cold start did overwrite it ...
    assert (tmp_path / "checkpoints" / "best_prev.pt").read_bytes() == old_bytes   # ... after a backup
    assert json.loads((tmp_path / "results" / "train_history_prev.json").read_text()) == old_hist


def test_successful_resume_continues_history(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, [])
    hist = json.loads((tmp_path / "results" / "train_history.json").read_text())
    assert [h["epoch"] for h in hist] == [0]
    _run(tmp_path, monkeypatch, ["--resume", "checkpoints/best.pt"])
    hist = json.loads((tmp_path / "results" / "train_history.json").read_text())
    assert [h["epoch"] for h in hist] == [0, 1]
    assert (tmp_path / "checkpoints" / "best_prev.pt").exists()
