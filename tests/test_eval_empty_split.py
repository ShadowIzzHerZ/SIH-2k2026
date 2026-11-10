"""
evaluate.eval_drift on an empty test split (e.g. a per-dataset breakdown
where every window of that dataset landed in train/val) must report zero
windows instead of crashing the whole evaluation with an IndexError.
"""
from __future__ import annotations

import math

import torch

from src.evaluate import eval_drift
from src.models.bias_correction_net import BiasCorrectionNet


class _Empty(torch.utils.data.Dataset):
    windows: list = []

    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


def test_eval_drift_empty_dataset():
    model = BiasCorrectionNet().eval()
    report = eval_drift(model, _Empty(), 0.1, torch.device("cpu"), 8, target_pct=10.0)
    assert report["n_windows"] == 0
    assert math.isnan(report["mean_drift_pct"])
    assert math.isnan(report["pass_rate_pct"])
