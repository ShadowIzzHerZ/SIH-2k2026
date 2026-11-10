"""
Config values that can never work must fail up front with a clear
ValueError, not as a shape error / IndexError deep inside training:
  - an even cnn_kernel_size makes every conv layer one sample longer than
    its input, so the corrections no longer line up with the T-sample window
  - heading_end_steps >= window_size indexes past the start of the window
Valid configs (including the shipped ones) must pass unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.models.bias_correction_net import BiasCorrectionNet
from src.train import compute_loss, validate_config

ROOT = Path(__file__).resolve().parent.parent


def _cfg():
    return yaml.safe_load(open(ROOT / "configs/default.yaml"))


def test_shipped_configs_pass():
    for path in ("configs/default.yaml", "configs/engineered_features.yaml"):
        validate_config(yaml.safe_load(open(ROOT / path)))


def test_even_kernel_rejected():
    cfg = _cfg()
    cfg["model"]["cnn_kernel_size"] = 4
    with pytest.raises(ValueError, match="cnn_kernel_size"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="cnn_kernel_size"):
        BiasCorrectionNet(cnn_kernel_size=4)


def test_odd_kernels_keep_length():
    for k in (1, 3, 5, 7):
        out = BiasCorrectionNet(cnn_kernel_size=k)(torch.zeros(2, 50, 6))
        assert out.shape == (2, 50, 2)


def test_heading_end_steps_too_long_rejected():
    cfg = _cfg()
    T = cfg["data"]["window_size"]
    cfg["train"]["loss_weights"]["heading_end"] = 1.0
    cfg["train"]["loss_weights"]["heading_end_steps"] = T
    with pytest.raises(ValueError, match="heading_end_steps"):
        validate_config(cfg)
    # T-1 is the longest span that fits and must stay allowed
    ok = copy.deepcopy(cfg)
    ok["train"]["loss_weights"]["heading_end_steps"] = T - 1
    validate_config(ok)
    pos = torch.cumsum(torch.ones(2, T, 2), dim=1)
    compute_loss(torch.ones(2, T), pos, torch.ones(2, T), pos, ok["train"]["loss_weights"])


def test_heading_end_steps_ignored_when_term_off():
    cfg = _cfg()
    cfg["train"]["loss_weights"]["heading_end"] = 0.0
    cfg["train"]["loss_weights"]["heading_end_steps"] = 10_000
    validate_config(cfg)
