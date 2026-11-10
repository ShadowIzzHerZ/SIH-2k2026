"""
build_lr_scheduler must never send the LR back up mid-run. A resumed run uses
a short cosine (T_max = max(patience + 2, 10)) while the loop can run for the
full configured epoch count, so a periodic CosineAnnealingLR would climb back
to the peak by 2*T_max. The resume schedule must decay and then hold a floor.

Run:
    python -m pytest -q tests/test_lr_schedule.py
"""
from __future__ import annotations

import torch

from src.train import RESUME_LR_FLOOR_FRAC, build_lr_scheduler

BASE_LR = 1.0e-3
TRAIN_CFG = {"epochs": 60, "early_stop_patience": 8}


def _lrs(resume: bool, n_epochs: int) -> list[float]:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    sched = build_lr_scheduler(opt, TRAIN_CFG, resume=resume)
    lrs = []
    for _ in range(n_epochs):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return lrs


def test_resume_lr_decays_then_holds_floor():
    t_max = max(TRAIN_CFG["early_stop_patience"] + 2, 10)
    lrs = _lrs(resume=True, n_epochs=3 * t_max)
    assert lrs[0] == BASE_LR
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12, f"LR went back up: {a:.3e} -> {b:.3e}"
    floor = BASE_LR * RESUME_LR_FLOOR_FRAC
    assert abs(lrs[-1] - floor) < 1e-12
    assert all(abs(lr - floor) < 1e-12 for lr in lrs[t_max:])
    # Still a genuinely low LR inside the guaranteed patience window.
    assert lrs[TRAIN_CFG["early_stop_patience"] - 1] < 2.5e-4


def test_cold_start_lr_never_rises_within_run():
    lrs = _lrs(resume=False, n_epochs=TRAIN_CFG["epochs"])
    assert lrs[0] == BASE_LR
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12
