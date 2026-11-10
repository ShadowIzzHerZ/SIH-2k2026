"""
eval.example_checks scoring (src/chain_eval.example_checks) on synthetic
drives, no data needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.chain_eval import example_checks
from src.data.io_vnbd_loader import ImuSequence
from src.data.windowing import IOVNBDWindowDataset, build_windows

DT = 0.1
ROOT = Path(__file__).resolve().parents[1]
CHECKS = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["eval"]["example_checks"]


class Correction(torch.nn.Module):
    def __init__(self, yaw_bias: float = 0.0):
        super().__init__()
        self.yaw_bias = yaw_bias

    def forward(self, imu):
        out = torch.zeros(imu.shape[0], imu.shape[1], 2)
        out[..., 1] = self.yaw_bias
        return out


def _dataset(speed_mps: float, n: int = 1500):
    t = np.arange(n) * DT
    lat = 30.0 + np.degrees(speed_mps * t / 6371000.0)   # due north
    seq = ImuSequence(path=Path("synthetic.csv"), time=t, accel=np.zeros((n, 3)), gyro=np.zeros((n, 3)),
                      speed_gt=np.full(n, speed_mps), lat=lat, lon=np.full(n, 76.0), heading_gt=np.zeros(n))
    return IOVNBDWindowDataset(build_windows(seq, window_size=50, stride=10, dt=DT))


def test_config_checks_are_the_ps_examples():
    assert [c["distance_m"] for c in CHECKS] == [50, 1000]


def test_good_model_passes_both_checks_at_highway_speed():
    res = example_checks(Correction(), _dataset(20.0), "cpu", DT, CHECKS)
    assert [r["distance_m"] for r in res] == [50, 1000]
    for r in res:
        assert r["n_blackouts"] > 0, r
        assert r["pass_rate_pct"] == 100.0, r


def test_yaw_bias_fails_the_long_check():
    res = example_checks(Correction(yaw_bias=0.05), _dataset(20.0), "cpu", DT, CHECKS)
    long = res[1]
    assert long["n_blackouts"] > 0 and long["pass_rate_pct"] == 0.0, long
    assert long["median_error_m"] > 100


def test_too_slow_for_one_km_in_a_minute_is_not_scored():
    res = example_checks(Correction(), _dataset(10.0), "cpu", DT, CHECKS)
    assert res[0]["n_blackouts"] > 0
    assert res[1]["n_blackouts"] == 0 and "pass_rate_pct" not in res[1]


def test_check_without_a_time_budget_is_rejected():
    with pytest.raises(ValueError):
        example_checks(Correction(), _dataset(20.0), "cpu", DT, [{"distance_m": 50, "max_drift_m": 5}])
