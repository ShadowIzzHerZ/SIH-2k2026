"""edge: input shorter than one 10 Hz tick gives a clear error, not a numpy crash.

Run:
    python -m pytest tests/test_edge_short_input.py
"""
from __future__ import annotations

import numpy as np
import pytest

from edge.cli import main as cli_main
from edge.io import write_csv
from edge.resample import resample


@pytest.mark.parametrize("method", ["mean", "interp"])
def test_resample_empty_input(method):
    t, x = resample(np.zeros(0), np.zeros((0, 3)), 10.0, method)
    assert t.shape == (0,) and x.shape == (0, 3)


def test_resample_shorter_than_one_tick():
    t = np.arange(10) * 0.005                # 45 ms at 200 Hz
    t_out, x_out = resample(t, np.ones((10, 3)), 10.0, "interp")
    assert t_out.shape == (0,) and x_out.shape == (0, 3)


def test_cli_run_rejects_input_shorter_than_one_tick(tmp_path):
    path = tmp_path / "short.csv"
    rows = [[f"{i * 0.005:.3f}", "0", "0", "9.81", "0", "0", "0", "", "", "", ""] for i in range(10)]
    write_csv(str(path), ["t", "ax", "ay", "az", "gx", "gy", "gz", "lat", "lon", "speed", "heading"], rows)
    with pytest.raises(SystemExit) as e:
        cli_main(["run", str(path), "--out-dir", str(tmp_path / "out")])
    assert "less than one" in str(e.value)
