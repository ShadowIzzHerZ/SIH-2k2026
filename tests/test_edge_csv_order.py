"""
edge.io.load_csv: the engine replays rows in file order like a live stream,
so an out-of-order or missing `t` must be reported, not passed through.
"""
from __future__ import annotations

import numpy as np
import pytest

from edge.io import load_csv

HDR = "t,ax,ay,az,gx,gy,gz,x,y,speed,heading\n"


def _write(tmp_path, lines):
    p = tmp_path / "drive.csv"
    p.write_text(HDR + "".join(lines))
    return str(p)


def _imu(t):
    return f"{t},0,0,9.81,0,0,0,,,,\n"


def test_sorted_file_loads(tmp_path):
    # equal timestamps (an IMU row and a GNSS row at the same t) are fine
    lines = [_imu(i * 0.01) for i in range(5)] + ["0.04,,,,,,,1,2,3,90\n"]
    rec = load_csv(_write(tmp_path, lines))
    assert np.all(np.diff(rec.t_imu) > 0)
    assert rec.t_gnss.tolist() == [0.04]


def test_backwards_time_raises(tmp_path):
    lines = [_imu(t) for t in (0.00, 0.01, 0.02, 0.015, 0.03)]
    with pytest.raises(ValueError, match="not sorted by t"):
        load_csv(_write(tmp_path, lines))


def test_missing_time_raises(tmp_path):
    lines = [_imu(0.0), _imu(0.01), ",0,0,9.81,0,0,0,,,,\n", _imu(0.03)]
    with pytest.raises(ValueError, match="no t"):
        load_csv(_write(tmp_path, lines))
