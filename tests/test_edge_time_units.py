"""
edge.io.load_csv: `t` must be seconds. Millisecond or nanosecond stamps used
to load fine and give a 0.2 Hz or 2e-7 Hz "IMU"; now they raise, naming the
unit to pass with --time-unit, and --time-unit converts them.
"""
from __future__ import annotations

import pytest

from edge.cli import build_parser
from edge.io import load_csv


def _write(tmp_path, scale, hz=200.0, n=50):
    p = tmp_path / "drive.csv"
    rows = [f"{int(round(i / hz * scale)) if scale > 1 else i / hz},0,0,9.81,0,0,0" for i in range(n)]
    p.write_text("t,ax,ay,az,gx,gy,gz\n" + "\n".join(rows) + "\n")
    return str(p)


def test_seconds_load(tmp_path):
    rec = load_csv(_write(tmp_path, 1))
    assert abs(rec.t_imu[1] - rec.t_imu[0] - 0.005) < 1e-9


@pytest.mark.parametrize("unit,scale", [("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
def test_wrong_unit_raises_and_names_fix(tmp_path, unit, scale):
    path = _write(tmp_path, scale)
    with pytest.raises(ValueError, match=f"--time-unit {unit}"):
        load_csv(path)
    rec = load_csv(path, time_unit=unit)
    assert abs(rec.t_imu[1] - rec.t_imu[0] - 0.005) < 1e-6


def test_cli_has_time_unit_flag():
    args = build_parser().parse_args(["run", "x.csv", "--time-unit", "ms"])
    assert args.time_unit == "ms"
