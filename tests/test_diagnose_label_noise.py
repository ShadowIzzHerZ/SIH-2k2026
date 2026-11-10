"""diagnose_label_noise must compare heading_gt (compass bearing, clockwise)
and gz (counter-clockwise) in the same sign convention. On a synthetic drive
where both sensors are perfect, the reported disagreement must be ~0 and the
correlation ~+1, whichever way the car turns."""
import sys
from pathlib import Path

import numpy as np

from src import diagnose_label_noise as dln
from src.data.io_vnbd_loader import ImuSequence


def _synthetic_drive() -> ImuSequence:
    dt = 0.1
    t = np.arange(600) * dt
    # left, straight, right; 0.2 rad/s is ~1.15 deg per step
    gz = np.where(t < 20, 0.2, np.where(t < 40, 0.0, -0.25))
    theta = np.concatenate([[0.3], 0.3 + np.cumsum(gz[:-1] * dt)])  # CCW from east
    bearing = (90.0 - np.degrees(theta)) % 360.0                      # compass: CW from north
    gyro = np.zeros((len(t), 3))
    gyro[:, 2] = gz
    return ImuSequence(
        path=Path("synthetic.csv"), time=t, accel=np.zeros((len(t), 3)), gyro=gyro,
        speed_gt=np.full(len(t), 8.0), lat=None, lon=None, heading_gt=bearing,
    )


def test_label_heading_ccw_matches_math_angle():
    out = dln.label_heading_ccw_deg(np.array([0.0, 90.0, 180.0, 270.0]))
    np.testing.assert_allclose(dln.wrap_deg(out - np.array([90.0, 0.0, -90.0, 180.0])), 0.0, atol=1e-9)


def test_perfect_sensors_agree(monkeypatch, capsys):
    seq = _synthetic_drive()
    monkeypatch.setattr(dln.glob, "glob", lambda *a, **k: ["synthetic.csv"])
    monkeypatch.setattr(dln, "load_sequence", lambda p: seq)
    monkeypatch.setattr(sys, "argv", ["diagnose_label_noise"])
    dln.main()
    rows = [l.split() for l in capsys.readouterr().out.splitlines() if l.startswith("[6.0, 10.0)")]
    assert len(rows) == 1
    n, rms, corr = int(rows[0][2]), float(rows[0][3]), float(rows[0][4])
    assert n == 599
    assert rms < 1e-6
    assert corr > 0.999


def _bin_row(monkeypatch, capsys, seq, argv):
    monkeypatch.setattr(dln.glob, "glob", lambda *a, **k: ["synthetic.csv"])
    monkeypatch.setattr(dln, "load_sequence", lambda p: seq)
    monkeypatch.setattr(sys, "argv", ["diagnose_label_noise", *argv])
    dln.main()
    out = capsys.readouterr().out
    rows = [l.split() for l in out.splitlines() if l.startswith("[6.0, 10.0)")]
    assert len(rows) == 1, out
    return float(rows[0][3]), float(rows[0][4]), out


def test_recover_yaw_is_opt_in(monkeypatch, capsys):
    """The yaw rate sits on the "wrong" gyro channel, as in IO-VNBD's phone
    logs. By default the script reads gz as-is (and finds no agreement);
    --recover-yaw rebuilds it with recover_yaw_axis first."""
    seq = _synthetic_drive()
    seq.gyro = seq.gyro[:, [2, 0, 1]]
    rms, _, out = _bin_row(monkeypatch, capsys, seq, [])
    assert rms > 0.5 and "recover_yaw_axis" not in out

    seq = _synthetic_drive()
    seq.gyro = seq.gyro[:, [2, 0, 1]]
    rms, corr, out = _bin_row(monkeypatch, capsys, seq, ["--recover-yaw"])
    assert rms < 0.05 and corr > 0.99, (rms, corr)
    assert "0 files skipped" in out
