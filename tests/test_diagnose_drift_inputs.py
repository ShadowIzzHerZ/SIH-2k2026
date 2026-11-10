"""diagnose_drift_inputs must report a label lag with the right sign and size,
and its drift figures must fall when that lag is undone. Checked on a synthetic
drive whose labels are the truth delayed by a known amount."""
from pathlib import Path

import numpy as np

from src import diagnose_drift_inputs as ddi
from src.data.io_vnbd_loader import ImuSequence

DT = 0.1


def _drive(label_delay_s: float, gz_scale: float = 1.0, n: int = 3000) -> ImuSequence:
    """A calibrated drive (accel x forward, gyro z yaw rate) that weaves and
    changes speed. Labels are the true state label_delay_s earlier."""
    t = np.arange(n) * DT
    gz = 0.15 * np.sin(2 * np.pi * t / 23.0) + 0.1 * np.sin(2 * np.pi * t / 7.0)
    a = 0.6 * np.sin(2 * np.pi * t / 17.0) + 0.3 * np.sin(2 * np.pi * t / 5.0)
    theta = 0.3 + np.concatenate([[0.0], np.cumsum(gz[:-1]) * DT])
    v = 12.0 + np.concatenate([[0.0], np.cumsum(a[:-1]) * DT])
    step = np.stack([v * np.cos(theta), v * np.sin(theta)], axis=1) * DT
    xy = np.concatenate([[[0.0, 0.0]], np.cumsum(step[:-1], axis=0)])

    k = int(round(label_delay_s / DT))
    late = lambda x: np.concatenate([np.repeat(x[:1], k, axis=0), x[:len(x) - k]]) if k else x   # noqa: E731
    theta_l, v_l, xy_l = late(theta), late(v), late(xy)
    R, lat0 = 6371000.0, 12.9
    lat = lat0 + np.degrees(xy_l[:, 1] / R)
    lon = 77.6 + np.degrees(xy_l[:, 0] / (R * np.cos(np.radians(lat0))))
    accel = np.zeros((n, 3))
    accel[:, 0] = a
    accel[:, 2] = 9.81
    gyro = np.zeros((n, 3))
    gyro[:, 2] = gz * gz_scale
    return ImuSequence(path=Path("synthetic.csv"), time=t, accel=accel, gyro=gyro, speed_gt=v_l,
                       lat=lat, lon=lon, heading_gt=(90.0 - np.degrees(theta_l)) % 360.0)


def test_no_lag_reads_as_zero_and_integration_is_exact():
    row = ddi.diagnose_sequence(_drive(0.0))
    assert abs(row["heading_lag_s"]) < 0.06
    assert abs(row["speed_lag_s"]) < 0.06
    assert np.median(row["drift_30s_pct"]["as_trained"]) < 0.5


def test_late_labels_read_as_a_positive_lag():
    row = ddi.diagnose_sequence(_drive(0.7))
    assert abs(row["heading_lag_s"] - 0.7) < 0.06
    assert abs(row["speed_lag_s"] - 0.7) < 0.06
    assert row["heading_corr_at_lag"] > row["heading_corr_at_zero"]


def test_undoing_the_lag_removes_the_drift_it_caused():
    row = ddi.diagnose_sequence(_drive(0.7))
    late, fixed = (np.median(row["drift_30s_pct"][k]) for k in ("as_trained", "labels_shifted"))
    assert late > 1.0
    assert fixed < 0.25 * late


def test_early_labels_read_as_a_negative_lag():
    row = ddi.diagnose_sequence(_drive(0.0))
    seq = _drive(0.0)
    seq.gyro[:, 2] = np.concatenate([np.zeros(4), seq.gyro[:-4, 2]])      # IMU 0.4 s late instead
    row = ddi.diagnose_sequence(seq)
    assert abs(row["heading_lag_s"] + 0.4) < 0.06


def test_a_worse_yaw_rate_is_reported_as_worse():
    seq = _drive(0.0, gz_scale=0.8)                  # what leveling left: 20% of the yaw rate lost
    true_gz = seq.gyro[:, 2] / 0.8                   # what the recovery had
    row = ddi.diagnose_sequence(seq, gz_before_leveling=true_gz)
    assert row["yaw_err_deg_10s_before_leveling"] < 0.1 < row["yaw_err_deg_10s_after_leveling"]
    assert np.median(row["drift_30s_pct"]["gz_from_recovery"]) < np.median(row["drift_30s_pct"]["as_trained"])


def test_shift_labels_pads_with_nan_and_moves_the_right_way():
    x = np.arange(10.0)
    np.testing.assert_array_equal(ddi.shift_labels(x, 0.3)[:7], x[3:])
    assert np.isnan(ddi.shift_labels(x, 0.3)[7:]).all()
    np.testing.assert_array_equal(ddi.shift_labels(x, -0.2)[2:], x[:8])
    np.testing.assert_array_equal(ddi.shift_labels(x, 0.0), x)


def test_summary_pools_windows_across_files():
    rows = [{"file": f, "leveling_tilt_deg": 3.0, **ddi.diagnose_sequence(_drive(d), gz_before_leveling=_drive(d).gyro[:, 2])}
            for f, d in (("a", 0.0), ("b", 0.6))]
    s = ddi.summarise(rows)
    assert s["n_files"] == 2
    assert abs(s["heading_lag_s_median"] - 0.3) < 0.06
    assert s["drift_30s_windows"]["as_trained"] == sum(len(r["drift_30s_pct"]["as_trained"]) for r in rows)
    assert s["n_files_yaw_worse_after_leveling"] == 0
