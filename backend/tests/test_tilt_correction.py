# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Roll/pitch get a complementary accelerometer correction instead of pure gyro integration."""
import unittest

import numpy as np

from app.fusion.coordinates import GRAVITY, euler_to_rotation_matrix
from app.fusion.kalman_filter import IMUKalmanFilter


def _run_still(kf, roll, pitch, gyro_bias, seconds=60.0, rate_hz=10.0):
    # Specific force of a device at rest with this tilt: R^T [0, 0, g]
    f_body = euler_to_rotation_matrix(roll, pitch, 0.0).T @ np.array([0.0, 0.0, GRAVITY])
    t = 1000.0
    for _ in range(int(seconds * rate_hz)):
        t += 1.0 / rate_hz
        kf.process_sample({
            "timestamp": t, "mode": "stationary",
            "ax": f_body[0], "ay": f_body[1], "az": f_body[2],
            "gx": gyro_bias, "gy": gyro_bias, "gz": 0.0,
        })
    return float(kf.x[6, 0]), float(kf.x[7, 0])


class TestTiltCorrection(unittest.TestCase):
    def test_gyro_only_tilt_drifts_with_bias(self):
        # The previous behaviour (no accel correction): a 0.01 rad/s bias integrates to 0.6 rad in 60 s
        kf = IMUKalmanFilter()
        kf._tilt_gate = 0.0
        roll, pitch = _run_still(kf, 0.0, 0.0, gyro_bias=0.01)
        self.assertAlmostEqual(roll, 0.6, delta=0.02)
        self.assertAlmostEqual(pitch, 0.6, delta=0.02)

    def test_accel_correction_bounds_drift(self):
        roll, pitch = _run_still(IMUKalmanFilter(), 0.0, 0.0, gyro_bias=0.01)
        # Steady state error is about bias * tau = 0.02 rad
        self.assertLess(abs(roll), 0.05)
        self.assertLess(abs(pitch), 0.05)

    def test_converges_to_true_tilt(self):
        roll, pitch = _run_still(IMUKalmanFilter(), np.radians(30), np.radians(-20), gyro_bias=0.0, seconds=30)
        self.assertAlmostEqual(roll, np.radians(30), delta=0.01)
        self.assertAlmostEqual(pitch, np.radians(-20), delta=0.01)

    def test_no_correction_under_strong_acceleration(self):
        kf = IMUKalmanFilter()
        kf.last_timestamp = 1000.0
        # |a| = 1.5 g: not gravity-dominated, so roll stays gyro-only
        kf.process_sample({"timestamp": 1000.1, "ax": 0.0, "ay": 1.5 * GRAVITY, "az": 0.0,
                           "gx": 0.0, "gy": 0.0, "gz": 0.0, "mode": "stationary"})
        self.assertEqual(float(kf.x[6, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
