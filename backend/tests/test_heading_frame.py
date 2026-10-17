# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""The magnetometer update must agree with the filter's yaw convention (CCW from East)."""

import unittest
import numpy as np

from app.fusion.coordinates import euler_to_rotation_matrix
from app.fusion.kalman_filter import IMUKalmanFilter

# Earth field in the ENU navigation frame: points North and down
B_NAV = np.array([0.0, 25.0, -40.0])


def _mag_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return R.T @ B_NAV


class TestMagnetometerHeadingFrame(unittest.TestCase):
    def test_mag_update_keeps_correct_yaw(self):
        for yaw_deg in (0.0, 30.0, 90.0, 179.0, -90.0):
            for roll, pitch in ((0.0, 0.0), (0.3, 0.0), (0.0, -0.3)):
                yaw = np.deg2rad(yaw_deg)
                kf = IMUKalmanFilter()
                kf.x[6, 0], kf.x[7, 0], kf.x[8, 0] = roll, pitch, yaw
                kf.update_magnetometer(_mag_body(roll, pitch, yaw))
                err = (float(kf.x[8, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
                self.assertAlmostEqual(err, 0.0, places=6, msg=f"yaw={yaw_deg} roll={roll} pitch={pitch}")

    def test_facing_north_walks_north(self):
        kf = IMUKalmanFilter()
        mag = _mag_body(0.0, 0.0, np.pi / 2)  # body x-axis pointing North
        for i in range(60):
            kf.process_sample({
                "timestamp": 1000.0 + i * 0.1, "mode": "moving",
                "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "mx": mag[0], "my": mag[1], "mz": mag[2],
            })
        # Displacement should be mostly along +y (North), not +x (East)
        self.assertGreater(kf.x[1, 0], 3.0 * abs(kf.x[0, 0]))


if __name__ == "__main__":
    unittest.main()
