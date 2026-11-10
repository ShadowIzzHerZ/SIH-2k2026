"""
Step timing must survive a to_state_dict/from_state_dict round trip, otherwise a
filter restored from Redis counts a spurious step on its first moving sample.
"""
import unittest

from app.fusion.kalman_filter import IMUKalmanFilter


def _moving_packet(ts):
    return {
        "timestamp": ts,
        "mode": "moving",
        "ax": 0.1, "ay": 1.2, "az": 10.5,
        "gx": 0.0, "gy": 0.0, "gz": 0.0,
    }


class TestKalmanStepTimeRestore(unittest.TestCase):
    def test_restored_filter_matches_live_step_count(self):
        t0 = 1_700_000_000.0
        kf = IMUKalmanFilter()
        res = kf.process_sample(_moving_packet(t0))
        self.assertEqual(res["step_count"], 1)

        restored = IMUKalmanFilter.from_state_dict(kf.to_state_dict())

        # 0.1 s later is well inside the 0.55 s cadence window: no new step.
        live = kf.process_sample(_moving_packet(t0 + 0.1))
        again = restored.process_sample(_moving_packet(t0 + 0.1))
        self.assertEqual(live["step_count"], 1)
        self.assertEqual(again["step_count"], live["step_count"])

    def test_old_state_dict_without_step_time_loads(self):
        kf = IMUKalmanFilter()
        state = kf.to_state_dict()
        state.pop("last_step_time", None)
        restored = IMUKalmanFilter.from_state_dict(state)
        self.assertEqual(restored._last_step_time, 0.0)


if __name__ == "__main__":
    unittest.main()
