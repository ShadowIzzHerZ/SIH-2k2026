# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
Live frames (P0) are processed before older backlog (P1) in the same filter. Backlog
older than the filter state must not step it backwards, but is still stored.
"""
import asyncio
import unittest
from unittest import mock

import numpy as np

from app.api import websocket
from app.fusion.kalman_filter import IMUKalmanFilter


def _packet(ts, seq, **kw):
    p = {"seq": seq, "timestamp": ts, "ax": 0.0, "ay": 0.0, "az": 9.81,
         "gx": 0.0, "gy": 0.0, "gz": 0.2, "mode": "moving"}
    p.update(kw)
    return p


class TestStaleSamplesInFilter(unittest.TestCase):
    def test_older_sample_leaves_state_untouched(self):
        kf = IMUKalmanFilter()
        for i in range(10):
            out = kf.process_sample(_packet(200.0 + 0.1 * i, i))
            self.assertFalse(out["stale"])
        x, P, ts, steps = kf.x.copy(), kf.P.copy(), kf.last_timestamp, kf.step_count

        out = kf.process_sample(_packet(150.0, 100))
        self.assertTrue(out["stale"])
        np.testing.assert_array_equal(kf.x, x)
        np.testing.assert_array_equal(kf.P, P)
        self.assertEqual(kf.last_timestamp, ts)
        self.assertEqual(kf.step_count, steps)
        self.assertEqual(out["x"], float(x[0, 0]))

        # Newer samples keep fusing normally
        self.assertFalse(kf.process_sample(_packet(201.5, 11))["stale"])
        self.assertEqual(kf.last_timestamp, 201.5)

    def test_packets_without_timestamp_still_fuse(self):
        kf = IMUKalmanFilter()
        kf.last_timestamp = 500.0
        p = _packet(0.0, 1)
        del p["timestamp"]
        self.assertFalse(kf.process_sample(p)["stale"])


class TestWorkerStoresStaleBacklog(unittest.IsolatedAsyncioTestCase):
    async def test_backlog_after_live_is_stored_unverified(self):
        dev = "stale_backlog_dev"
        persisted = []

        async def capture(payload):
            persisted.append(payload)

        queue = websocket.IngestionQueueManager(maxsize=100)
        patches = [
            mock.patch.object(websocket, "queue_mgr", queue),
            mock.patch.object(websocket, "_persist_point", capture),
            mock.patch.object(websocket.cache_manager, "set_live_device_state", mock.AsyncMock()),
        ]
        for p in patches:
            p.start()
        websocket.device_kalman_filters[dev] = IMUKalmanFilter()
        # Backlog from an outage (t=100..100.4) is queued together with live frames
        # (t=200..); the queue hands over all live frames first.
        await queue.enqueue_backlog_chunk(dev, [_packet(100.0 + 0.1 * i, i + 1) for i in range(5)])
        for i in range(5):
            await queue.enqueue_live(dev, _packet(200.0 + 0.1 * i, 10 + i))
        worker = asyncio.create_task(websocket.queue_worker())
        try:
            await asyncio.wait_for(queue._queue.join(), timeout=5)
            await asyncio.sleep(0)  # let the persist tasks run
        finally:
            worker.cancel()
            for p in patches:
                p.stop()
            kf = websocket.device_kalman_filters.pop(dev)

        self.assertEqual(len(persisted), 10)
        live = [p for p in persisted if not p["is_backlog"]]
        backlog = [p for p in persisted if p["is_backlog"]]
        self.assertTrue(all(p["is_verified"] for p in live))
        self.assertTrue(all(not p["is_verified"] for p in backlog))
        self.assertEqual([p["seq"] for p in backlog], [1, 2, 3, 4, 5])
        # The live filter stayed at the newest live time
        self.assertAlmostEqual(kf.last_timestamp, 200.4)


if __name__ == "__main__":
    unittest.main()
