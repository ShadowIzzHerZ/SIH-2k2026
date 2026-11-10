# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Per-device in-memory state is freed once a device goes idle."""
import asyncio
import os
import unittest

os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

from app.api import websocket as ws_mod
from app.core.queue_manager import IngestionQueueManager
from app.fusion.kalman_filter import IMUKalmanFilter


class TestSeqCounters(unittest.IsolatedAsyncioTestCase):
    async def test_live_frames_without_seq_are_numbered_and_forgettable(self):
        q = IngestionQueueManager(maxsize=10)
        await q.enqueue_live("dev", {"timestamp": 1.0})
        await q.enqueue_live("dev", {"timestamp": 1.1})
        await q.enqueue_live("dev", {"seq": 40, "timestamp": 1.2})
        await q.enqueue_live("dev", {"timestamp": 1.3})
        seqs = [(await q.get_next_item()).seq_num for _ in range(4)]
        self.assertEqual(seqs, [1, 2, 40, 41])
        q.forget_device("dev")
        self.assertEqual(q._device_seq_counters, {})


class TestIdleSweep(unittest.TestCase):
    def setUp(self):
        self._saved = (dict(ws_mod.device_kalman_filters), dict(ws_mod._device_last_seen),
                       dict(ws_mod.active_connections))
        ws_mod.device_kalman_filters.clear()
        ws_mod._device_last_seen.clear()
        ws_mod.active_connections.clear()

    def tearDown(self):
        for d, saved in zip((ws_mod.device_kalman_filters, ws_mod._device_last_seen,
                             ws_mod.active_connections), self._saved):
            d.clear()
            d.update(saved)

    def test_sweep_frees_only_idle_disconnected_devices(self):
        ttl = ws_mod.settings.DEVICE_IDLE_TTL_SEC
        now = 10_000.0
        for dev in ("idle", "recent", "connected"):
            ws_mod.device_kalman_filters[dev] = IMUKalmanFilter()
        ws_mod._touch_device("idle", now - ttl - 1)
        ws_mod._touch_device("recent", now - ttl / 2)
        ws_mod._touch_device("connected", now - ttl * 10)
        ws_mod.active_connections["connected"] = object()
        ws_mod.queue_mgr._device_seq_counters["idle"] = 7

        self.assertEqual(ws_mod.sweep_idle_devices(now), 1)
        self.assertEqual(set(ws_mod.device_kalman_filters), {"recent", "connected"})
        self.assertNotIn("idle", ws_mod._device_last_seen)
        self.assertNotIn("idle", ws_mod.queue_mgr._device_seq_counters)

        # Once the socket closes, the long-idle device is freed on the next sweep
        ws_mod.active_connections.pop("connected")
        self.assertEqual(ws_mod.sweep_idle_devices(now), 1)
        self.assertEqual(set(ws_mod.device_kalman_filters), {"recent"})

    def test_filter_never_touched_is_freed(self):
        # e.g. created by a code path that did not record activity
        ws_mod.device_kalman_filters["orphan"] = IMUKalmanFilter()
        self.assertEqual(ws_mod.sweep_idle_devices(10_000.0), 1)
        self.assertEqual(ws_mod.device_kalman_filters, {})


if __name__ == "__main__":
    unittest.main()
