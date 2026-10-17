# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""Background DB writes are tracked and drained on shutdown instead of being dropped."""

import asyncio
import unittest
from unittest import mock

from app.api import websocket as ws


class TestPersistTaskTracking(unittest.IsolatedAsyncioTestCase):
    async def test_worker_writes_are_tracked_and_drained(self):
        written = []

        async def slow_persist(payload):
            await asyncio.sleep(0.2)
            written.append(payload["seq"])

        with mock.patch.object(ws, "_persist_point", slow_persist), \
             mock.patch.object(ws, "queue_mgr", ws.IngestionQueueManager(maxsize=10)):
            await ws.queue_mgr.enqueue_live("persist_dev", {
                "seq": 7, "timestamp": 1000.0, "mode": "stationary",
                "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0,
            })
            worker = asyncio.create_task(ws.queue_worker())
            for _ in range(100):
                if ws._pending_writes:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(ws._pending_writes), 1)

            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker
            await ws.drain_pending_writes()

        self.assertEqual(written, [7])
        self.assertEqual(len(ws._pending_writes), 0)
        ws.device_kalman_filters.pop("persist_dev", None)


if __name__ == "__main__":
    unittest.main()
