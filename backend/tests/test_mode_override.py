# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""POST /device/{id}/mode is a lasting override until set back to 'adaptive'."""
import asyncio
import os
import unittest
from unittest import mock

os.environ.setdefault("SECRET_KEY", "test-only-secret-key")


class TestModeOverride(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from app.db.database import init_db
        from app.db import models  # noqa: F401  (register tables before create_all)
        await init_db()

    async def test_override_applies_to_later_packets_until_cleared(self):
        import httpx
        from app.main import app
        from app.api import websocket as ws_mod

        dev = "mode_override_dev"
        # A private queue: the module-level one would stay bound to this test's
        # event loop and break later tests that run the app on their own loop.
        queue_patch = mock.patch.object(ws_mod, "queue_mgr", ws_mod.IngestionQueueManager(maxsize=100))
        queue_patch.start()
        worker = asyncio.create_task(ws_mod.queue_worker())
        t = 1_000.0
        seq = 0

        async def send_moving_packets(n):
            nonlocal t, seq
            for _ in range(n):
                seq += 1
                t += 0.1
                await ws_mod.queue_mgr.enqueue_live(dev, {
                    "seq": seq, "timestamp": t, "mode": "moving",
                    "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0,
                })
            await asyncio.wait_for(ws_mod.queue_mgr._queue.join(), timeout=5)
            return ws_mod.device_kalman_filters[dev]

        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                kf = await send_moving_packets(3)
                self.assertEqual(kf.movement_state, "MOVING")

                res = await client.post(f"/device/{dev}/mode", json={"mode": "stationary"})
                self.assertEqual(res.status_code, 200)

                # Packets still say "moving", but the forced mode wins
                kf = await send_moving_packets(5)
                self.assertEqual(kf.movement_state, "REST")
                self.assertEqual(float(kf.x[3, 0]), 0.0)
                x_frozen = float(kf.x[0, 0])
                kf = await send_moving_packets(5)
                self.assertEqual(float(kf.x[0, 0]), x_frozen)

                # 'adaptive' clears the override; the packet mode applies again
                res = await client.post(f"/device/{dev}/mode", json={"mode": "adaptive"})
                self.assertEqual(res.status_code, 200)
                kf = await send_moving_packets(3)
                self.assertEqual(kf.movement_state, "MOVING")
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            queue_patch.stop()
            getattr(ws_mod, "device_mode_overrides", {}).pop(dev, None)
            ws_mod.device_kalman_filters.pop(dev, None)


if __name__ == "__main__":
    unittest.main()
