# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""The database restore path (Redis miss) recovers the step timer from the stored rows."""
import unittest
from unittest import mock

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.api import websocket
from app.db import models  # noqa: F401  (registers the tables on Base)
from app.db.crud import save_trajectory_point
from app.db.database import Base
from app.fusion.kalman_filter import IMUKalmanFilter


class TestDbRestoreStepTime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
        self.patches = [
            mock.patch.object(websocket, "AsyncSessionLocal", self.session_factory),
            # Force a Redis/memory cache miss so restore falls through to the database
            mock.patch.object(websocket.cache_manager, "get_live_device_state", mock.AsyncMock(return_value=None)),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in self.patches:
            p.stop()
        await self.engine.dispose()

    async def _store(self, device_id, seq, ts, fused, is_verified=True):
        async with self.session_factory() as session:
            await save_trajectory_point(session, {
                "device_id": device_id, "seq": seq, "timestamp": ts,
                **{k: fused[k] for k in ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
                                         "movement_state", "step_count")},
                "is_backlog": False, "is_verified": is_verified,
            })

    async def test_restore_recovers_last_step_time(self):
        dev = "db_step_dev"
        kf = IMUKalmanFilter()
        t0 = 1_700_000_000.0
        n = 0
        # Walk ("moving" mode steps on a 0.55 s cadence), then a few samples without a new step
        for i in range(32):
            ts = t0 + 0.1 * i
            fused = kf.process_sample({"timestamp": ts, "mode": "moving", "ax": 0.0, "ay": 0.0, "az": 9.81,
                                       "gx": 0.0, "gy": 0.0, "gz": 0.0})
            await self._store(dev, i, ts, fused)
            n = i
        self.assertGreater(kf.step_count, 3)
        self.assertLess(kf._last_step_time, kf.last_timestamp)
        # An unfused row with the same count but an older time must not be taken as the step
        await self._store(dev, n + 1, t0 - 5.0, {**fused, "step_count": kf.step_count}, is_verified=False)

        restored = await websocket.restore_or_init_filter(dev)
        self.assertEqual(restored.step_count, kf.step_count)
        self.assertAlmostEqual(restored.last_timestamp, kf.last_timestamp, places=3)
        self.assertAlmostEqual(restored._last_step_time, kf._last_step_time, places=3)

    async def test_no_steps_keeps_default(self):
        dev = "db_nostep_dev"
        kf = IMUKalmanFilter()
        fused = kf.process_sample({"timestamp": 1_700_000_000.0, "mode": "stationary",
                                   "ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0})
        await self._store(dev, 1, 1_700_000_000.0, fused)
        restored = await websocket.restore_or_init_filter(dev)
        self.assertEqual(restored.step_count, 0)
        self.assertEqual(restored._last_step_time, 0.0)


if __name__ == "__main__":
    unittest.main()
