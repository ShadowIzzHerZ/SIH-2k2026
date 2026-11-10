# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""
The trajectory row written by the queue worker must carry the filter covariance,
and restoring a filter from the database (Redis miss) must recover P from it.
"""

import unittest
from unittest import mock

import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.api import websocket
from app.core.queue_manager import IngestionItem
from app.db import models  # noqa: F401  (registers the tables on Base)
from app.db.database import Base


class _StopWorker(Exception):
    pass


class TestCovariancePersistence(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.session_factory = async_sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self.patches = [
            mock.patch.object(websocket, "AsyncSessionLocal", self.session_factory),
            # Force a Redis/memory cache miss so restore falls through to the database
            mock.patch.object(websocket.cache_manager, "get_live_device_state",
                              mock.AsyncMock(return_value=None)),
            mock.patch.object(websocket.cache_manager, "set_live_device_state",
                              mock.AsyncMock()),
        ]
        for p in self.patches:
            p.start()
        websocket.device_kalman_filters.clear()

    async def asyncTearDown(self):
        for p in self.patches:
            p.stop()
        websocket.device_kalman_filters.clear()
        await self.engine.dispose()

    async def test_worker_saves_covariance_and_db_restore_recovers_p(self):
        device_id = "cov-device"
        kf = websocket.IMUKalmanFilter()
        websocket.device_kalman_filters[device_id] = kf

        packet = {"seq": 1, "timestamp": 1_700_000_000.0,
                  "ax": 0.3, "ay": 0.1, "az": 9.81, "gx": 0.01, "gy": 0.0, "gz": 0.02,
                  "mx": 25.0, "my": 5.0, "mz": 40.0}
        item = IngestionItem(priority=0, timestamp=packet["timestamp"], seq_num=1,
                             device_id=device_id, data=packet)

        persisted = []
        written = websocket.asyncio.Event()
        real_persist = websocket._persist_point

        async def capture(payload):
            persisted.append(payload)
            await real_persist(payload)
            written.set()

        # Run one queue_worker iteration: get_next_item yields the item, then stops the loop
        get_next = mock.AsyncMock(side_effect=[item, _StopWorker()])
        with mock.patch.object(websocket.queue_mgr, "get_next_item", get_next), \
             mock.patch.object(websocket.queue_mgr, "task_done"), \
             mock.patch.object(websocket, "_persist_point", capture):
            with self.assertRaises(_StopWorker):
                await websocket.queue_worker()
            # Let the background persistence task finish
            await websocket.asyncio.wait_for(written.wait(), timeout=5)

        self.assertEqual(len(persisted), 1)
        expected_P = kf.P.copy()
        self.assertEqual(np.shape(persisted[0]["covariance"]), (9, 9))

        # Simulate the in-memory filter and Redis state being gone
        websocket.device_kalman_filters.clear()
        restored = await websocket.restore_or_init_filter(device_id)

        np.testing.assert_allclose(restored.P, expected_P)
        # And it is not just the default prior
        self.assertFalse(np.allclose(restored.P, np.eye(9) * 0.1))


if __name__ == "__main__":
    unittest.main()
