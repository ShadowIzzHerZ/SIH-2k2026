# 10Hz Dead Reckoning Backend Engine - SIH 2026
import tempfile
import unittest

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.database import Base
from app.db import models  # noqa: F401
from app.db.crud import save_trajectory_point, get_trajectory_history


class TestTrajectoryHistory(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self._tmp.name}/hist.db")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.Session = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self._tmp.cleanup()

    async def test_limit_returns_latest_points_in_order(self):
        async with self.Session() as s:
            for i in range(10):
                await save_trajectory_point(s, {"device_id": "h1", "seq": i, "timestamp": 1767225600.0 + i})
            pts = await get_trajectory_history(s, "h1", limit=3)
        self.assertEqual([p.seq_num for p in pts], [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
