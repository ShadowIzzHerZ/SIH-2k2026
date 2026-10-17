# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""/health reports live Redis reachability instead of the startup-time flag."""
import unittest
from unittest import mock

import httpx

from app.main import app
from app.cache.redis_client import cache_manager


class _FakeRedis:
    def __init__(self, up: bool):
        self.up = up

    async def ping(self):
        if not self.up:
            raise ConnectionError("redis down")
        return True


class TestHealthRedisPing(unittest.IsolatedAsyncioTestCase):
    async def _health(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/health")
            self.assertEqual(res.status_code, 200)
            return res.json()

    async def test_health_follows_redis_state(self):
        fake = _FakeRedis(up=True)
        with mock.patch.object(cache_manager, "_redis", fake), \
             mock.patch.object(cache_manager, "is_redis_connected", True):
            self.assertTrue((await self._health())["redis_connected"])
            fake.up = False  # Redis goes away after startup
            self.assertFalse((await self._health())["redis_connected"])

    async def test_health_without_redis_client(self):
        with mock.patch.object(cache_manager, "_redis", None):
            self.assertFalse((await self._health())["redis_connected"])


if __name__ == "__main__":
    unittest.main()
