# 10Hz Dead Reckoning Backend Engine - SIH 2026
"""The in-memory cache fallback frees expired keys even if they are never read."""
import os
import unittest
from unittest import mock

os.environ.setdefault("SECRET_KEY", "test-only-secret-key")

from app.cache import redis_client
from app.cache.redis_client import InMemoryCacheFallback


class TestMemoryCacheExpiry(unittest.IsolatedAsyncioTestCase):
    async def test_unread_expired_keys_are_purged_on_later_writes(self):
        now = [1_000.0]
        with mock.patch.object(redis_client.time, "time", lambda: now[0]):
            cache = InMemoryCacheFallback()
            for i in range(100):
                await cache.set(f"device:state:dev{i}", "{}", ex=10)
            await cache.set("calib:keep", "{}", ex=10_000)
            await cache.set("no_ttl", "{}")

            now[0] += 120  # past the TTL and the purge interval
            await cache.set("device:state:active", "{}", ex=10)

            self.assertEqual(set(cache._store), {"calib:keep", "no_ttl", "device:state:active"})
            self.assertEqual(await cache.get("calib:keep"), "{}")


if __name__ == "__main__":
    unittest.main()
