# 10Hz Dead Reckoning Backend Engine - SIH 2026
import json
import logging
import time
from typing import Optional, Any, Dict
import redis.asyncio as aioredis
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class InMemoryCacheFallback:
    """High-speed in-memory LRU/TTL cache fallback when Redis is unreachable."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if not item:
            return None
        if item["expires_at"] and time.time() > item["expires_at"]:
            del self._store[key]
            return None
        return item["value"]

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        expires_at = time.time() + ex if ex else None
        self._store[key] = {"value": value, "expires_at": expires_at}
        return True

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

class CacheManager:
    """
    Unified Cache Manager for Dead Reckoning State and Calibration.
    Connects to Redis; gracefully degrades to memory cache if Redis is down.
    """
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._memory_fallback = InMemoryCacheFallback()
        self.is_redis_connected = False

    async def connect(self):
        try:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0
            )
            await self._redis.ping()
            self.is_redis_connected = True
            logger.info("Successfully connected to Redis cache layer.")
        except Exception as e:
            self.is_redis_connected = False
            logger.warning(f"Redis not reachable ({e}). Using high-performance in-memory cache fallback.")

    async def disconnect(self):
        if self._redis and self.is_redis_connected:
            await self._redis.aclose()
            logger.info("Disconnected from Redis.")

    async def get(self, key: str) -> Optional[str]:
        if self.is_redis_connected and self._redis:
            try:
                return await self._redis.get(key)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}. Falling back to memory.")
        return await self._memory_fallback.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        if self.is_redis_connected and self._redis:
            try:
                return await self._redis.set(key, value, ex=ex)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}. Falling back to memory.")
        return await self._memory_fallback.set(key, value, ex=ex)

    # ---------------- Entity Cache Helpers ----------------

    async def get_live_device_state(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Fetch current live state (position, velocity, orientation, covariance)."""
        key = f"device:state:{device_id}"
        val = await self.get(key)
        return json.loads(val) if val else None

    async def set_live_device_state(self, device_id: str, state: Dict[str, Any]):
        """Cache live state with 1-hour TTL."""
        key = f"device:state:{device_id}"
        await self.set(key, json.dumps(state), ex=settings.REDIS_TTL_LIVE_SEC)

    async def get_calibration(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Fetch per-device calibration constants."""
        key = f"calib:{device_id}"
        val = await self.get(key)
        return json.loads(val) if val else None

    async def set_calibration(self, device_id: str, calib_data: Dict[str, Any]):
        """Store calibration constants (TTL = 24 hours)."""
        key = f"calib:{device_id}"
        await self.set(key, json.dumps(calib_data), ex=settings.REDIS_TTL_CALIBRATION_SEC)

cache_manager = CacheManager()

