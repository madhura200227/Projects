"""
utils/cache.py – Async Redis cache with TTL, serialisation, and key namespacing.
"""

import json
import logging
from typing import Any, Optional
from datetime import datetime

import aioredis
from utils.config import settings

logger = logging.getLogger("infraguard.cache")

_redis: Optional[aioredis.Redis] = None


async def init_redis() -> aioredis.Redis:
    global _redis
    try:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        await _redis.ping()
        logger.info(f"✅ Redis connected: {settings.REDIS_URL}")
        return _redis
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}). Using in-memory fallback.")
        return InMemoryFallback()


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await init_redis()
    return _redis


class InMemoryFallback:
    """Simple in-memory dict cache when Redis is unavailable."""

    def __init__(self):
        self._store: dict = {}
        self._expiry: dict = {}

    async def ping(self): return True
    async def close(self): pass

    async def get(self, key: str) -> Optional[str]:
        import time
        exp = self._expiry.get(key)
        if exp and time.time() > exp:
            self._store.pop(key, None)
            return None
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int = 300) -> bool:
        import time
        self._store[key] = value
        self._expiry[key] = time.time() + ex
        return True

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) else 0

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def keys(self, pattern: str = "*") -> list:
        return list(self._store.keys())


class Cache:
    """Typed cache wrapper with JSON serialisation."""

    def __init__(self, namespace: str, ttl: int = 300):
        self.namespace = namespace
        self.ttl = ttl

    def _key(self, key: str) -> str:
        return f"infraguard:{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        try:
            r = await get_redis()
            raw = await r.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"Cache get error [{self.namespace}:{key}]: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            r = await get_redis()
            serialised = json.dumps(value, default=str)
            await r.set(self._key(key), serialised, ex=ttl or self.ttl)
            return True
        except Exception as e:
            logger.debug(f"Cache set error [{self.namespace}:{key}]: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            r = await get_redis()
            return await r.delete(self._key(key)) > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        try:
            r = await get_redis()
            return await r.exists(self._key(key)) > 0
        except Exception:
            return False

    async def set_with_metadata(self, key: str, value: Any, source: str, ttl: Optional[int] = None) -> bool:
        return await self.set(key, {
            "data": value,
            "source": source,
            "cached_at": datetime.utcnow().isoformat(),
            "ttl_seconds": ttl or self.ttl,
        }, ttl)

    async def get_data(self, key: str) -> Optional[Any]:
        result = await self.get(key)
        if result and isinstance(result, dict) and "data" in result:
            return result["data"]
        return result


# ── Named caches ─────────────────────────────────────────────────────────────
traffic_cache    = Cache("traffic",    ttl=settings.REDIS_TTL_TRAFFIC)
aqi_cache        = Cache("aqi",        ttl=settings.REDIS_TTL_AQI)
weather_cache    = Cache("weather",    ttl=settings.REDIS_TTL_WEATHER)
flood_cache      = Cache("flood",      ttl=settings.REDIS_TTL_FLOOD)
prediction_cache = Cache("prediction", ttl=settings.REDIS_TTL_PREDICTION)
scrape_cache     = Cache("scrape",     ttl=settings.REDIS_TTL_SCRAPE)
alert_cache      = Cache("alerts",     ttl=3600)
