"""
app/core/cache.py
──────────────────────────────────────────────
Redis cache for storing query answers.

Why we need this:
  Every GPT-4o call costs money and takes 3-8 seconds.
  If the same question is asked again within 1 hour,
  we return the saved answer instantly — no API call.

Key format:  legal_rag:{query_hash}
Value:       JSON string of the answer
TTL:         1 hour (3600 seconds)

Example:
  Question: "What is punishment for murder under BNS?"
  Hash:     "a3f8b2c1"
  Redis key: "legal_rag:a3f8b2c1"
"""
import json
from typing import Optional

import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logger import get_logger

log = get_logger(__name__)

# Connection pool — created once, reused for all requests
_pool: Optional[aioredis.ConnectionPool] = None


def _get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _pool


def _client() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=_get_pool())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.1, max=1))
async def cache_get(query_hash: str) -> Optional[dict]:
    """
    Check if this question has been answered before.
    Returns the saved answer dict, or None if not cached.

    If Redis is down, returns None gracefully
    (system continues without cache).
    """
    try:
        r = _client()
        raw = await r.get(f"legal_rag:{query_hash}")
        if raw:
            log.info("cache_hit", query_hash=query_hash)
            return json.loads(raw)
        log.info("cache_miss", query_hash=query_hash)
        return None
    except Exception as exc:
        log.warning("cache_get_failed", error=str(exc))
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.1, max=1))
async def cache_set(query_hash: str, data: dict) -> None:
    """
    Save an answer to Redis for 1 hour.
    Next time the same question is asked → instant answer.
    """
    try:
        settings = get_settings()
        r = _client()
        await r.setex(
            f"legal_rag:{query_hash}",
            settings.redis_ttl_seconds,
            json.dumps(data),
        )
        log.info("cache_set", query_hash=query_hash, ttl=settings.redis_ttl_seconds)
    except Exception as exc:
        log.warning("cache_set_failed", error=str(exc))


async def cache_health() -> bool:
    """
    Ping Redis to check if it is running.
    Used by the /health endpoint.
    Returns True if healthy, False if down.
    """
    try:
        r = _client()
        return await r.ping()
    except Exception:
        return False