"""
ForgeMind AI Suite — Redis Cache Layer
Hot agent output caching with TTL-based eviction.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisCache:
    """Async Redis cache for hot agent outputs and shared state."""

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        self._client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
        await self._client.ping()
        logger.info("Redis connected at %s", settings.redis_url)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis disconnected")

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    # ── Key-Value Operations ──

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get a cached value by key."""
        raw = await self.client.get(key)
        if raw:
            return json.loads(raw)
        return None

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> None:
        """Set a value with TTL."""
        await self.client.setex(key, ttl_seconds, json.dumps(value, default=str))

    async def delete(self, key: str) -> None:
        """Delete a cached key."""
        await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(await self.client.exists(key))

    # ── Agent-Specific Helpers ──

    async def cache_agent_output(
        self,
        agent_name: str,
        entity_id: str,
        output: dict[str, Any],
        ttl_seconds: int = 600,
    ) -> None:
        """Cache an agent's latest output for an entity (machine, sku, supplier, etc.)."""
        key = f"agent:{agent_name}:entity:{entity_id}:latest"
        await self.set(key, output, ttl_seconds)

    async def get_agent_output(
        self,
        agent_name: str,
        entity_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get the latest cached output from an agent for an entity."""
        key = f"agent:{agent_name}:entity:{entity_id}:latest"
        return await self.get(key)

    async def cache_health_score(
        self,
        machine_id: str,
        score: float,
        ttl_seconds: int = 120,
    ) -> None:
        """Cache machine health score for quick dashboard access."""
        key = f"health:machine:{machine_id}"
        await self.set(key, {"machine_id": machine_id, "score": score}, ttl_seconds)

    async def get_health_score(self, machine_id: str) -> Optional[float]:
        """Get cached machine health score."""
        data = await self.get(f"health:machine:{machine_id}")
        return data.get("score") if data else None

    # ── Pub/Sub for Real-Time Updates ──

    async def publish_event(self, channel: str, event: dict[str, Any]) -> None:
        """Publish an event via Redis pub/sub."""
        await self.client.publish(channel, json.dumps(event, default=str))

    # ── Rate Limiting ──

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """Simple sliding window rate limiter. Returns True if within limit."""
        pipe = self.client.pipeline()
        await pipe.incr(key)
        await pipe.expire(key, window_seconds)
        results = await pipe.execute()
        current_count = results[0]
        return current_count <= max_requests


# Singleton instance
redis_cache = RedisCache()
