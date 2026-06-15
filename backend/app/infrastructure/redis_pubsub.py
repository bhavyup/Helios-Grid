from __future__ import annotations

import redis.asyncio as redis

from app.core.settings import settings


class _NullPubSub:
    async def subscribe(self, *args, **kwargs):
        return None

    async def get_message(self, *args, **kwargs):
        return None

    async def close(self):
        return None


class _NullAsyncRedis:
    def pubsub(self):
        return _NullPubSub()

    async def close(self):
        return None


def create_async_redis() -> redis.Redis:
    if not settings.redis_url:
        return _NullAsyncRedis()  # type: ignore[return-value]
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=settings.redis_decode_responses,
    )


__all__ = ["create_async_redis"]
