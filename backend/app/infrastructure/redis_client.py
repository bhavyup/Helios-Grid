from __future__ import annotations

from typing import Generator

import redis
from redis import Redis

from app.core.settings import settings


class _NullRedis:
    def publish(self, *args, **kwargs):
        return 0

    def close(self):
        return None


def create_redis_client() -> Redis:
    if not settings.redis_url:
        return _NullRedis()  # type: ignore[return-value]
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=settings.redis_decode_responses,
    )


redis_client = create_redis_client()


def get_redis() -> Generator[Redis, None, None]:
    client = create_redis_client()
    try:
        yield client
    finally:
        client.close()
