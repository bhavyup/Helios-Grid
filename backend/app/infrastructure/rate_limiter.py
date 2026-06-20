from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.settings import settings

# Use Redis storage when available for distributed rate-limits; fall back to in-memory.
storage = settings.redis_url if settings.redis_url else None
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=storage,
    default_limits=[settings.effective_rate_limit_default],
)

__all__ = ["limiter"]
