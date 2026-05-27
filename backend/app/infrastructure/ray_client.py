from __future__ import annotations

import ray

from app.core.settings import settings


def init_ray() -> None:
    if ray.is_initialized():
        return

    address = settings.ray_address.strip() if settings.ray_address else ""
    if address:
        ray.init(
            address=address,
            namespace=settings.ray_namespace,
            ignore_reinit_error=True,
        )
    else:
        ray.init(
            namespace=settings.ray_namespace,
            ignore_reinit_error=True,
        )


__all__ = ["init_ray"]
