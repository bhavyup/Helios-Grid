from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from app.core.settings import settings


def create_s3_client() -> BaseClient:
    config = Config(
        s3={"addressing_style": "path" if settings.s3_force_path_style else "virtual"}
    )

    client_kwargs: dict[str, object] = {
        "service_name": "s3",
        "region_name": settings.s3_region or None,
        "use_ssl": settings.s3_use_ssl,
        "config": config,
    }

    if settings.s3_endpoint_url:
        client_kwargs["endpoint_url"] = settings.s3_endpoint_url

    if settings.s3_access_key and settings.s3_secret_key:
        client_kwargs["aws_access_key_id"] = settings.s3_access_key
        client_kwargs["aws_secret_access_key"] = settings.s3_secret_key

    return boto3.client(**client_kwargs)


__all__ = ["create_s3_client"]
