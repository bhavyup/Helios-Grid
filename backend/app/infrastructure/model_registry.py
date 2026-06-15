from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict

import torch

from app.core.settings import settings
from app.infrastructure.s3_client import create_s3_client

logger = logging.getLogger(__name__)


def registry_enabled() -> bool:
    return bool(settings.model_registry_enabled and settings.s3_bucket)


def store_training_artifacts(
    run_id: str,
    created_at: str,
    params: Dict[str, Any],
    training_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    model: torch.nn.Module,
) -> Dict[str, str] | None:
    if not registry_enabled():
        return None

    client = create_s3_client()
    bucket = settings.s3_bucket
    base_prefix = _normalize_prefix(settings.s3_prefix)
    run_prefix = f"{base_prefix}{run_id}/"

    checkpoint_key = f"{run_prefix}checkpoint.pt"
    metadata_key = f"{run_prefix}metadata.json"

    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "params": params,
        "training": training_summary,
        "comparison": comparison,
    }

    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        client.upload_fileobj(buffer, bucket, checkpoint_key)

        client.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=json.dumps(metadata).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:
        logger.exception("Failed to store model registry artifacts")
        return None

    return {
        "checkpoint_uri": f"s3://{bucket}/{checkpoint_key}",
        "metadata_uri": f"s3://{bucket}/{metadata_key}",
    }


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    cleaned = prefix.strip("/")
    return f"{cleaned}/"


__all__ = ["registry_enabled", "store_training_artifacts"]
