from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.core.settings import settings
from app.infrastructure.redis_client import redis_client

logger = logging.getLogger(__name__)


def publish_simulation_event(event_type: str, payload: Dict[str, Any]) -> None:
    if not settings.simulation_ws_enabled:
        return

    message = {
        "type": event_type,
        "payload": payload,
    }

    try:
        redis_client.publish(settings.simulation_pubsub_channel, json.dumps(message))
    except Exception:
        logger.exception("Failed to publish simulation event %s", event_type)
