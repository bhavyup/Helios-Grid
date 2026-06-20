"""WebSocket stream for real-time simulation updates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.settings import settings
from app.infrastructure.redis_pubsub import create_async_redis

router = APIRouter(prefix="/ws", tags=["simulation"])


@router.websocket("/simulation")
async def simulation_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    redis_client = create_async_redis()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(settings.simulation_pubsub_channel)

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=1.0,
            )
            if message is None:
                await asyncio.sleep(0.1)
                continue

            data: Any = message.get("data")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")

            if isinstance(data, str):
                await websocket.send_text(data)
            else:
                await websocket.send_text(
                    json.dumps({"type": "unknown", "payload": data})
                )
    except WebSocketDisconnect:
        return
    finally:
        await pubsub.close()
        await redis_client.close()
