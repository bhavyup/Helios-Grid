from __future__ import annotations

import json

from app.core import settings
from app.infrastructure import redis_client as redis_client_module
from app.infrastructure import redis_pubsub as redis_pubsub_module
from app.infrastructure import simulation_events


class _FakeRedis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.calls.append((channel, message))
        return 1


def test_publish_simulation_event_serializes_payload(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(simulation_events, "redis_client", fake_redis, raising=False)
    monkeypatch.setattr(settings, "simulation_ws_enabled", True, raising=False)
    monkeypatch.setattr(settings, "simulation_pubsub_channel", "tests.simulations", raising=False)

    simulation_events.publish_simulation_event("simulation.step", {"step": 1, "reward": 2.5})

    assert fake_redis.calls == [
        (
            "tests.simulations",
            json.dumps({"type": "simulation.step", "payload": {"step": 1, "reward": 2.5}}),
        )
    ]


def test_redis_client_factories_use_settings_url(monkeypatch):
    captured_sync: dict[str, object] = {}
    captured_async: dict[str, object] = {}

    class _DummyRedis:
        def __init__(self, sink: dict[str, object]) -> None:
            self.sink = sink

        def close(self) -> None:
            self.sink["closed"] = True

    def fake_sync_from_url(url, decode_responses):
        captured_sync["url"] = url
        captured_sync["decode_responses"] = decode_responses
        return _DummyRedis(captured_sync)

    def fake_async_from_url(url, decode_responses):
        captured_async["url"] = url
        captured_async["decode_responses"] = decode_responses
        return _DummyRedis(captured_async)

    monkeypatch.setattr(redis_client_module.redis.Redis, "from_url", fake_sync_from_url, raising=False)
    monkeypatch.setattr(redis_pubsub_module.redis.Redis, "from_url", fake_async_from_url, raising=False)
    monkeypatch.setattr(settings, "redis_url", "redis://localhost:6380/1", raising=False)
    monkeypatch.setattr(settings, "redis_decode_responses", False, raising=False)

    sync_client = redis_client_module.create_redis_client()
    async_client = redis_pubsub_module.create_async_redis()

    assert isinstance(sync_client, _DummyRedis)
    assert isinstance(async_client, _DummyRedis)
    assert captured_sync["url"] == "redis://localhost:6380/1"
    assert captured_async["url"] == "redis://localhost:6380/1"
    assert captured_sync["decode_responses"] is False
    assert captured_async["decode_responses"] is False
