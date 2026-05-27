from __future__ import annotations

import os
from types import SimpleNamespace

import requests

from app.core import secret_manager
from app.core import settings


def test_get_secret_from_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "envsecret")
    # Ensure env backend
    monkeypatch.setattr(settings, "secret_backend", "env", raising=False)
    assert secret_manager.get_secret("JWT_SECRET_KEY") == "envsecret"


def test_get_secret_from_vault(monkeypatch):
    # Ensure no env var
    monkeypatch.delenv("MY_SECRET", raising=False)
    # Configure settings to use vault
    monkeypatch.setattr(settings, "secret_backend", "vault", raising=False)
    monkeypatch.setattr(settings, "vault_addr", "https://vault.example.com", raising=False)
    monkeypatch.setattr(settings, "vault_token", "token", raising=False)

    class DummyResp:
        status_code = 200

        def json(self):
            return {"data": {"data": {"value": "vaultsecret"}}}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: DummyResp())

    assert secret_manager.get_secret("MY_SECRET") == "vaultsecret"
