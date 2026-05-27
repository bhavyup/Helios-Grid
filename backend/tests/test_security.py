from __future__ import annotations

import os

import pytest

from app.core import security
from app.core import settings


def test_hash_and_verify_password():
    pwd = "s3cr3t"
    hashed = security.hash_password(pwd)
    assert security.verify_password(pwd, hashed)


def test_create_and_decode_tokens(monkeypatch):
    # Ensure JWT secret is set via env
    monkeypatch.setenv("JWT_SECRET_KEY", "testkey")
    monkeypatch.setattr(settings, "jwt_secret_key", None, raising=False)

    access, _ = security.create_access_token("user1", "user")
    payload = security.decode_token(access)
    assert payload["sub"] == "user1"
    assert payload["role"] == "user"

    refresh, _ = security.create_refresh_token("user1", "user", token_jti="jti-123")
    payload2 = security.decode_token(refresh)
    assert payload2["type"] == "refresh"
    assert payload2["jti"] == "jti-123"


def test_missing_jwt_secret_raises(monkeypatch):
    # Unset env and settings
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(settings, "jwt_secret_key", None, raising=False)
    # Also set secret backend to env so secret_manager won't find it otherwise
    monkeypatch.setattr(settings, "secret_backend", "env", raising=False)

    with pytest.raises(RuntimeError):
        security.create_access_token("u", "r")
