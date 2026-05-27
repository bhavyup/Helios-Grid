from __future__ import annotations

import os
import logging
from typing import Optional

import requests

from app.core.settings import settings

logger = logging.getLogger(__name__)


def _from_env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _from_vault(name: str) -> Optional[str]:
    # Minimal Vault KV v2 support. Requires settings.vault_addr and settings.vault_token.
    addr = settings.vault_addr or os.environ.get("VAULT_ADDR")
    token = settings.vault_token or os.environ.get("VAULT_TOKEN")
    if not addr or not token:
        return None
    try:
        # Assumes secrets stored at path: secret/data/<name>
        url = f"{addr.rstrip('/')}/v1/secret/data/{name}"
        headers = {"X-Vault-Token": token}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            logger.debug("Vault returned %s for %s", resp.status_code, name)
            return None
        data = resp.json()
        return data.get("data", {}).get("data", {}).get("value")
    except Exception:
        logger.exception("Vault lookup failed for %s", name)
        return None


def get_secret(name: str, fallback: Optional[str] = None) -> Optional[str]:
    # 1) Environment
    val = _from_env(name)
    if val:
        return val

    # 2) Secret backend
    backend = (settings.secret_backend or "env").lower()
    if backend == "vault":
        val = _from_vault(name)
        if val:
            return val

    # 3) Doppler or other backends could be added here
    # For now, Doppler usage should inject env vars into the process.

    return fallback


__all__ = ["get_secret"]
