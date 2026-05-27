from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.settings import settings
from app.core.secret_manager import get_secret

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(subject: str, role: str) -> Tuple[str, datetime]:
    now = datetime.now(tz=timezone.utc)
    expires_at = now + timedelta(minutes=settings.access_token_expires_minutes)
    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": expires_at,
    }
    secret = settings.jwt_secret_key or get_secret("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT secret key not configured. Set JWT_SECRET_KEY in env or secret backend.")
    token = jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)
    return token, expires_at


def create_refresh_token(subject: str, role: str, token_jti: str) -> Tuple[str, datetime]:
    now = datetime.now(tz=timezone.utc)
    expires_at = now + timedelta(days=settings.refresh_token_expires_days)
    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": "refresh",
        "jti": token_jti,
        "iat": int(now.timestamp()),
        "exp": expires_at,
    }
    secret = settings.jwt_secret_key or get_secret("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT secret key not configured. Set JWT_SECRET_KEY in env or secret backend.")
    token = jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)
    return token, expires_at


def decode_token(token: str) -> Dict[str, Any]:
    secret = settings.jwt_secret_key or get_secret("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT secret key not configured. Set JWT_SECRET_KEY in env or secret backend.")
    return jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])


__all__ = [
    "JWTError",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
