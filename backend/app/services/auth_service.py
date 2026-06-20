from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.security import (
    JWTError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.settings import settings
from app.repositories.db_models import RefreshToken, User


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    user: User


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def get_user_by_email(db: Session, email: str) -> User | None:
    normalized = normalize_email(email)
    return db.query(User).filter(User.email == normalized).one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).one_or_none()


def create_user(db: Session, email: str, password: str, role: str = "user") -> User:
    normalized = normalize_email(email)
    if get_user_by_email(db, normalized) is not None:
        raise ValueError("User already exists")

    user = User(
        email=normalized,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def issue_token_pair(db: Session, user: User) -> TokenPair:
    access_token, access_expires_at = create_access_token(str(user.id), user.role)

    refresh_jti = str(uuid4())
    refresh_token, refresh_expires_at = create_refresh_token(
        str(user.id),
        user.role,
        refresh_jti,
    )

    refresh_record = RefreshToken(
        user_id=user.id,
        token_jti=refresh_jti,
        token_hash=hash_refresh_token(refresh_token),
        expires_at=refresh_expires_at,
    )
    db.add(refresh_record)
    db.commit()

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=refresh_expires_at,
        user=user,
    )


def refresh_token_pair(db: Session, refresh_token: str) -> TokenPair:
    try:
        payload = decode_token(refresh_token)
    except JWTError as exc:
        raise ValueError("Invalid refresh token") from exc

    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")

    token_jti = payload.get("jti")
    user_id = payload.get("sub")
    if not token_jti or not user_id:
        raise ValueError("Invalid refresh token")

    record = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_jti == str(token_jti))
        .one_or_none()
    )
    if record is None:
        raise ValueError("Refresh token not recognized")
    if record.revoked_at is not None:
        raise ValueError("Refresh token has been revoked")
    if _normalize_utc(record.expires_at) <= _normalize_utc(datetime.now(tz=UTC)):
        raise ValueError("Refresh token has expired")
    if record.token_hash != hash_refresh_token(refresh_token):
        raise ValueError("Refresh token mismatch")

    user = get_user_by_id(db, int(user_id))
    if user is None or not user.is_active:
        raise ValueError("User is inactive")

    if settings.refresh_token_rotate:
        record.revoked_at = datetime.now(tz=UTC)
        new_refresh_jti = str(uuid4())
        record.replaced_by_jti = new_refresh_jti

        new_refresh_token, new_refresh_expires_at = create_refresh_token(
            str(user.id),
            user.role,
            new_refresh_jti,
        )
        new_record = RefreshToken(
            user_id=user.id,
            token_jti=new_refresh_jti,
            token_hash=hash_refresh_token(new_refresh_token),
            expires_at=new_refresh_expires_at,
        )
        db.add(new_record)
        db.commit()

        access_token, access_expires_at = create_access_token(str(user.id), user.role)
        return TokenPair(
            access_token=access_token,
            refresh_token=new_refresh_token,
            access_expires_at=access_expires_at,
            refresh_expires_at=new_refresh_expires_at,
            user=user,
        )

    access_token, access_expires_at = create_access_token(str(user.id), user.role)
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=record.expires_at,
        user=user,
    )


def revoke_refresh_token(db: Session, refresh_token: str) -> bool:
    try:
        payload = decode_token(refresh_token)
    except JWTError as exc:
        raise ValueError("Invalid refresh token") from exc

    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")

    token_jti = payload.get("jti")
    if not token_jti:
        raise ValueError("Invalid refresh token")

    record = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_jti == str(token_jti))
        .one_or_none()
    )
    if record is None:
        return False

    if record.revoked_at is None:
        record.revoked_at = datetime.now(tz=UTC)
        db.commit()
    return True
