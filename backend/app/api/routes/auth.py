"""Authentication routes for JWT access and refresh tokens."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import require_active_user, require_roles
from app.core.settings import settings
from app.infrastructure.database import get_db
from app.infrastructure.rate_limiter import limiter
from app.repositories.db_models import User
from app.services.auth_service import (
    TokenPair,
    authenticate_user,
    create_user,
    issue_token_pair,
    refresh_token_pair,
    revoke_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int
    user: UserResponse


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=int(user.id),
        email=str(user.email),
        role=str(user.role),
        is_active=bool(user.is_active),
    )


def _build_token_response(token_pair: TokenPair) -> TokenResponse:
    now = datetime.now(tz=UTC)
    access_seconds = max(0, int((token_pair.access_expires_at - now).total_seconds()))
    refresh_seconds = max(
        0,
        int((token_pair.refresh_expires_at - now).total_seconds()),
    )
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=access_seconds,
        refresh_expires_in=refresh_seconds,
        user=_to_user_response(token_pair.user),
    )


@router.post("/register", response_model=TokenResponse)
@limiter.limit(settings.effective_rate_limit_auth)
def register_user(
    request: Request,
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    try:
        user = create_user(db, email=payload.email, password=payload.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    token_pair = issue_token_pair(db, user)
    return _build_token_response(token_pair)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.effective_rate_limit_auth)
def login_user(
    request: Request,
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = authenticate_user(db, email=payload.email, password=payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive",
        )

    token_pair = issue_token_pair(db, user)
    return _build_token_response(token_pair)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(settings.effective_rate_limit_auth)
def refresh_tokens(
    request: Request,
    payload: RefreshRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    try:
        token_pair = refresh_token_pair(db, refresh_token=payload.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    return _build_token_response(token_pair)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(settings.effective_rate_limit_auth)
def logout_user(
    request: Request,
    payload: LogoutRequest,
    db: Session = Depends(get_db),
) -> None:
    try:
        revoke_refresh_token(db, refresh_token=payload.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    return None


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(require_active_user)) -> UserResponse:
    return _to_user_response(current_user)


@router.get("/users", response_model=list[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _current_user: User | None = Depends(require_roles("admin")),
) -> list[UserResponse]:
    users = db.query(User).order_by(User.id.asc()).all()
    return [_to_user_response(user) for user in users]
