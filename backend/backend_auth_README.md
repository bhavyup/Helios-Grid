This commit adds JWT authentication with access and refresh tokens,
including RBAC role checks and protected routes.  Key points:

- `app/core/security.py`: JWT helpers and password hashing.
- `app/services/auth_service.py`: user CRUD, token issuance and refresh.
- `app/api/routes/auth.py`: login/register/refresh/logout endpoints.
- `app/api/deps.py`: FastAPI dependencies to require authentication and roles.
- `app/repositories/db_models.py`: added RefreshToken model.
- Alembic migration: `alembic/versions/0002_add_refresh_tokens.py`.

To enable: set `AUTH_ENABLED=true` and configure `JWT_SECRET_KEY`.

Next steps: integrate email/password validation UI, secure token storage on client.
