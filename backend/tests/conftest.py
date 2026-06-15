import os
import sys
from types import ModuleType

os.environ.setdefault("AUTH_ENABLED", "true")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///./test_auth.db")


def _install_ray_stub() -> None:
    if "ray" in sys.modules:
        return

    ray_stub = ModuleType("ray")
    ray_state = {"initialized": False}

    class _RemoteFunction:
        def __init__(self, func):
            self._func = func
            self.__wrapped__ = func

        def remote(self, *args, **kwargs):
            return self._func(*args, **kwargs)

        def options(self, **kwargs):
            return self

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    def remote(func=None, **kwargs):
        if func is None:
            return lambda actual: _RemoteFunction(actual)
        return _RemoteFunction(func)

    def is_initialized():
        return ray_state["initialized"]

    def init(**kwargs):
        ray_state["initialized"] = True
        return {"initialized": True, **kwargs}

    def get(value):
        return value

    def wait(values, timeout=None):
        return (list(values), [])

    ray_stub.remote = remote
    ray_stub.is_initialized = is_initialized
    ray_stub.init = init
    ray_stub.get = get
    ray_stub.wait = wait
    sys.modules["ray"] = ray_stub


_install_ray_stub()

from fastapi.testclient import TestClient
import pytest

from app.infrastructure.database import Base, SessionLocal, engine
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def auth_headers(client):
    payload = {"email": "test@example.com", "password": "test-pass-123"}
    register = client.post("/auth/register", json=payload)
    if register.status_code not in (200, 409):
        raise AssertionError(f"Failed to register test user: {register.status_code}")

    login = client.post("/auth/login", json=payload)
    if login.status_code != 200:
        raise AssertionError(f"Failed to login test user: {login.status_code}")

    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
