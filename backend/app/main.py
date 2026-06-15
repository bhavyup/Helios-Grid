from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.simulation import router as simulation_router
from app.api.routes.simulation_ws import router as simulation_ws_router
from app.api.routes.training import router as training_router
from app.core.settings import settings
from app.infrastructure.rate_limiter import limiter
from app.infrastructure.logging_setup import configure_logging, request_logging_middleware
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware


configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.parsed_cors_allow_methods,
        allow_headers=settings.parsed_cors_allow_headers,
    )

    # Attach rate limiter
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def monitoring_middleware(request, call_next):
        return await request_logging_middleware(request, call_next)

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(simulation_router)
    app.include_router(simulation_ws_router)
    app.include_router(training_router)

    return app


app = create_app()