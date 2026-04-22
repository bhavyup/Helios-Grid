from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.simulation import router as simulation_router
from app.api.routes.training import router as training_router
from app.core.config import settings


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

    app.include_router(health_router)
    app.include_router(simulation_router)
    app.include_router(training_router)

    return app


app = create_app()