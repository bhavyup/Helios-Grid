"""API route modules for Helios-Grid backend."""

from app.api.routes.health import router as health_router
from app.api.routes.simulation import router as simulation_router
from app.api.routes.training import router as training_router

__all__ = ["health_router", "simulation_router", "training_router"]