from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

from app.infrastructure.monitoring import metrics_response

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
def get_metrics() -> Response:
    return metrics_response()
