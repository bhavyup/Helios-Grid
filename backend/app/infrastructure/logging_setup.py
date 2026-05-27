from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable
from uuid import uuid4

import structlog
from fastapi import Request, Response
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars

from app.infrastructure.monitoring import record_error, observe_request


def configure_logging() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

    pre_chain = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processors=[
            merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    structlog.configure(
        processors=pre_chain + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


async def request_logging_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    clear_contextvars()
    request_id = request.headers.get("x-request-id") or uuid4().hex
    trace_id = request.headers.get("x-trace-id") or uuid4().hex
    bind_contextvars(request_id=request_id, trace_id=trace_id)

    start = time.perf_counter()
    logger = structlog.get_logger("http")

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.perf_counter() - start
        observe_request(request.method, request.url.path, 500, duration)
        record_error("http", exc.__class__.__name__)
        logger.exception(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round(duration * 1000.0, 2),
        )
        raise

    duration = time.perf_counter() - start
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Trace-Id"] = trace_id
    observe_request(request.method, request.url.path, response.status_code, duration)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000.0, 2),
    )
    return response


__all__ = ["configure_logging", "request_logging_middleware"]