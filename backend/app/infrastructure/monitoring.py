from __future__ import annotations

import os
import time
from typing import Any, Callable, TypeVar

import psutil
import torch
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

_FUNCTION = TypeVar("_FUNCTION", bound=Callable[..., Any])

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "helios_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route", "status"],
)
HTTP_REQUESTS_TOTAL = Counter(
    "helios_http_requests_total",
    "Total HTTP requests processed",
    ["method", "route", "status"],
)
HTTP_REQUEST_FAILURES_TOTAL = Counter(
    "helios_http_request_failures_total",
    "Total HTTP requests that failed",
    ["method", "route", "status"],
)
TRAINING_JOBS_TOTAL = Counter(
    "helios_training_jobs_total",
    "Total PPO training jobs started",
    ["result"],
)
TRAINING_RUN_DURATION_SECONDS = Histogram(
    "helios_training_run_duration_seconds",
    "PPO training run duration in seconds",
    ["result"],
)
TRAINING_REWARD_LAST = Gauge(
    "helios_training_reward_last",
    "Latest PPO training reward value",
)
TRAINING_REWARD_MEAN = Gauge(
    "helios_training_reward_mean",
    "Mean PPO reward curve value",
)
TRAINING_EPISODES = Gauge(
    "helios_training_episodes",
    "Episodes in the latest PPO training run",
)
SIMULATION_EPISODES = Gauge(
    "helios_simulation_episode_id",
    "Latest simulation episode identifier",
)
SIMULATION_STEPS = Gauge(
    "helios_simulation_steps_executed",
    "Steps executed in the latest simulation session",
)
SYSTEM_CPU_PERCENT = Gauge(
    "helios_system_cpu_percent",
    "Current system CPU utilization percentage",
)
SYSTEM_MEMORY_PERCENT = Gauge(
    "helios_system_memory_percent",
    "Current system memory utilization percentage",
)
GPU_MEMORY_USED_MB = Gauge(
    "helios_gpu_memory_used_mb",
    "Current GPU memory used in megabytes",
)
GPU_MEMORY_TOTAL_MB = Gauge(
    "helios_gpu_memory_total_mb",
    "Total GPU memory in megabytes",
)
GPU_AVAILABLE = Gauge(
    "helios_gpu_available",
    "Whether CUDA is available on this host",
)
ERRORS_TOTAL = Counter(
    "helios_errors_total",
    "Application errors recorded by subsystem",
    ["subsystem", "error_type"],
)

_process = psutil.Process(os.getpid())


def _route_name(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return str(route.path)
    return request.url.path


def _status_bucket(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    return "5xx"


def observe_request(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    status = _status_bucket(status_code)
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route, status=status).observe(duration_seconds)
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status=status).inc()
    if status_code >= 400:
        HTTP_REQUEST_FAILURES_TOTAL.labels(method=method, route=route, status=status).inc()


def record_error(subsystem: str, error_type: str) -> None:
    ERRORS_TOTAL.labels(subsystem=subsystem, error_type=error_type).inc()


def record_training_job_started() -> None:
    TRAINING_JOBS_TOTAL.labels(result="started").inc()


def record_training_job_completed(training_summary: dict[str, Any]) -> None:
    TRAINING_JOBS_TOTAL.labels(result="completed").inc()
    reward_curve = training_summary.get("reward_curve") or []
    if reward_curve:
        TRAINING_REWARD_LAST.set(float(reward_curve[-1]))
        TRAINING_REWARD_MEAN.set(float(sum(reward_curve) / len(reward_curve)))
    episodes = training_summary.get("episodes")
    if episodes is not None:
        TRAINING_EPISODES.set(float(episodes))


def record_training_job_failed() -> None:
    TRAINING_JOBS_TOTAL.labels(result="failed").inc()


def record_training_duration(duration_seconds: float, result: str) -> None:
    TRAINING_RUN_DURATION_SECONDS.labels(result=result).observe(duration_seconds)


def record_simulation_state(state_payload: dict[str, Any]) -> None:
    episode_id = state_payload.get("episode_id")
    step_count = state_payload.get("step_count")
    if episode_id is not None:
        SIMULATION_EPISODES.set(float(episode_id))
    if step_count is not None:
        SIMULATION_STEPS.set(float(step_count))


def collect_runtime_metrics() -> None:
    SYSTEM_CPU_PERCENT.set(float(psutil.cpu_percent(interval=None)))
    SYSTEM_MEMORY_PERCENT.set(float(psutil.virtual_memory().percent))

    cuda_available = bool(torch.cuda.is_available())
    GPU_AVAILABLE.set(1.0 if cuda_available else 0.0)
    if cuda_available:
        total = float(torch.cuda.get_device_properties(0).total_memory) / (1024 * 1024)
        used = float(torch.cuda.memory_reserved(0)) / (1024 * 1024)
        GPU_MEMORY_USED_MB.set(used)
        GPU_MEMORY_TOTAL_MB.set(total)
    else:
        GPU_MEMORY_USED_MB.set(0.0)
        GPU_MEMORY_TOTAL_MB.set(0.0)


def metrics_response() -> Response:
    collect_runtime_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def record_http_metrics(request: Request, call_next: Callable[[Request], Any]) -> Response:
    start = time.perf_counter()
    route_name = _route_name(request)
    method = request.method

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.perf_counter() - start
        observe_request(method, route_name, 500, duration)
        record_error("http", exc.__class__.__name__)
        raise

    duration = time.perf_counter() - start
    observe_request(method, route_name, response.status_code, duration)
    return response


__all__ = [
    "collect_runtime_metrics",
    "metrics_response",
    "observe_request",
    "record_error",
    "record_http_metrics",
    "record_simulation_state",
    "record_training_duration",
    "record_training_job_completed",
    "record_training_job_failed",
    "record_training_job_started",
]
