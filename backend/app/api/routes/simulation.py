"""Simulation lifecycle routes for dashboard and evaluation workflows."""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.simulation_service import simulation_service

router = APIRouter(prefix="/simulation", tags=["simulation"])


class ResetRequest(BaseModel):
    """Request body for starting a fresh simulation episode."""

    seed: int | None = Field(default=None, ge=0)
    num_households: int | None = Field(default=None, ge=1, le=256)
    max_episode_steps: int | None = Field(default=None, ge=1, le=100_000)
    weather_data_path: str | None = Field(default=None)


class StepRequest(BaseModel):
    """Request body for a single simulation step."""

    house_actions: List[List[float]] | None = None
    market_action: int | None = Field(default=None, ge=0, le=1)
    use_autopilot: bool = True


class RunRequest(BaseModel):
    """Request body for multi-step execution."""

    steps: int = Field(default=24, ge=1, le=5_000)
    market_action: int | None = Field(default=None, ge=0, le=1)
    use_autopilot: bool = True


class CsvProfileRequest(BaseModel):
    """Request body for CSV inspection and schema compatibility profiling."""

    file_path: str = Field(min_length=1)
    role: str = Field(default="auto")
    preview_rows: int = Field(default=5, ge=1, le=20)


class DeriveWeatherRequest(BaseModel):
    """Request body for deriving GridEnv weather CSV from a source timeseries CSV."""

    file_path: str = Field(min_length=1)
    solar_column: str = Field(min_length=1)
    wind_column: str = Field(min_length=1)
    timestamp_column: str | None = Field(default=None)
    temperature_column: str | None = Field(default=None)
    humidity_column: str | None = Field(default=None)
    output_path: str | None = Field(default=None)
    normalize_signals: bool = True


@router.post("/reset")
def reset_simulation(request: ResetRequest) -> dict[str, Any]:
    """Reset the in-memory episode and return initial state."""
    try:
        return simulation_service.reset(
            seed=request.seed,
            num_households=request.num_households,
            max_episode_steps=request.max_episode_steps,
            weather_data_path=request.weather_data_path,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/step")
def step_simulation(request: StepRequest) -> dict[str, Any]:
    """Advance one timestep and return updated state and diagnostics."""
    return simulation_service.step(
        house_actions=request.house_actions,
        market_action=request.market_action,
        use_autopilot=request.use_autopilot,
    )


@router.post("/run")
def run_simulation(request: RunRequest) -> dict[str, Any]:
    """Execute multiple steps and return trajectory with summary metrics."""
    return simulation_service.run(
        steps=request.steps,
        market_action=request.market_action,
        use_autopilot=request.use_autopilot,
    )


@router.get("/state")
def get_simulation_state(include_topology: bool = True) -> dict[str, Any]:
    """Return the latest state snapshot."""
    return simulation_service.get_state(include_topology=include_topology)


@router.get("/metrics")
def get_simulation_metrics() -> dict[str, Any]:
    """Return episode aggregate metrics."""
    return simulation_service.get_metrics()


@router.get("/history")
def get_simulation_history(
    limit: int = Query(default=300, ge=1, le=5_000),
) -> list[dict[str, Any]]:
    """Return latest trajectory points for chart rendering."""
    return simulation_service.get_history(limit=limit)


@router.get("/data/schemas")
def get_csv_schemas() -> dict[str, Any]:
    """Return CSV schema guidance and runtime support metadata."""
    return simulation_service.get_csv_schemas()


@router.post("/data/profile")
def profile_csv_data(request: CsvProfileRequest) -> dict[str, Any]:
    """Inspect a CSV file and return role compatibility and usage guidance."""
    try:
        return simulation_service.profile_csv(
            file_path=request.file_path,
            role=request.role,
            preview_rows=request.preview_rows,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/derive-weather")
def derive_weather_csv(request: DeriveWeatherRequest) -> dict[str, Any]:
    """Create a weather-compatible CSV for GridEnv from source CSV columns."""
    try:
        return simulation_service.derive_weather_csv(
            file_path=request.file_path,
            solar_column=request.solar_column,
            wind_column=request.wind_column,
            timestamp_column=request.timestamp_column,
            temperature_column=request.temperature_column,
            humidity_column=request.humidity_column,
            output_path=request.output_path,
            normalize_signals=request.normalize_signals,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
