"""Simulation lifecycle routes for dashboard and evaluation workflows."""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import require_active_user
from app.core.settings import settings
from app.infrastructure.rate_limiter import limiter
from app.services.simulation_service import simulation_service

router = APIRouter(
    prefix="/simulation",
    tags=["simulation"],
    dependencies=[Depends(require_active_user)],
)


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0)
    num_households: int | None = Field(default=None, ge=1, le=256)
    max_episode_steps: int | None = Field(default=None, ge=1, le=100_000)
    weather_data_path: str | None = Field(default=None)

    household_data_path: str | None = Field(default=None)
    market_data_path: str | None = Field(default=None)


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
    irradiance_column: str | None = Field(default=None)
    ghi_column: str | None = Field(default=None)
    dni_column: str | None = Field(default=None)
    dhi_column: str | None = Field(default=None)
    pv_power_column: str | None = Field(default=None)
    output_path: str | None = Field(default=None)
    normalize_signals: bool = True
    panel_tilt: float | None = Field(default=None)
    panel_azimuth: float | None = Field(default=None)
    panel_area: float | None = Field(default=None)
    panel_efficiency: float | None = Field(default=None)
    temp_coefficient: float | None = Field(default=None)


class DeriveHouseholdRequest(BaseModel):
    file_path: str = Field(min_length=1)
    consumption_column: str = Field(min_length=1)
    timestamp_column: str | None = None
    household_id_column: str | None = None
    pv_generation_column: str | None = None
    net_load_column: str | None = None
    output_path: str | None = None
    normalize_signals: bool = False


class DeriveMarketRequest(BaseModel):
    file_path: str = Field(min_length=1)
    price_column: str = Field(min_length=1)

    supply_column: str | None = Field(default=None)
    demand_column: str | None = Field(default=None)

    timestamp_column: str | None = Field(default=None)
    bid_column: str | None = Field(default=None)
    ask_column: str | None = Field(default=None)
    clearing_price_column: str | None = Field(default=None)
    output_path: str | None = Field(default=None)
    normalize_signals: bool = False


class UploadWeatherResponse(BaseModel):
    """Response body for storing an uploaded weather or source CSV."""

    file_path: str
    resolved_path: str
    rows: int
    columns: list[str]
    usage_recommendation: str


class CsvPathOption(BaseModel):
    """A backend CSV file that can be reused in the dashboard."""

    path: str
    kind: str
    label: str


class CsvPathsResponse(BaseModel):
    """Response body for reusable CSV path suggestions."""

    paths: list[CsvPathOption]
    count: int


@limiter.limit(settings.rate_limit_simulation)
@router.post("/reset")
def reset_simulation(request: Request, payload: ResetRequest) -> dict[str, Any]:
    """Reset the in-memory episode and return initial state."""
    try:
        return simulation_service.reset(
            seed=payload.seed,
            num_households=payload.num_households,
            max_episode_steps=payload.max_episode_steps,
            weather_data_path=payload.weather_data_path,
            household_data_path=payload.household_data_path,
            market_data_path=payload.market_data_path,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@limiter.limit(settings.rate_limit_simulation)
@router.post("/step")
def step_simulation(request: Request, payload: StepRequest) -> dict[str, Any]:
    """Advance one timestep and return updated state and diagnostics."""
    return simulation_service.step(
        house_actions=payload.house_actions,
        market_action=payload.market_action,
        use_autopilot=payload.use_autopilot,
    )


@limiter.limit(settings.rate_limit_simulation)
@router.post("/run")
def run_simulation(request: Request, payload: RunRequest) -> dict[str, Any]:
    """Execute multiple steps and return trajectory with summary metrics."""
    return simulation_service.run(
        steps=payload.steps,
        market_action=payload.market_action,
        use_autopilot=payload.use_autopilot,
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


@router.get("/data/paths", response_model=CsvPathsResponse)
def get_csv_paths() -> dict[str, Any]:
    """Return backend CSV paths that the dashboard can surface as suggestions."""
    return simulation_service.list_available_csv_paths()


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
            irradiance_column=request.irradiance_column,
            ghi_column=request.ghi_column,
            dni_column=request.dni_column,
            dhi_column=request.dhi_column,
            pv_power_column=request.pv_power_column,
            output_path=request.output_path,
            normalize_signals=request.normalize_signals,
            panel_tilt=request.panel_tilt,
            panel_azimuth=request.panel_azimuth,
            panel_area=request.panel_area,
            panel_efficiency=request.panel_efficiency,
            temp_coefficient=request.temp_coefficient,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/derive-household")
def derive_household_csv(request: DeriveHouseholdRequest) -> dict[str, Any]:
    try:
        return simulation_service.derive_household_csv(**request.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/derive-market")
def derive_market_csv(request: DeriveMarketRequest) -> dict[str, Any]:
    try:
        return simulation_service.derive_market_csv(**request.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/upload-weather", response_model=UploadWeatherResponse)
async def upload_weather_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """Store an uploaded CSV so it can be used as the simulation weather source."""
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Uploaded file must have a filename"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        payload = await file.read()
        return simulation_service.store_uploaded_weather_csv(
            file_name=file.filename,
            file_bytes=payload,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/upload-household", response_model=UploadWeatherResponse)
async def upload_household_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Uploaded file must have a filename"
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        payload = await file.read()
        return simulation_service.store_uploaded_household_csv(
            file_name=file.filename,
            file_bytes=payload,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/data/upload-market", response_model=UploadWeatherResponse)
async def upload_market_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Uploaded file must have a filename"
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        payload = await file.read()
        return simulation_service.store_uploaded_market_csv(
            file_name=file.filename,
            file_bytes=payload,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
