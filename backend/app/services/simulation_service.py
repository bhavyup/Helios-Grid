"""Simulation session service for deterministic, API-driven grid runs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from app.core.project_config import config
from app.infrastructure.monitoring import record_simulation_state
from app.infrastructure.simulation_events import publish_simulation_event


@dataclass
class StepRecord:
    """Compact time-series point used by the dashboard and metrics layer."""

    step: int
    timestamp: str
    reward: float
    done: bool
    supply: float
    demand: float
    price: float
    grid_import: float
    renewable_utilization: float


class SimulationService:
    """Manages one in-memory simulation session with deterministic stepping."""

    def __init__(self, history_limit: int = 5_000) -> None:
        self._lock = RLock()
        self._history: deque[StepRecord] = deque(maxlen=history_limit)

        self._env: Any = None
        self._episode_id = 0
        self._step_count = 0
        self._seed = self._resolve_default_seed()
        self._active_weather_data_path: str | None = None
        self._active_household_data_path: str | None = None
        self._active_market_data_path: str | None = None

        self._latest_observation: dict[str, Any] | None = None
        self._latest_info: dict[str, Any] = {}
        # PV defaults used for simple pv_power estimation when explicit column is absent
        self._default_panel_area = float(config.get("pv", {}).get("panel_area", 1.0))
        self._default_panel_efficiency = float(
            config.get("pv", {}).get("panel_efficiency", 0.15)
        )
        self._default_panel_tilt = float(config.get("pv", {}).get("panel_tilt", 30.0))
        self._default_panel_azimuth = float(
            config.get("pv", {}).get("panel_azimuth", 180.0)
        )
        self._default_temp_coefficient = float(
            config.get("pv", {}).get("temp_coefficient", -0.004)
        )
        self._default_noct = float(config.get("pv", {}).get("noct", 45.0))

    def reset(
        self,
        seed: int | None = None,
        num_households: int | None = None,
        max_episode_steps: int | None = None,
        weather_data_path: str | None = None,
        household_data_path: str | None = None,
        market_data_path: str | None = None,
    ) -> dict[str, Any]:
        """Start a fresh simulation episode and return the initial state."""
        with self._lock:
            effective_seed = self._seed if seed is None else int(seed)
            effective_households = (
                self._resolve_default_households()
                if num_households is None
                else int(num_households)
            )
            effective_max_steps = (
                self._resolve_default_max_steps()
                if max_episode_steps is None
                else int(max_episode_steps)
            )
            effective_weather_data_path = self._resolve_weather_data_path(
                weather_data_path
            )

            effective_household_data_path = self._resolve_optional_csv_path(
                household_data_path
            )
            effective_market_data_path = self._resolve_optional_csv_path(
                market_data_path
            )

            self._env = self._create_env(
                seed=effective_seed,
                num_households=effective_households,
                max_episode_steps=effective_max_steps,
                weather_data_path=effective_weather_data_path,
                household_data_path=effective_household_data_path,
                market_data_path=effective_market_data_path,
            )

            self._seed = effective_seed
            self._episode_id += 1
            self._step_count = 0
            self._active_weather_data_path = effective_weather_data_path
            self._active_household_data_path = effective_household_data_path
            self._active_market_data_path = effective_market_data_path
            self._history.clear()
            self._latest_info = {}
            self._latest_observation = self._env.reset()
            payload = self._build_state_payload(include_topology=True)
            record_simulation_state(payload)
            publish_simulation_event("simulation.reset", payload)
            return payload

    def step(
        self,
        house_actions: list[list[float]] | None = None,
        market_action: int | None = None,
        use_autopilot: bool = True,
    ) -> dict[str, Any]:
        """Advance one timestep and return state, diagnostics, and metrics."""
        with self._lock:
            self._ensure_env()

            assert self._latest_observation is not None
            action_payload = self._build_action_payload(
                observation=self._latest_observation,
                house_actions=house_actions,
                market_action=market_action,
                use_autopilot=use_autopilot,
            )

            observation, reward, done, info = self._env.step(action_payload)
            self._step_count += 1

            self._latest_observation = observation
            self._latest_info = dict(info)
            self._latest_info["done"] = bool(done)

            self._history.append(
                self._build_step_record(observation, reward, done, info)
            )

            payload = self._build_state_payload(include_topology=False)
            record_simulation_state(payload)
            payload["step_result"] = {
                "reward": float(reward),
                "done": bool(done),
                "info": self._to_jsonable(info),
            }
            publish_simulation_event("simulation.step", payload)
            return payload

    def run(
        self,
        steps: int,
        use_autopilot: bool = True,
        market_action: int | None = None,
    ) -> dict[str, Any]:
        """Run multiple steps and return trajectory plus latest state snapshot."""
        if steps <= 0:
            raise ValueError("steps must be greater than zero")

        trajectory: list[dict[str, Any]] = []
        for _ in range(steps):
            step_payload = self.step(
                house_actions=None,
                market_action=market_action,
                use_autopilot=use_autopilot,
            )
            trajectory.append(step_payload["trajectory_point"])
            if bool(step_payload["step_result"]["done"]):
                break

        return {
            "state": self._build_state_payload(include_topology=False),
            "trajectory": trajectory,
            "metrics": self.get_metrics(),
        }

    def get_state(self, include_topology: bool = True) -> dict[str, Any]:
        """Return the latest state; lazily initializes the session when missing."""
        with self._lock:
            self._ensure_env()
            return self._build_state_payload(include_topology=include_topology)

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregate metrics for the current episode."""
        with self._lock:
            if not self._history:
                return {
                    "episode_id": self._episode_id,
                    "steps_executed": self._step_count,
                    "cumulative_reward": 0.0,
                    "average_step_reward": 0.0,
                    "latest_step_reward": 0.0,
                    "average_price": 0.0,
                    "latest_price": 0.0,
                    "peak_demand": 0.0,
                    "peak_supply": 0.0,
                    "total_grid_import": 0.0,
                    "average_renewable_utilization": 0.0,
                }

            rewards = np.asarray(
                [record.reward for record in self._history], dtype=np.float64
            )
            prices = np.asarray(
                [record.price for record in self._history], dtype=np.float64
            )
            demand = np.asarray(
                [record.demand for record in self._history], dtype=np.float64
            )
            supply = np.asarray(
                [record.supply for record in self._history], dtype=np.float64
            )
            grid_imports = np.asarray(
                [record.grid_import for record in self._history],
                dtype=np.float64,
            )
            renewable = np.asarray(
                [record.renewable_utilization for record in self._history],
                dtype=np.float64,
            )

            return {
                "episode_id": self._episode_id,
                "steps_executed": self._step_count,
                "cumulative_reward": float(rewards.sum()),
                "average_step_reward": float(rewards.mean()),
                "latest_step_reward": float(rewards[-1]),
                "average_price": float(prices.mean()),
                "latest_price": float(prices[-1]),
                "peak_demand": float(demand.max()),
                "peak_supply": float(supply.max()),
                "total_grid_import": float(grid_imports.sum()),
                "average_renewable_utilization": float(renewable.mean()),
            }

    def get_history(self, limit: int = 300) -> list[dict[str, Any]]:
        """Return the latest trajectory points for chart rendering."""
        with self._lock:
            capped = max(1, min(int(limit), len(self._history)))
            return [asdict(record) for record in list(self._history)[-capped:]]

    def get_csv_schemas(self) -> dict[str, Any]:
        """Return expected CSV role schemas and current runtime support status."""
        return {
            "weather": {
                "description": "Time-series weather features consumed by GridEnv via load_weather_data.",
                "required_all": [],
                "required_any": [
                    "temperature",
                    "solar_irradiance",
                    "wind_speed",
                    "humidity",
                ],
                "recommended": [
                    "temperature",
                    "solar_irradiance",
                    "wind_speed",
                    "humidity",
                ],
                "runtime_supported_now": True,
                "runtime_usage": (
                    "Pass weather_data_path, household_data_path, "
                    "and market_data_path when resetting simulation."
                ),
            },
            "household": {
                "description": "Household demand series consumed by load_household_data.",
                "required_all": ["consumption"],
                "required_any": [],
                "recommended": ["consumption"],
                "runtime_supported_now": True,
                "runtime_usage": (
                    "Pass household_data_path, weather_data_path, "
                    "and market_data_path when resetting simulation to enable "
                    "household data integration in GridEnv."
                ),
            },
            "market": {
                "description": "Market time-series consumed by load_market_data / MarketEnv.",
                "required_all": ["supply", "demand", "price"],
                "required_any": [],
                "recommended": ["supply", "demand", "price"],
                "runtime_supported_now": True,
                "runtime_usage": (
                    "Pass market_data_path, weather_data_path, "
                    "and household_data_path when resetting simulation to enable "
                    "market data integration in GridEnv."
                ),
            },
        }

    def profile_csv(
        self,
        file_path: str,
        role: str = "auto",
        preview_rows: int = 5,
    ) -> dict[str, Any]:
        """Inspect a CSV file and report role compatibility and runtime usage guidance."""
        requested_role = role.strip().lower()
        allowed_roles = {"auto", "weather", "household", "market"}
        if requested_role not in allowed_roles:
            raise ValueError("role must be one of: auto, weather, household, market")

        if preview_rows <= 0:
            raise ValueError("preview_rows must be greater than zero")

        resolved_path = self._resolve_file_path(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"CSV file not found: {resolved_path}")
        if resolved_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        preview_count = min(max(int(preview_rows), 1), 10)
        frame = pd.read_csv(resolved_path, nrows=preview_count)
        total_rows = self._count_csv_rows(resolved_path)

        analysis_rows = min(max(int(preview_rows) * 30, 120), 2500)
        analysis_frame = pd.read_csv(resolved_path, nrows=analysis_rows)

        time_profile: dict[str, Any] | None = None
        timestamp_candidates = [
            c
            for c in analysis_frame.columns
            if str(c).lower() in {"utc_timestamp", "timestamp", "time", "date"}
            or "timestamp" in str(c).lower()
        ]
        timestamp_col = timestamp_candidates[0] if timestamp_candidates else None

        if timestamp_col:
            ts = pd.to_datetime(
                analysis_frame[timestamp_col], errors="coerce", utc=True
            )
            ok_rate = float(ts.notna().mean())
            if ok_rate > 0.6:
                ts_clean = ts.dropna().sort_values()
                diffs = ts_clean.diff().dropna()
                step_seconds = diffs.dt.total_seconds()
                median_step = (
                    float(step_seconds.median()) if len(step_seconds) else None
                )

                time_profile = {
                    "timestamp_column": str(timestamp_col),
                    "parse_ok_rate": ok_rate,
                    "start": str(ts_clean.iloc[0]) if len(ts_clean) else None,
                    "end": str(ts_clean.iloc[-1]) if len(ts_clean) else None,
                    "median_step_seconds": median_step,
                    "rows_analyzed": int(len(analysis_frame.index)),
                }
            else:
                time_profile = {
                    "timestamp_column": str(timestamp_col),
                    "parse_ok_rate": ok_rate,
                    "start": None,
                    "end": None,
                    "median_step_seconds": None,
                    "rows_analyzed": int(len(analysis_frame.index)),
                }

        columns = [str(column) for column in frame.columns]
        normalized_columns = {column.strip().lower() for column in columns}
        schemas = self.get_csv_schemas()

        compatibility: dict[str, Any] = {}
        for role_name, schema in schemas.items():
            required_all = set(schema.get("required_all", []))
            required_any = set(schema.get("required_any", []))
            recommended = set(schema.get("recommended", []))

            missing_required_all = sorted(required_all - normalized_columns)
            matched_required_any = sorted(required_any.intersection(normalized_columns))
            missing_required_any = (
                sorted(required_any)
                if required_any and not matched_required_any
                else []
            )

            matched_columns = sorted(
                normalized_columns.intersection(
                    required_all.union(required_any).union(recommended)
                )
            )
            compatible = not missing_required_all and not missing_required_any
            coverage_score = (
                len(required_all.intersection(normalized_columns)) * 3
                + len(required_any.intersection(normalized_columns)) * 2
                + len(recommended.intersection(normalized_columns))
            )

            compatibility[role_name] = {
                "compatible": compatible,
                "matched_columns": matched_columns,
                "missing_required_all": missing_required_all,
                "missing_required_any": missing_required_any,
                "coverage_score": int(coverage_score),
                "runtime_supported_now": bool(
                    schema.get("runtime_supported_now", False)
                ),
            }

        ranked_roles = sorted(
            compatibility.items(),
            key=lambda item: (
                bool(item[1]["compatible"]),
                int(item[1]["coverage_score"]),
            ),
            reverse=True,
        )

        inferred_role = "unknown"
        if ranked_roles and int(ranked_roles[0][1]["coverage_score"]) > 0:
            inferred_role = str(ranked_roles[0][0])

        selected_role = inferred_role if requested_role == "auto" else requested_role
        selected_compatibility = compatibility.get(selected_role, {})

        role_diagnostics: dict[str, Any] = {}

        # household diagnostics (best-effort)
        [str(c).lower() for c in analysis_frame.columns]
        household_id_col = next(
            (
                c
                for c in analysis_frame.columns
                if "household" in str(c).lower() and "id" in str(c).lower()
            ),
            None,
        )
        cons_col = next(
            (
                c
                for c in analysis_frame.columns
                if "consump" in str(c).lower()
                or "load" in str(c).lower()
                or "demand" in str(c).lower()
            ),
            None,
        )

        if cons_col:
            cons_series = pd.to_numeric(
                analysis_frame[cons_col], errors="coerce"
            ).dropna()
            role_diagnostics["household"] = {
                "guessed_household_id_column": (
                    str(household_id_col) if household_id_col else None
                ),
                "guessed_consumption_column": str(cons_col),
                "consumption_min": (
                    float(cons_series.min()) if len(cons_series) else 0.0
                ),
                "consumption_max": (
                    float(cons_series.max()) if len(cons_series) else 0.0
                ),
                "consumption_mean": (
                    float(cons_series.mean()) if len(cons_series) else 0.0
                ),
                "unique_households": (
                    int(analysis_frame[household_id_col].nunique())
                    if household_id_col
                    else None
                ),
            }

        # market diagnostics
        price_col = next(
            (c for c in analysis_frame.columns if "price" in str(c).lower()), None
        )
        bid_col = next(
            (
                c
                for c in analysis_frame.columns
                if str(c).lower() == "bid" or "bid" in str(c).lower()
            ),
            None,
        )
        ask_col = next(
            (
                c
                for c in analysis_frame.columns
                if str(c).lower() == "ask" or "ask" in str(c).lower()
            ),
            None,
        )

        if price_col:
            p = pd.to_numeric(analysis_frame[price_col], errors="coerce").dropna()
            bid = (
                pd.to_numeric(analysis_frame[bid_col], errors="coerce")
                if bid_col
                else None
            )
            ask = (
                pd.to_numeric(analysis_frame[ask_col], errors="coerce")
                if ask_col
                else None
            )

            bad_spread = None
            if bid_col and ask_col:
                valid = bid.notna() & ask.notna()
                bad_spread = int((bid[valid] > ask[valid]).sum())

            role_diagnostics["market"] = {
                "guessed_price_column": str(price_col),
                "price_min": float(p.min()) if len(p) else 0.0,
                "price_max": float(p.max()) if len(p) else 0.0,
                "price_mean": float(p.mean()) if len(p) else 0.0,
                "bid_gt_ask_count": bad_spread,
            }

        preview_rows_payload = (
            frame.head(preview_count)
            .where(pd.notnull(frame), None)
            .to_dict(orient="records")
        )
        null_counts = {
            str(column): int(frame[column].isna().sum()) for column in frame.columns
        }

        recommendation = self._build_csv_usage_recommendation(
            selected_role=selected_role,
            compatibility=selected_compatibility,
        )

        # Unit heuristics: warn when columns look like generation vs irradiance
        unit_warnings: list[dict[str, Any]] = []
        numeric_preview = frame.select_dtypes(include=[np.number])
        for col in numeric_preview.columns:
            col_lower = str(col).lower()
            col_vals = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(col_vals) == 0:
                continue
            max_v = float(col_vals.max())

            if (
                "solar" in col_lower or "irradi" in col_lower or "ghi" in col_lower
            ) and max_v < 50:
                unit_warnings.append(
                    {
                        "column": str(col),
                        "kind": "likely_generation_not_irradiance",
                        "message": (
                            f"'{col}' looks too small for irradiance (max={max_v:.2f}). "
                            "It is likely PV generation in kW rather than irradiance in W/m²."
                        ),
                        "suggestion": "Map this column to pv_power or pick a true irradiance field.",
                    }
                )
            if (
                "solar" in col_lower
                or "irradi" in col_lower
                or "ghi" in col_lower
                or "dni" in col_lower
                or "dhi" in col_lower
            ) and max_v > 200:
                unit_warnings.append(
                    {
                        "column": str(col),
                        "kind": "likely_irradiance_not_generation",
                        "message": (
                            f"'{col}' looks too large for PV power (max={max_v:.2f}). "
                            "It is likely irradiance in W/m² rather than generation in kW."
                        ),
                        "suggestion": "Map this column to irradiance or solar_irradiance instead of pv_power.",
                    }
                )

        return {
            "file_path": file_path,
            "resolved_path": str(resolved_path),
            "rows": int(total_rows),
            "column_count": int(len(columns)),
            "columns": columns,
            "numeric_columns": [
                str(column)
                for column in frame.select_dtypes(include=[np.number]).columns
            ],
            "null_counts": null_counts,
            "null_counts_scope": "preview_rows",
            "preview_rows": self._to_jsonable(preview_rows_payload),
            "requested_role": requested_role,
            "inferred_role": inferred_role,
            "selected_role": selected_role,
            "compatibility": compatibility,
            "can_use_now": bool(
                selected_compatibility.get("compatible", False)
                and selected_compatibility.get("runtime_supported_now", False)
            ),
            "usage_recommendation": recommendation,
            "unit_warnings": unit_warnings,
            "time_profile": time_profile,
            "role_diagnostics": role_diagnostics,
        }

    def derive_weather_csv(
        self,
        file_path: str,
        solar_column: str,
        wind_column: str,
        timestamp_column: str | None = None,
        temperature_column: str | None = None,
        humidity_column: str | None = None,
        irradiance_column: str | None = None,
        ghi_column: str | None = None,
        dni_column: str | None = None,
        dhi_column: str | None = None,
        pv_power_column: str | None = None,
        output_path: str | None = None,
        normalize_signals: bool = True,
        panel_tilt: float | None = None,
        panel_azimuth: float | None = None,
        panel_area: float | None = None,
        panel_efficiency: float | None = None,
        temp_coefficient: float | None = None,
    ) -> dict[str, Any]:
        """Generate a GridEnv-compatible weather CSV from a wide source timeseries file.

        Backward compatible notes:
        - `irradiance_column` is preferred for explicit irradiance input. If absent,
            the legacy `solar_column` will be used as `solar_irradiance` output.
        - If `pv_power_column` is not provided but irradiance is available, a
            simple PV power estimate will be computed using conservative defaults.
        """

        source_path = self._resolve_file_path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")
        if source_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        # Required: solar (legacy) or irradiance preferred, and wind
        required_columns = [solar_column, wind_column]
        optional_columns = [
            timestamp_column,
            temperature_column,
            humidity_column,
            irradiance_column,
            ghi_column,
            dni_column,
            dhi_column,
            pv_power_column,
        ]
        selected_columns = [
            column_name
            for column_name in required_columns + optional_columns
            if column_name is not None and column_name.strip() != ""
        ]

        frame = pd.read_csv(source_path, usecols=selected_columns)
        missing_columns = [
            column_name
            for column_name in required_columns
            if column_name not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required columns in source CSV: " + ", ".join(missing_columns)
            )

        # Read numeric series with safe coercion
        # Prefer explicit irradiance_column if provided; otherwise use legacy solar_column
        irradiance_source = None
        if irradiance_column and irradiance_column in frame.columns:
            irradiance_source = irradiance_column
        elif solar_column and solar_column in frame.columns:
            irradiance_source = solar_column

        # Also detect explicit pv_power column if present
        pv_power_source = (
            pv_power_column
            if (pv_power_column and pv_power_column in frame.columns)
            else None
        )

        # If only pv_power is present but no irradiance, do NOT assume pv_power is irradiance
        if irradiance_source is not None:
            irradiance_series = pd.to_numeric(
                frame[irradiance_source], errors="coerce"
            ).fillna(0.0)
        else:
            irradiance_series = pd.Series(np.zeros(len(frame.index)), dtype=np.float32)

        wind_series = pd.to_numeric(frame[wind_column], errors="coerce").fillna(0.0)

        # Normalization scales (preserve legacy behavior for solar/wind)
        irradiance_scale = float(max(irradiance_series.max(), 1.0))
        wind_scale = float(max(wind_series.max(), 1.0))
        if normalize_signals:
            irradiance_output = np.clip(irradiance_series / irradiance_scale, 0.0, 1.0)
            wind_output = np.clip(wind_series / wind_scale, 0.0, 1.0)
        else:
            irradiance_output = irradiance_series
            wind_output = wind_series

        if temperature_column and temperature_column in frame.columns:
            temperature_output = (
                pd.to_numeric(frame[temperature_column], errors="coerce")
                .fillna(20.0)
                .astype(np.float32)
            )
        else:
            temperature_output = pd.Series(
                np.full(len(frame.index), 20.0, dtype=np.float32)
            )

        if humidity_column and humidity_column in frame.columns:
            humidity_output = (
                pd.to_numeric(frame[humidity_column], errors="coerce")
                .fillna(50.0)
                .clip(0.0, 100.0)
                .astype(np.float32)
            )
        else:
            humidity_output = pd.Series(
                np.full(len(frame.index), 50.0, dtype=np.float32)
            )

        # Optional auxiliary columns (GHI/DNI/DHI)
        def _optional_series(col_name: str | None) -> pd.Series:
            if col_name and col_name in frame.columns:
                return (
                    pd.to_numeric(frame[col_name], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                )
            return pd.Series(np.full(len(frame.index), 0.0, dtype=np.float32))

        ghi_output = _optional_series(ghi_column)
        dni_output = _optional_series(dni_column)
        dhi_output = _optional_series(dhi_column)

        # pv_power: prefer explicit column, otherwise estimate from raw irradiance (W/m^2)
        if pv_power_source is not None:
            pv_power_output = (
                pd.to_numeric(frame[pv_power_source], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        else:
            # Simple PV estimate: use raw irradiance_series (W/m^2) * area * efficiency
            panel_area = (
                panel_area
                if panel_area is not None
                else float(getattr(self, "_default_panel_area", 1.0))
            )
            panel_eff = (
                panel_efficiency
                if panel_efficiency is not None
                else float(getattr(self, "_default_panel_efficiency", 0.15))
            )
            temp_coeff = (
                temp_coefficient
                if temp_coefficient is not None
                else float(getattr(self, "_default_temp_coefficient", -0.004))
            )
            noct = float(getattr(self, "_default_noct", 45.0))
            panel_tilt_value = (
                panel_tilt
                if panel_tilt is not None
                else float(getattr(self, "_default_panel_tilt", 30.0))
            )
            panel_azimuth_value = (
                panel_azimuth
                if panel_azimuth is not None
                else float(getattr(self, "_default_panel_azimuth", 180.0))
            )
            site_latitude = float(config.get("site", {}).get("latitude", 35.0))

            def _fallback_orientation_factor() -> float:
                tilt_rad = np.deg2rad(panel_tilt_value)
                azimuth_rad = np.deg2rad(panel_azimuth_value)
                ideal_tilt = np.deg2rad(np.clip(abs(site_latitude), 0.0, 60.0))
                tilt_alignment = float(
                    np.clip(np.cos(tilt_rad - ideal_tilt), 0.15, 1.0)
                )
                south_alignment = float(
                    np.clip(np.cos(azimuth_rad - np.deg2rad(180.0)), 0.2, 1.0)
                )
                return float(np.clip(tilt_alignment * south_alignment, 0.1, 1.0))

            tilt_factor: float | np.ndarray
            tilt_factor = _fallback_orientation_factor()

            if timestamp_column and timestamp_column in frame.columns:
                timestamps = pd.to_datetime(
                    frame[timestamp_column], errors="coerce", utc=True
                )
                valid_mask = timestamps.notna().to_numpy()
                if bool(valid_mask.any()):
                    ts_valid = timestamps[valid_mask]
                    day_of_year = ts_valid.dt.dayofyear.to_numpy(dtype=np.float32)
                    hour_decimal = (
                        ts_valid.dt.hour.to_numpy(dtype=np.float32)
                        + ts_valid.dt.minute.to_numpy(dtype=np.float32) / 60.0
                        + ts_valid.dt.second.to_numpy(dtype=np.float32) / 3600.0
                    )

                    latitude_rad = np.deg2rad(site_latitude)
                    declination = np.deg2rad(
                        23.45
                        * np.sin(np.deg2rad(360.0 * (284.0 + day_of_year) / 365.0))
                    )
                    hour_angle = np.deg2rad(15.0 * (hour_decimal - 12.0))

                    sin_altitude = np.sin(latitude_rad) * np.sin(declination) + np.cos(
                        latitude_rad
                    ) * np.cos(declination) * np.cos(hour_angle)
                    sin_altitude = np.clip(sin_altitude, 0.0, 1.0)
                    altitude = np.arcsin(sin_altitude)
                    cos_altitude = np.cos(altitude)

                    solar_azimuth = (
                        np.arctan2(
                            np.sin(hour_angle),
                            np.cos(hour_angle) * np.sin(latitude_rad)
                            - np.tan(declination) * np.cos(latitude_rad),
                        )
                        + np.pi
                    )

                    sun_x = cos_altitude * np.sin(solar_azimuth)
                    sun_y = cos_altitude * np.cos(solar_azimuth)
                    sun_z = np.sin(altitude)

                    tilt_rad = np.deg2rad(panel_tilt_value)
                    azimuth_rad = np.deg2rad(panel_azimuth_value)
                    panel_x = np.sin(tilt_rad) * np.sin(azimuth_rad)
                    panel_y = np.sin(tilt_rad) * np.cos(azimuth_rad)
                    panel_z = np.cos(tilt_rad)

                    cos_incidence = np.clip(
                        sun_x * panel_x + sun_y * panel_y + sun_z * panel_z,
                        0.0,
                        1.0,
                    )
                    poa_factor = np.divide(
                        cos_incidence,
                        np.maximum(sin_altitude, 0.1),
                        out=np.zeros_like(cos_incidence),
                        where=np.ones_like(cos_incidence, dtype=bool),
                    )
                    poa_factor = np.clip(poa_factor, 0.1, 2.5)

                    tilt_factor_array = np.full(
                        len(frame.index),
                        _fallback_orientation_factor(),
                        dtype=np.float32,
                    )
                    tilt_factor_array[valid_mask] = poa_factor.astype(np.float32)
                    tilt_factor = tilt_factor_array

            # Temperature-adjustment: if temperature column provided, estimate module temp
            if temperature_column and temperature_column in frame.columns:
                ambient_temp = pd.to_numeric(
                    frame[temperature_column], errors="coerce"
                ).fillna(20.0)
                # crude module temperature estimate using NOCT and irradiance (W/m2)
                module_temp = ambient_temp + (noct - 20.0) / 800.0 * irradiance_series
                temp_adj = (1.0 + temp_coeff * (module_temp - 25.0)).clip(0.5, 1.5)
            else:
                temp_adj = 1.0

            pv_kw = (
                irradiance_series * panel_area * panel_eff * tilt_factor * temp_adj
            ) / 1000.0
            pv_power_output = pd.Series(np.asarray(pv_kw, dtype=np.float32))

        weather_frame = pd.DataFrame(
            {
                "temperature": temperature_output,
                "solar_irradiance": np.asarray(irradiance_output, dtype=np.float32),
                "wind_speed": np.asarray(wind_output, dtype=np.float32),
                "humidity": humidity_output,
                "ghi": ghi_output,
                "dni": dni_output,
                "dhi": dhi_output,
                "pv_power": pv_power_output,
            }
        )

        if timestamp_column and timestamp_column in frame.columns:
            weather_frame.insert(
                0, "utc_timestamp", frame[timestamp_column].astype(str)
            )

        resolved_output_path = self._resolve_output_csv_path(
            source_path=source_path,
            output_path=output_path,
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        weather_frame.to_csv(resolved_output_path, index=False)

        return {
            "source_file_path": file_path,
            "resolved_source_path": str(source_path),
            "output_file_path": str(resolved_output_path),
            "rows": int(len(weather_frame.index)),
            "columns": [str(column_name) for column_name in weather_frame.columns],
            "column_mapping": {
                "solar_irradiance": irradiance_source,
                "wind_speed": wind_column,
                "temperature": temperature_column,
                "humidity": humidity_column,
                "utc_timestamp": timestamp_column,
                "ghi": ghi_column,
                "dni": dni_column,
                "dhi": dhi_column,
                "pv_power": (
                    pv_power_source if pv_power_source in frame.columns else None
                ),
            },
            "normalization": {
                "enabled": bool(normalize_signals),
                "irradiance_scale": irradiance_scale,
                "wind_scale": wind_scale,
            },
            "usage_recommendation": (
                "Use the output_file_path as weather_data_path in the dashboard CSV controls."
            ),
        }

    def derive_household_csv(
        self,
        file_path: str,
        consumption_column: str,
        timestamp_column: str | None = None,
        household_id_column: str | None = None,
        pv_generation_column: str | None = None,
        net_load_column: str | None = None,
        output_path: str | None = None,
        normalize_signals: bool = False,
    ) -> dict[str, Any]:
        source_path = self._resolve_file_path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")
        if source_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        required = [consumption_column]
        selected = [
            c
            for c in [
                timestamp_column,
                household_id_column,
                consumption_column,
                pv_generation_column,
                net_load_column,
            ]
            if c
        ]
        frame = pd.read_csv(source_path, usecols=selected)

        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                "Missing required columns in source CSV: " + ", ".join(missing)
            )

        cons = pd.to_numeric(frame[consumption_column], errors="coerce").fillna(0.0)
        if normalize_signals:
            scale = float(max(cons.max(), 1.0))
            cons_out = np.clip(cons / scale, 0.0, 1.0).astype(np.float32)
        else:
            scale = float(max(cons.max(), 1.0))
            cons_out = cons.astype(np.float32)

        out = pd.DataFrame({"consumption": cons_out})

        if pv_generation_column and pv_generation_column in frame.columns:
            out["pv_generation"] = (
                pd.to_numeric(frame[pv_generation_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        if net_load_column and net_load_column in frame.columns:
            out["net_load"] = (
                pd.to_numeric(frame[net_load_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )

        if household_id_column and household_id_column in frame.columns:
            out.insert(0, "household_id", frame[household_id_column])

        if timestamp_column and timestamp_column in frame.columns:
            out.insert(0, "utc_timestamp", frame[timestamp_column].astype(str))

        resolved_output = self._resolve_output_role_csv_path(
            source_path, output_path, role="household"
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(resolved_output, index=False)

        return {
            "source_file_path": file_path,
            "resolved_source_path": str(source_path),
            "output_file_path": str(resolved_output),
            "rows": int(len(out.index)),
            "columns": [str(c) for c in out.columns],
            "column_mapping": {
                "utc_timestamp": timestamp_column,
                "household_id": household_id_column,
                "consumption": consumption_column,
                "pv_generation": pv_generation_column,
                "net_load": net_load_column,
            },
            "normalization": {
                "enabled": bool(normalize_signals),
                "consumption_scale": float(scale),
            },
            "usage_recommendation": (
                "Use output_file_path as household_data_path in the dashboard CSV controls."
            ),
        }

    def derive_market_csv(
        self,
        file_path: str,
        supply_column: str | None = None,
        demand_column: str | None = None,
        price_column: str = Field(min_length=1),
        timestamp_column: str | None = None,
        bid_column: str | None = None,
        ask_column: str | None = None,
        clearing_price_column: str | None = None,
        output_path: str | None = None,
        normalize_signals: bool = False,
    ) -> dict[str, Any]:
        source_path = self._resolve_file_path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")
        if source_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        required = [price_column]
        selected = [
            c
            for c in [
                timestamp_column,
                supply_column,
                demand_column,
                price_column,
                bid_column,
                ask_column,
                clearing_price_column,
            ]
            if c
        ]
        frame = pd.read_csv(source_path, usecols=selected)

        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                "Missing required columns in source CSV: " + ", ".join(missing)
            )

        warnings: list[str] = []
        price = pd.to_numeric(frame[price_column], errors="coerce").fillna(0.0)

        if supply_column and supply_column in frame.columns:
            supply = pd.to_numeric(frame[supply_column], errors="coerce").fillna(0.0)
        else:
            supply = pd.Series(np.zeros(len(frame.index)), dtype=np.float32)
            warnings.append("supply_column not provided; supply filled with zeros.")

        if demand_column and demand_column in frame.columns:
            demand = pd.to_numeric(frame[demand_column], errors="coerce").fillna(0.0)
        else:
            demand = pd.Series(np.zeros(len(frame.index)), dtype=np.float32)
            warnings.append("demand_column not provided; demand filled with zeros.")

        out = pd.DataFrame(
            {
                "supply": supply.astype(np.float32),
                "demand": demand.astype(np.float32),
                "price": price.astype(np.float32),
            }
        )

        if bid_column and bid_column in frame.columns:
            out["bid"] = (
                pd.to_numeric(frame[bid_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        if ask_column and ask_column in frame.columns:
            out["ask"] = (
                pd.to_numeric(frame[ask_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        if clearing_price_column and clearing_price_column in frame.columns:
            out["clearing_price"] = (
                pd.to_numeric(frame[clearing_price_column], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )

        if timestamp_column and timestamp_column in frame.columns:
            out.insert(0, "utc_timestamp", frame[timestamp_column].astype(str))

        resolved_output = self._resolve_output_role_csv_path(
            source_path, output_path, role="market"
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(resolved_output, index=False)

        return {
            "source_file_path": file_path,
            "resolved_source_path": str(source_path),
            "output_file_path": str(resolved_output),
            "rows": int(len(out.index)),
            "columns": [str(c) for c in out.columns],
            "column_mapping": {
                "utc_timestamp": timestamp_column,
                "supply": supply_column,
                "demand": demand_column,
                "price": price_column,
                "bid": bid_column,
                "ask": ask_column,
                "clearing_price": clearing_price_column,
            },
            "warnings": warnings,
            "normalization": {"enabled": bool(normalize_signals)},
            "usage_recommendation": (
                "Use output_file_path as market_data_path in the dashboard CSV controls."
            ),
        }

    def store_uploaded_household_csv(
        self, file_name: str, file_bytes: bytes
    ) -> dict[str, Any]:
        if not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")

        backend_root = Path(__file__).resolve().parents[2]
        upload_dir = backend_root / "data" / "uploads" / "household"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = (
            "".join(
                c if c.isalnum() or c in {"-", "_"} else "_"
                for c in Path(file_name).stem
            ).strip("_")
            or "uploaded_household"
        )

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        resolved_path = (upload_dir / f"{timestamp}_{safe_stem}.csv").resolve()
        resolved_path.write_bytes(file_bytes)

        frame = pd.read_csv(resolved_path, nrows=5)
        total_rows = self._count_csv_rows(resolved_path)

        return {
            "file_path": file_name,
            "resolved_path": str(resolved_path),
            "rows": int(total_rows),
            "columns": [str(c) for c in frame.columns],
            "usage_recommendation": (
                "Use resolved_path as household_data_path in /simulation/reset. "
                "Profile it in Inspect first; derive a canonical household CSV if needed."
            ),
        }

    def store_uploaded_market_csv(
        self, file_name: str, file_bytes: bytes
    ) -> dict[str, Any]:
        if not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")

        backend_root = Path(__file__).resolve().parents[2]
        upload_dir = backend_root / "data" / "uploads" / "market"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = (
            "".join(
                c if c.isalnum() or c in {"-", "_"} else "_"
                for c in Path(file_name).stem
            ).strip("_")
            or "uploaded_market"
        )

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        resolved_path = (upload_dir / f"{timestamp}_{safe_stem}.csv").resolve()
        resolved_path.write_bytes(file_bytes)

        frame = pd.read_csv(resolved_path, nrows=5)
        total_rows = self._count_csv_rows(resolved_path)

        return {
            "file_path": file_name,
            "resolved_path": str(resolved_path),
            "rows": int(total_rows),
            "columns": [str(c) for c in frame.columns],
            "usage_recommendation": (
                "Use resolved_path as market_data_path in /simulation/reset. "
                "Profile it in Inspect first; derive a canonical market CSV if needed."
            ),
        }

    def store_uploaded_weather_csv(
        self, file_name: str, file_bytes: bytes
    ) -> dict[str, Any]:
        """Persist an uploaded CSV inside the backend workspace for later reset calls."""
        if not file_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")

        backend_root = Path(__file__).resolve().parents[2]
        upload_dir = backend_root / "data" / "uploads" / "weather"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in Path(file_name).stem
        ).strip("_")
        if not safe_stem:
            safe_stem = "uploaded_weather"

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        resolved_path = (upload_dir / f"{timestamp}_{safe_stem}.csv").resolve()
        resolved_path.write_bytes(file_bytes)

        frame = pd.read_csv(resolved_path, nrows=5)
        total_rows = self._count_csv_rows(resolved_path)

        return {
            "file_path": file_name,
            "resolved_path": str(resolved_path),
            "rows": int(total_rows),
            "columns": [str(column_name) for column_name in frame.columns],
            "usage_recommendation": (
                "Use resolved_path as weather_data_path in /simulation/reset. "
                "If this is a wide source dataset, profile it and derive a weather CSV first."
            ),
        }

    @staticmethod
    def list_available_csv_paths() -> dict[str, Any]:
        """Return backend CSV files that can be used or re-used by the dashboard."""
        backend_root = Path(__file__).resolve().parents[2]
        candidate_dirs = [
            backend_root / "data" / "uploads" / "weather",
            backend_root / "data" / "weather_data",
            backend_root / "data" / "historical_energy_data",
            backend_root / "data" / "market_prices",
            backend_root / "data" / "uploads" / "household",
            backend_root / "data" / "uploads" / "market",
        ]

        seen_paths: set[str] = set()
        items: list[dict[str, Any]] = []

        for directory in candidate_dirs:
            if not directory.exists() or not directory.is_dir():
                continue

            for csv_path in sorted(directory.glob("*.csv")):
                resolved_path = str(csv_path.resolve())
                if resolved_path in seen_paths:
                    continue

                seen_paths.add(resolved_path)
                if "uploads" in csv_path.parts:
                    if "weather" in csv_path.parts:
                        kind = "uploaded-weather"
                    elif "household" in csv_path.parts:
                        kind = "uploaded-household"
                    elif "market" in csv_path.parts:
                        kind = "uploaded-market"
                    else:
                        kind = "uploaded"
                elif "weather_data" in csv_path.parts:
                    kind = "derived-or-weather"
                elif "historical_energy_data" in csv_path.parts:
                    kind = "household"
                elif "market_prices" in csv_path.parts:
                    kind = "market"
                else:
                    kind = "unknown"

                items.append(
                    {
                        "path": resolved_path,
                        "kind": kind,
                        "label": f"{kind}: {csv_path.name}",
                    }
                )

        return {
            "paths": items,
            "count": len(items),
        }

    def _create_env(
        self,
        seed: int,
        num_households: int,
        max_episode_steps: int,
        weather_data_path: str,
        household_data_path: str | None,
        market_data_path: str | None,
    ) -> Any:
        # Lazy import avoids importing heavy model stack at app startup.
        from app.simulations.grid_env import GridEnv

        paths = self._resolve_data_paths()
        env = GridEnv(
            grid_topology_file=paths["grid_topology"],
            weather_file=weather_data_path,
            household_file=household_data_path,
            market_file=market_data_path,
            num_households=num_households,
            max_episode_steps=max_episode_steps,
        )
        env.seed(seed)
        return env

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        self.reset(seed=self._seed)

    def _build_action_payload(
        self,
        observation: dict[str, Any],
        house_actions: list[list[float]] | None,
        market_action: int | None,
        use_autopilot: bool,
    ) -> dict[str, Any]:
        assert self._env is not None

        if house_actions is None:
            if use_autopilot:
                parsed_house_actions = self._build_autopilot_actions(observation)
            else:
                parsed_house_actions = np.zeros(
                    (self._env.num_households, 6),
                    dtype=np.float32,
                )
        else:
            parsed_house_actions = np.asarray(house_actions, dtype=np.float32)
            expected_shape = (self._env.num_households, 6)
            if parsed_house_actions.shape != expected_shape:
                raise ValueError(
                    f"house_actions shape mismatch: expected {expected_shape}, "
                    f"received {parsed_house_actions.shape}"
                )
            parsed_house_actions = np.clip(parsed_house_actions, 0.0, 1.0)

        resolved_market_action = 1 if market_action is None else int(market_action)
        resolved_market_action = int(np.clip(resolved_market_action, 0, 1))

        return {
            "house_actions": parsed_house_actions,
            "market_actions": resolved_market_action,
        }

    def _build_autopilot_actions(self, observation: dict[str, Any]) -> np.ndarray:
        assert self._env is not None

        house_states = np.asarray(observation["house_states"], dtype=np.float32)
        max_battery = float(self._resolve_default_max_battery())
        price_ceiling = float(self._resolve_price_max())
        default_price = float(self._resolve_default_price())

        actions = np.zeros((self._env.num_households, 6), dtype=np.float32)
        for index, state in enumerate(house_states):
            consumption = float(state[1])
            production = float(state[2])
            battery_level = float(state[3])
            price = float(state[4])
            net_balance = float(state[9])

            demand_response = np.clip(
                1.0 - (price / max(price_ceiling, 1e-6)),
                0.0,
                1.0,
            )
            charge_signal = 0.7 if battery_level < (0.45 * max_battery) else 0.15
            discharge_signal = (
                0.65
                if (battery_level > (0.60 * max_battery) and price > default_price)
                else 0.10
            )
            buy_signal = np.clip(-net_balance, 0.0, 1.0)
            sell_signal = np.clip(net_balance, 0.0, 1.0)
            grid_import = np.clip(max(consumption - production, 0.0), 0.0, 1.0)

            actions[index] = np.array(
                [
                    demand_response,
                    charge_signal,
                    discharge_signal,
                    float(buy_signal),
                    float(sell_signal),
                    float(grid_import),
                ],
                dtype=np.float32,
            )

        return actions

    def _build_state_payload(self, include_topology: bool) -> dict[str, Any]:
        assert self._env is not None
        assert self._latest_observation is not None

        payload: dict[str, Any] = {
            "episode_id": self._episode_id,
            "seed": self._seed,
            "step": self._step_count,
            "data_sources": {
                "weather_data": (
                    self._active_weather_data_path
                    if self._active_weather_data_path is not None
                    else self._resolve_data_paths()["weather_data"]
                ),
                "household_data": self._active_household_data_path,
                "market_data": self._active_market_data_path,
            },
            "observation": self._to_jsonable(self._latest_observation),
            "latest_info": self._to_jsonable(self._latest_info),
            "metrics": self.get_metrics(),
        }

        trajectory_point = self._build_trajectory_point()
        payload["trajectory_point"] = trajectory_point

        if include_topology:
            payload["topology"] = self._build_topology_payload()

        return payload

    def _build_topology_payload(self) -> dict[str, Any]:
        assert self._env is not None

        nodes = []
        for node_id, attrs in self._env.graph.nodes(data=True):
            node_payload = {
                "id": int(node_id),
                "type": str(attrs.get("type", "unknown")),
                "label": str(attrs.get("label", node_id)),
            }
            node_payload.update(self._to_jsonable(attrs))
            node_payload["id"] = int(node_id)
            node_payload["type"] = str(attrs.get("type", "unknown"))
            node_payload["label"] = str(attrs.get("label", node_id))
            nodes.append(node_payload)

        edges = []
        for source, target, attrs in self._env.graph.edges(data=True):
            edges.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "weight": float(attrs.get("weight", 1.0)),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "layout": self._to_jsonable(
                self._env.graph.graph.get("layout", "neighborhood-grid")
            ),
            "bounds": self._to_jsonable(self._env.graph.graph.get("bounds")),
        }

    def _build_step_record(
        self,
        observation: dict[str, Any],
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> StepRecord:
        house_states = np.asarray(observation["house_states"], dtype=np.float32)
        market_snapshot = (
            info.get("market_snapshot", {}) if isinstance(info, dict) else {}
        )

        demand = float(market_snapshot.get("effective_demand", 0.0))
        supply = float(market_snapshot.get("effective_supply", 0.0))
        price = float(market_snapshot.get("clearing_price", 0.0))

        grid_import = 0.0
        renewable_utilization = 0.0
        if house_states.size > 0:
            grid_import = float(np.sum(house_states[:, 5]))
            total_consumption = float(np.sum(house_states[:, 1]))
            total_renewable = float(np.sum(house_states[:, 2]))
            if total_consumption > 0.0:
                renewable_utilization = float(
                    np.clip(total_renewable / total_consumption, 0.0, 1.0)
                )

        return StepRecord(
            step=self._step_count,
            timestamp=datetime.now(tz=UTC).isoformat(),
            reward=float(reward),
            done=bool(done),
            supply=supply,
            demand=demand,
            price=price,
            grid_import=grid_import,
            renewable_utilization=renewable_utilization,
        )

    def _build_trajectory_point(self) -> dict[str, Any]:
        if not self._history:
            return {
                "step": self._step_count,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "reward": 0.0,
                "done": False,
                "supply": 0.0,
                "demand": 0.0,
                "price": 0.0,
                "grid_import": 0.0,
                "renewable_utilization": 0.0,
            }

        return asdict(self._history[-1])

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(k): SimulationService._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SimulationService._to_jsonable(item) for item in value]
        return value

    @staticmethod
    def _resolve_default_seed() -> int:
        reproducibility = config.get("reproducibility", {})
        repro_seed = (
            reproducibility.get("seed", None)
            if hasattr(reproducibility, "get")
            else None
        )
        if repro_seed is not None:
            return int(repro_seed)

        return int(config.get("seed", 42))

    @staticmethod
    def _resolve_default_households() -> int:
        env_cfg = config.get("env", {})
        env_value = (
            env_cfg.get("num_households", None) if hasattr(env_cfg, "get") else None
        )
        return int(
            config.get("num_households", env_value if env_value is not None else 10)
        )

    @staticmethod
    def _resolve_default_max_steps() -> int:
        env_cfg = config.get("env", {})
        env_value = (
            env_cfg.get("max_episode_steps", None) if hasattr(env_cfg, "get") else None
        )
        training_steps = config.get("training_steps", None)
        if training_steps is not None:
            return int(training_steps)
        return int(env_value if env_value is not None else 1000)

    @staticmethod
    def _resolve_default_max_battery() -> float:
        env_cfg = config.get("env", {})
        env_value = (
            env_cfg.get("max_battery", None) if hasattr(env_cfg, "get") else None
        )
        top_level = config.get("max_battery", None)
        if top_level is not None:
            return float(top_level)
        return float(env_value if env_value is not None else 10.0)

    @staticmethod
    def _resolve_default_price() -> float:
        market_cfg = config.get("market", {})
        market_value = (
            market_cfg.get("default_price", None)
            if hasattr(market_cfg, "get")
            else None
        )
        return float(market_value if market_value is not None else 0.3)

    @staticmethod
    def _resolve_price_max() -> float:
        market_cfg = config.get("market", {})
        market_value = (
            market_cfg.get("price_max", None) if hasattr(market_cfg, "get") else None
        )
        return float(market_value if market_value is not None else 1.0)

    def _resolve_weather_data_path(self, weather_data_path: str | None) -> str:
        defaults = self._resolve_data_paths()
        if weather_data_path is None or weather_data_path.strip() == "":
            return str(defaults["weather_data"])

        resolved = self._resolve_file_path(weather_data_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Weather CSV not found: {resolved}")
        if resolved.suffix.lower() != ".csv":
            raise ValueError("weather_data_path must point to a .csv file")
        return str(resolved)

    def _resolve_optional_csv_path(self, maybe_path: str | None) -> str | None:
        if maybe_path is None or maybe_path.strip() == "":
            return None
        resolved = self._resolve_file_path(maybe_path)
        if not resolved.exists():
            raise FileNotFoundError(f"CSV not found: {resolved}")
        if resolved.suffix.lower() != ".csv":
            raise ValueError("Path must point to a .csv file")
        return str(resolved)

    @staticmethod
    def _resolve_file_path(file_path: str) -> Path:
        # Safely resolve a user-provided path and restrict it to allowed project roots.
        raw_path = Path(file_path).expanduser()

        backend_root = Path(__file__).resolve().parents[2].resolve()
        workspace_root = backend_root.parent.resolve()

        def _is_within_allowed(path: Path) -> bool:
            try:
                resolved = path.resolve(strict=False)
            except Exception:
                return False
            try:
                resolved.relative_to(backend_root)
                return True
            except Exception:
                pass
            try:
                resolved.relative_to(workspace_root)
                return True
            except Exception:
                return False

        # Absolute paths must be located inside the project workspace/backend
        if raw_path.is_absolute():
            resolved = raw_path.resolve(strict=False)
            if not _is_within_allowed(resolved):
                raise ValueError("Absolute paths outside the project are forbidden")
            if resolved.exists():
                return resolved

        # If the exact path is missing, try to recover by matching the basename
        # against known CSV directories inside the backend workspace.
        candidate_name = raw_path.name
        if candidate_name:
            backend_candidates = [
                backend_root / "data" / "uploads" / "weather" / candidate_name,
                backend_root / "data" / "weather_data" / candidate_name,
                backend_root / "data" / "historical_energy_data" / candidate_name,
                backend_root / "data" / "market_prices" / candidate_name,
            ]
            matched_candidates = [
                candidate.resolve(strict=False)
                for candidate in backend_candidates
                if candidate.exists()
            ]
            if len(matched_candidates) == 1:
                return matched_candidates[0]
            if len(matched_candidates) > 1:
                return matched_candidates[0]

        if raw_path.is_absolute():
            return resolved

        # For relative references prefer backend then workspace
        backend_candidate = (backend_root / raw_path).resolve(strict=False)
        workspace_candidate = (workspace_root / raw_path).resolve(strict=False)

        if backend_candidate.exists() and _is_within_allowed(backend_candidate):
            return backend_candidate
        if workspace_candidate.exists() and _is_within_allowed(workspace_candidate):
            return workspace_candidate

        # If neither exists, return a backend-local candidate (for output paths)
        if _is_within_allowed(backend_candidate):
            return backend_candidate

        raise ValueError("Resolved path is outside allowed project directories")

    @staticmethod
    def _count_csv_rows(file_path: Path) -> int:
        row_count = 0
        with file_path.open(
            "r", encoding="utf-8", errors="ignore", newline=""
        ) as handle:
            # Exclude header row from the total timestep count.
            for _ in handle:
                row_count += 1

        return max(row_count - 1, 0)

    @staticmethod
    def _resolve_output_role_csv_path(
        source_path: Path, output_path: str | None, role: str
    ) -> Path:
        if output_path is not None and output_path.strip() != "":
            requested = SimulationService._resolve_file_path(output_path)
            if requested.suffix.lower() != ".csv":
                raise ValueError("output_path must point to a .csv file")
            return requested

        backend_root = Path(__file__).resolve().parents[2]
        stem = source_path.stem.replace(" ", "_")
        if role == "household":
            default_name = f"derived_{stem}_household.csv"
            return (
                backend_root / "data" / "historical_energy_data" / default_name
            ).resolve()
        if role == "market":
            default_name = f"derived_{stem}_market.csv"
            return (backend_root / "data" / "market_prices" / default_name).resolve()

        default_name = f"derived_{stem}_{role}.csv"
        return (backend_root / "data" / default_name).resolve()

    @staticmethod
    def _resolve_output_csv_path(source_path: Path, output_path: str | None) -> Path:
        if output_path is not None and output_path.strip() != "":
            requested = SimulationService._resolve_file_path(output_path)
            if requested.suffix.lower() != ".csv":
                raise ValueError("output_path must point to a .csv file")
            return requested

        backend_root = Path(__file__).resolve().parents[2]
        stem = source_path.stem.replace(" ", "_")
        default_name = f"derived_{stem}_weather.csv"
        return (backend_root / "data" / "weather_data" / default_name).resolve()

    @staticmethod
    def _build_csv_usage_recommendation(
        selected_role: str,
        compatibility: dict[str, Any],
    ) -> str:
        if selected_role == "weather":
            if compatibility.get("compatible", False):
                return (
                    "This CSV is compatible with weather ingestion and can be used immediately "
                    "by passing weather_data_path when resetting the simulation."
                )
            return (
                "This CSV is not yet weather-compatible. Add at least one weather signal column "
                "(temperature, solar_irradiance, wind_speed, humidity)."
            )

        if selected_role == "household":
            if compatibility.get("compatible", False):
                return (
                    "This CSV is compatible with household ingestion and can be used immediately "
                    "by passing household_data_path when resetting the simulation."
                )
            return "Add a consumption column to make this household-compatible."

        if selected_role == "market":
            if compatibility.get("compatible", False):
                return (
                    "This CSV is compatible with market ingestion and can be used immediately "
                    "by passing market_data_path when resetting the simulation."
                )
            return (
                "Add supply, demand, and price columns to make this market-compatible."
            )

        return (
            "Could not confidently infer a role. Choose weather/household/market explicitly "
            "to validate against a specific schema."
        )

    @staticmethod
    def _resolve_data_paths() -> dict[str, str]:
        defaults = {
            "grid_topology": "data/grid_topology/sample_grid.json",
            "weather_data": "data/weather_data/sample_weather.csv",
        }
        data_paths = config.get("data_paths", {})
        if hasattr(data_paths, "get"):
            return {
                "grid_topology": str(
                    data_paths.get("grid_topology", defaults["grid_topology"])
                ),
                "weather_data": str(
                    data_paths.get("weather_data", defaults["weather_data"])
                ),
            }
        return defaults


simulation_service = SimulationService()
