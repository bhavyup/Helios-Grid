"""Simulation session service for deterministic, API-driven grid runs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.project_config import config


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
        self._history: Deque[StepRecord] = deque(maxlen=history_limit)

        self._env: Any = None
        self._episode_id = 0
        self._step_count = 0
        self._seed = self._resolve_default_seed()
        self._active_weather_data_path: str | None = None

        self._latest_observation: Dict[str, Any] | None = None
        self._latest_info: Dict[str, Any] = {}

    def reset(
        self,
        seed: int | None = None,
        num_households: int | None = None,
        max_episode_steps: int | None = None,
        weather_data_path: str | None = None,
    ) -> Dict[str, Any]:
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
            effective_weather_data_path = self._resolve_weather_data_path(weather_data_path)

            self._env = self._create_env(
                seed=effective_seed,
                num_households=effective_households,
                max_episode_steps=effective_max_steps,
                weather_data_path=effective_weather_data_path,
            )

            self._seed = effective_seed
            self._episode_id += 1
            self._step_count = 0
            self._active_weather_data_path = effective_weather_data_path
            self._history.clear()
            self._latest_info = {}
            self._latest_observation = self._env.reset()

            return self._build_state_payload(include_topology=True)

    def step(
        self,
        house_actions: List[List[float]] | None = None,
        market_action: int | None = None,
        use_autopilot: bool = True,
    ) -> Dict[str, Any]:
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

            self._history.append(self._build_step_record(observation, reward, done, info))

            payload = self._build_state_payload(include_topology=False)
            payload["step_result"] = {
                "reward": float(reward),
                "done": bool(done),
                "info": self._to_jsonable(info),
            }
            return payload

    def run(
        self,
        steps: int,
        use_autopilot: bool = True,
        market_action: int | None = None,
    ) -> Dict[str, Any]:
        """Run multiple steps and return trajectory plus latest state snapshot."""
        if steps <= 0:
            raise ValueError("steps must be greater than zero")

        trajectory: List[Dict[str, Any]] = []
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

    def get_state(self, include_topology: bool = True) -> Dict[str, Any]:
        """Return the latest state; lazily initializes the session when missing."""
        with self._lock:
            self._ensure_env()
            return self._build_state_payload(include_topology=include_topology)

    def get_metrics(self) -> Dict[str, Any]:
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

            rewards = np.asarray([record.reward for record in self._history], dtype=np.float64)
            prices = np.asarray([record.price for record in self._history], dtype=np.float64)
            demand = np.asarray([record.demand for record in self._history], dtype=np.float64)
            supply = np.asarray([record.supply for record in self._history], dtype=np.float64)
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

    def get_history(self, limit: int = 300) -> List[Dict[str, Any]]:
        """Return the latest trajectory points for chart rendering."""
        with self._lock:
            capped = max(1, min(int(limit), len(self._history)))
            return [asdict(record) for record in list(self._history)[-capped:]]

    def get_csv_schemas(self) -> Dict[str, Any]:
        """Return expected CSV role schemas and current runtime support status."""
        return {
            "weather": {
                "description": "Time-series weather features consumed by GridEnv via load_weather_data.",
                "required_all": [],
                "required_any": ["temperature", "solar_irradiance", "wind_speed", "humidity"],
                "recommended": ["temperature", "solar_irradiance", "wind_speed", "humidity"],
                "runtime_supported_now": True,
                "runtime_usage": "Pass weather_data_path when resetting simulation.",
            },
            "household": {
                "description": "Household demand series consumed by load_household_data.",
                "required_all": ["consumption"],
                "required_any": [],
                "recommended": ["consumption"],
                "runtime_supported_now": False,
                "runtime_usage": "Supported by data loaders; GridEnv runtime integration is planned.",
            },
            "market": {
                "description": "Market time-series consumed by load_market_data / MarketEnv.",
                "required_all": ["supply", "demand", "price"],
                "required_any": [],
                "recommended": ["supply", "demand", "price"],
                "runtime_supported_now": False,
                "runtime_usage": "Supported by data loaders; GridEnv runtime integration is planned.",
            },
        }

    def profile_csv(
        self,
        file_path: str,
        role: str = "auto",
        preview_rows: int = 5,
    ) -> Dict[str, Any]:
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

        columns = [str(column) for column in frame.columns]
        normalized_columns = {column.strip().lower() for column in columns}
        schemas = self.get_csv_schemas()

        compatibility: Dict[str, Any] = {}
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
                normalized_columns.intersection(required_all.union(required_any).union(recommended))
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
                "runtime_supported_now": bool(schema.get("runtime_supported_now", False)),
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

        preview_rows_payload = (
            frame.head(preview_count)
            .where(pd.notnull(frame), None)
            .to_dict(orient="records")
        )
        null_counts = {
            str(column): int(frame[column].isna().sum())
            for column in frame.columns
        }

        recommendation = self._build_csv_usage_recommendation(
            selected_role=selected_role,
            compatibility=selected_compatibility,
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
                selected_role == "weather"
                and selected_compatibility.get("compatible", False)
                and selected_compatibility.get("runtime_supported_now", False)
            ),
            "usage_recommendation": recommendation,
        }

    def derive_weather_csv(
        self,
        file_path: str,
        solar_column: str,
        wind_column: str,
        timestamp_column: str | None = None,
        temperature_column: str | None = None,
        humidity_column: str | None = None,
        output_path: str | None = None,
        normalize_signals: bool = True,
    ) -> Dict[str, Any]:
        """Generate a GridEnv-compatible weather CSV from a wide source timeseries file."""
        source_path = self._resolve_file_path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")
        if source_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported")

        required_columns = [solar_column, wind_column]
        optional_columns = [
            timestamp_column,
            temperature_column,
            humidity_column,
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
                "Missing required columns in source CSV: "
                + ", ".join(missing_columns)
            )

        solar_series = pd.to_numeric(frame[solar_column], errors="coerce").fillna(0.0)
        wind_series = pd.to_numeric(frame[wind_column], errors="coerce").fillna(0.0)

        solar_scale = float(max(solar_series.max(), 1.0))
        wind_scale = float(max(wind_series.max(), 1.0))
        if normalize_signals:
            solar_output = np.clip(solar_series / solar_scale, 0.0, 1.0)
            wind_output = np.clip(wind_series / wind_scale, 0.0, 1.0)
        else:
            solar_output = solar_series
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

        weather_frame = pd.DataFrame(
            {
                "temperature": temperature_output,
                "solar_irradiance": np.asarray(solar_output, dtype=np.float32),
                "wind_speed": np.asarray(wind_output, dtype=np.float32),
                "humidity": humidity_output,
            }
        )

        if timestamp_column and timestamp_column in frame.columns:
            weather_frame.insert(
                0,
                "utc_timestamp",
                frame[timestamp_column].astype(str),
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
                "solar_irradiance": solar_column,
                "wind_speed": wind_column,
                "temperature": temperature_column,
                "humidity": humidity_column,
                "utc_timestamp": timestamp_column,
            },
            "normalization": {
                "enabled": bool(normalize_signals),
                "solar_scale": solar_scale,
                "wind_scale": wind_scale,
            },
            "usage_recommendation": (
                "Use the output_file_path as weather_data_path in /simulation/reset "
                "or in the dashboard CSV controls."
            ),
        }

    def _create_env(
        self,
        seed: int,
        num_households: int,
        max_episode_steps: int,
        weather_data_path: str,
    ) -> Any:
        # Lazy import avoids importing heavy model stack at app startup.
        from app.envs.grid_env import GridEnv

        paths = self._resolve_data_paths()
        env = GridEnv(
            grid_topology_file=paths["grid_topology"],
            weather_file=weather_data_path,
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
        observation: Dict[str, Any],
        house_actions: List[List[float]] | None,
        market_action: int | None,
        use_autopilot: bool,
    ) -> Dict[str, Any]:
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

    def _build_autopilot_actions(self, observation: Dict[str, Any]) -> np.ndarray:
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
                0.65 if (battery_level > (0.60 * max_battery) and price > default_price) else 0.10
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

    def _build_state_payload(self, include_topology: bool) -> Dict[str, Any]:
        assert self._env is not None
        assert self._latest_observation is not None

        payload: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "seed": self._seed,
            "step": self._step_count,
            "data_sources": {
                "weather_data": self._active_weather_data_path
                if self._active_weather_data_path is not None
                else self._resolve_data_paths()["weather_data"],
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

    def _build_topology_payload(self) -> Dict[str, Any]:
        assert self._env is not None

        nodes = []
        for node_id, attrs in self._env.graph.nodes(data=True):
            nodes.append(
                {
                    "id": int(node_id),
                    "type": str(attrs.get("type", "unknown")),
                    "label": str(attrs.get("label", node_id)),
                }
            )

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
        }

    def _build_step_record(
        self,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> StepRecord:
        house_states = np.asarray(observation["house_states"], dtype=np.float32)
        market_snapshot = info.get("market_snapshot", {}) if isinstance(info, dict) else {}

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
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            reward=float(reward),
            done=bool(done),
            supply=supply,
            demand=demand,
            price=price,
            grid_import=grid_import,
            renewable_utilization=renewable_utilization,
        )

    def _build_trajectory_point(self) -> Dict[str, Any]:
        if not self._history:
            return {
                "step": self._step_count,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
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
        repro_seed = reproducibility.get("seed", None) if hasattr(reproducibility, "get") else None
        if repro_seed is not None:
            return int(repro_seed)

        return int(config.get("seed", 42))

    @staticmethod
    def _resolve_default_households() -> int:
        env_cfg = config.get("env", {})
        env_value = env_cfg.get("num_households", None) if hasattr(env_cfg, "get") else None
        return int(config.get("num_households", env_value if env_value is not None else 10))

    @staticmethod
    def _resolve_default_max_steps() -> int:
        env_cfg = config.get("env", {})
        env_value = env_cfg.get("max_episode_steps", None) if hasattr(env_cfg, "get") else None
        training_steps = config.get("training_steps", None)
        if training_steps is not None:
            return int(training_steps)
        return int(env_value if env_value is not None else 1000)

    @staticmethod
    def _resolve_default_max_battery() -> float:
        env_cfg = config.get("env", {})
        env_value = env_cfg.get("max_battery", None) if hasattr(env_cfg, "get") else None
        top_level = config.get("max_battery", None)
        if top_level is not None:
            return float(top_level)
        return float(env_value if env_value is not None else 10.0)

    @staticmethod
    def _resolve_default_price() -> float:
        market_cfg = config.get("market", {})
        market_value = market_cfg.get("default_price", None) if hasattr(market_cfg, "get") else None
        return float(market_value if market_value is not None else 0.3)

    @staticmethod
    def _resolve_price_max() -> float:
        market_cfg = config.get("market", {})
        market_value = market_cfg.get("price_max", None) if hasattr(market_cfg, "get") else None
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

    @staticmethod
    def _resolve_file_path(file_path: str) -> Path:
        raw_path = Path(file_path).expanduser()
        if raw_path.is_absolute():
            return raw_path

        backend_root = Path(__file__).resolve().parents[2]
        workspace_root = backend_root.parent

        backend_candidate = (backend_root / raw_path).resolve()
        if backend_candidate.exists():
            return backend_candidate

        workspace_candidate = (workspace_root / raw_path).resolve()
        if workspace_candidate.exists():
            return workspace_candidate

        return backend_candidate

    @staticmethod
    def _count_csv_rows(file_path: Path) -> int:
        row_count = 0
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            # Exclude header row from the total timestep count.
            for _ in handle:
                row_count += 1

        return max(row_count - 1, 0)

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
        compatibility: Dict[str, Any],
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
                    "Schema matches household loader expectations. It can be consumed by "
                    "load_household_data and is ready for upcoming GridEnv household-data wiring."
                )
            return "Add a consumption column to make this household-compatible."

        if selected_role == "market":
            if compatibility.get("compatible", False):
                return (
                    "Schema matches market loader expectations. It can be consumed by load_market_data "
                    "and is ready for upcoming GridEnv market-data wiring."
                )
            return "Add supply, demand, and price columns to make this market-compatible."

        return (
            "Could not confidently infer a role. Choose weather/household/market explicitly "
            "to validate against a specific schema."
        )

    @staticmethod
    def _resolve_data_paths() -> Dict[str, str]:
        defaults = {
            "grid_topology": "data/grid_topology/sample_grid.json",
            "weather_data": "data/weather_data/sample_weather.csv",
        }
        data_paths = config.get("data_paths", {})
        if hasattr(data_paths, "get"):
            return {
                "grid_topology": str(data_paths.get("grid_topology", defaults["grid_topology"])),
                "weather_data": str(data_paths.get("weather_data", defaults["weather_data"])),
            }
        return defaults


simulation_service = SimulationService()
