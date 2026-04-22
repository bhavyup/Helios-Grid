"""Integration tests for simulation lifecycle API routes."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def _build_zero_actions(num_households: int) -> list[list[float]]:
    return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(num_households)]


def test_simulation_reset_returns_state_with_topology() -> None:
    client = TestClient(app)

    response = client.post(
        "/simulation/reset",
        json={"seed": 123, "num_households": 4, "max_episode_steps": 32},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["seed"] == 123
    assert payload["step"] == 0
    assert "observation" in payload
    assert "topology" in payload
    assert payload["topology"]["node_count"] >= 1
    assert len(payload["observation"]["house_states"]) == 4


def test_simulation_step_is_deterministic_for_same_seed_and_actions() -> None:
    client = TestClient(app)

    reset_payload = {"seed": 777, "num_households": 5, "max_episode_steps": 24}
    fixed_actions = _build_zero_actions(num_households=5)

    first_reset = client.post("/simulation/reset", json=reset_payload)
    assert first_reset.status_code == 200

    first_step = client.post(
        "/simulation/step",
        json={
            "house_actions": fixed_actions,
            "market_action": 1,
            "use_autopilot": False,
        },
    )
    assert first_step.status_code == 200
    first_payload = first_step.json()

    second_reset = client.post("/simulation/reset", json=reset_payload)
    assert second_reset.status_code == 200

    second_step = client.post(
        "/simulation/step",
        json={
            "house_actions": fixed_actions,
            "market_action": 1,
            "use_autopilot": False,
        },
    )
    assert second_step.status_code == 200
    second_payload = second_step.json()

    assert first_payload["step_result"]["reward"] == second_payload["step_result"]["reward"]
    assert first_payload["trajectory_point"]["price"] == second_payload["trajectory_point"]["price"]

    first_house_states = np.asarray(
        first_payload["observation"]["house_states"],
        dtype=np.float32,
    )
    second_house_states = np.asarray(
        second_payload["observation"]["house_states"],
        dtype=np.float32,
    )
    assert np.allclose(first_house_states, second_house_states)


def test_run_history_and_metrics_endpoints_are_consistent() -> None:
    client = TestClient(app)

    reset_response = client.post(
        "/simulation/reset",
        json={"seed": 101, "num_households": 6, "max_episode_steps": 40},
    )
    assert reset_response.status_code == 200

    run_response = client.post(
        "/simulation/run",
        json={"steps": 5, "use_autopilot": True},
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()

    trajectory = run_payload["trajectory"]
    assert 1 <= len(trajectory) <= 5

    metrics_response = client.get("/simulation/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["steps_executed"] >= len(trajectory)

    history_response = client.get("/simulation/history", params={"limit": 2})
    assert history_response.status_code == 200
    history = history_response.json()
    assert 1 <= len(history) <= 2


def test_csv_profile_endpoint_reports_weather_compatibility(tmp_path) -> None:
    client = TestClient(app)

    weather_csv = tmp_path / "weather_input.csv"
    weather_csv.write_text(
        "temperature,solar_irradiance,wind_speed,humidity\n"
        "24.1,0.65,3.2,41\n"
        "25.0,0.72,3.6,39\n",
        encoding="utf-8",
    )

    response = client.post(
        "/simulation/data/profile",
        json={"file_path": str(weather_csv), "role": "weather", "preview_rows": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_role"] == "weather"
    assert payload["can_use_now"] is True
    assert payload["compatibility"]["weather"]["compatible"] is True
    assert payload["rows"] == 2


def test_reset_accepts_custom_weather_csv_path(tmp_path) -> None:
    client = TestClient(app)

    weather_csv = tmp_path / "custom_weather.csv"
    weather_csv.write_text(
        "temperature,solar_irradiance\n"
        "20.0,0.40\n"
        "21.2,0.52\n"
        "22.5,0.63\n",
        encoding="utf-8",
    )

    response = client.post(
        "/simulation/reset",
        json={
            "seed": 2026,
            "num_households": 4,
            "max_episode_steps": 12,
            "weather_data_path": str(weather_csv),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_sources"]["weather_data"] == str(weather_csv)
    assert payload["seed"] == 2026
    assert payload["step"] == 0


def test_derive_weather_endpoint_generates_reset_ready_csv(tmp_path) -> None:
    client = TestClient(app)

    source_csv = tmp_path / "source_timeseries.csv"
    source_csv.write_text(
        "utc_timestamp,DE_solar_generation_actual,DE_wind_onshore_generation_actual\n"
        "2015-01-01T00:00:00Z,200,500\n"
        "2015-01-01T01:00:00Z,300,700\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "derived_weather.csv"
    derive_response = client.post(
        "/simulation/data/derive-weather",
        json={
            "file_path": str(source_csv),
            "solar_column": "DE_solar_generation_actual",
            "wind_column": "DE_wind_onshore_generation_actual",
            "timestamp_column": "utc_timestamp",
            "output_path": str(output_csv),
            "normalize_signals": True,
        },
    )

    assert derive_response.status_code == 200
    derive_payload = derive_response.json()
    assert derive_payload["rows"] == 2
    assert derive_payload["output_file_path"] == str(output_csv)

    reset_response = client.post(
        "/simulation/reset",
        json={
            "seed": 42,
            "num_households": 3,
            "max_episode_steps": 8,
            "weather_data_path": str(output_csv),
        },
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["data_sources"]["weather_data"] == str(output_csv)
