"""Integration tests for simulation lifecycle API routes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def _build_zero_actions(num_households: int) -> list[list[float]]:
    return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(num_households)]


def test_simulation_reset_returns_state_with_topology(auth_headers) -> None:
    client = TestClient(app)

    response = client.post(
        "/simulation/reset",
        json={"seed": 123, "num_households": 4, "max_episode_steps": 32},
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["seed"] == 123
    assert payload["step"] == 0
    assert "observation" in payload
    assert "topology" in payload
    assert payload["topology"]["layout"] == "neighborhood-grid"
    assert payload["topology"]["node_count"] >= 1
    assert "x" in payload["topology"]["nodes"][0]
    assert "y" in payload["topology"]["nodes"][0]
    assert len(payload["observation"]["house_states"]) == 4


def test_simulation_step_is_deterministic_for_same_seed_and_actions(
    auth_headers,
) -> None:
    client = TestClient(app)

    reset_payload = {"seed": 777, "num_households": 5, "max_episode_steps": 24}
    fixed_actions = _build_zero_actions(num_households=5)

    first_reset = client.post(
        "/simulation/reset", json=reset_payload, headers=auth_headers
    )
    assert first_reset.status_code == 200

    first_step = client.post(
        "/simulation/step",
        json={
            "house_actions": fixed_actions,
            "market_action": 1,
            "use_autopilot": False,
        },
        headers=auth_headers,
    )
    assert first_step.status_code == 200
    first_payload = first_step.json()

    second_reset = client.post(
        "/simulation/reset", json=reset_payload, headers=auth_headers
    )
    assert second_reset.status_code == 200

    second_step = client.post(
        "/simulation/step",
        json={
            "house_actions": fixed_actions,
            "market_action": 1,
            "use_autopilot": False,
        },
        headers=auth_headers,
    )
    assert second_step.status_code == 200
    second_payload = second_step.json()

    assert (
        first_payload["step_result"]["reward"]
        == second_payload["step_result"]["reward"]
    )
    assert (
        first_payload["trajectory_point"]["price"]
        == second_payload["trajectory_point"]["price"]
    )

    first_house_states = np.asarray(
        first_payload["observation"]["house_states"],
        dtype=np.float32,
    )
    second_house_states = np.asarray(
        second_payload["observation"]["house_states"],
        dtype=np.float32,
    )
    assert np.allclose(first_house_states, second_house_states)


def test_run_history_and_metrics_endpoints_are_consistent(auth_headers) -> None:
    client = TestClient(app)

    reset_response = client.post(
        "/simulation/reset",
        json={"seed": 101, "num_households": 6, "max_episode_steps": 40},
        headers=auth_headers,
    )
    assert reset_response.status_code == 200

    run_response = client.post(
        "/simulation/run",
        json={"steps": 5, "use_autopilot": True},
        headers=auth_headers,
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()

    trajectory = run_payload["trajectory"]
    assert 1 <= len(trajectory) <= 5

    metrics_response = client.get("/simulation/metrics", headers=auth_headers)
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["steps_executed"] >= len(trajectory)

    history_response = client.get(
        "/simulation/history",
        params={"limit": 2},
        headers=auth_headers,
    )
    assert history_response.status_code == 200
    history = history_response.json()
    assert 1 <= len(history) <= 2


def test_csv_profile_endpoint_reports_weather_compatibility(auth_headers) -> None:
    client = TestClient(app)
    artifacts_dir = Path.cwd() / "test_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    weather_csv = artifacts_dir / "weather_input.csv"
    weather_csv.write_text(
        "temperature,solar_irradiance,wind_speed,humidity\n"
        "24.1,0.65,3.2,41\n"
        "25.0,0.72,3.6,39\n",
        encoding="utf-8",
    )

    response = client.post(
        "/simulation/data/profile",
        json={"file_path": str(weather_csv), "role": "weather", "preview_rows": 2},
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_role"] == "weather"
    assert payload["can_use_now"] is True
    assert payload["compatibility"]["weather"]["compatible"] is True
    assert payload["rows"] == 2


def test_reset_accepts_custom_weather_csv_path(auth_headers) -> None:
    client = TestClient(app)
    artifacts_dir = Path.cwd() / "test_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    weather_csv = artifacts_dir / "custom_weather.csv"
    weather_csv.write_text(
        "utc_timestamp,temperature,solar_irradiance\n"
        "2026-01-01T12:00:00Z,20.0,0.40\n"
        "2026-01-01T12:15:00Z,21.2,0.52\n"
        "2026-01-01T12:30:00Z,22.5,0.63\n",
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
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_sources"]["weather_data"] == str(weather_csv)
    assert payload["seed"] == 2026
    assert payload["step"] == 0

    step_response = client.post(
        "/simulation/step",
        json={"use_autopilot": True},
        headers=auth_headers,
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["step_result"]["info"]["weather"]["utc_timestamp"]


def test_derive_weather_endpoint_generates_reset_ready_csv(auth_headers) -> None:
    client = TestClient(app)
    artifacts_dir = Path.cwd() / "test_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    source_csv = artifacts_dir / "source_timeseries.csv"
    source_csv.write_text(
        "utc_timestamp,DE_solar_generation_actual,DE_wind_onshore_generation_actual\n"
        "2015-01-01T00:00:00Z,200,500\n"
        "2015-01-01T01:00:00Z,300,700\n",
        encoding="utf-8",
    )

    output_csv = artifacts_dir / "derived_weather.csv"
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
        headers=auth_headers,
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
        headers=auth_headers,
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["data_sources"]["weather_data"] == str(output_csv)


def test_step_emits_p2p_trades_in_market_snapshot(auth_headers) -> None:
    client = TestClient(app)

    reset = client.post(
        "/simulation/reset",
        json={"seed": 42, "num_households": 16, "max_episode_steps": 10},
        headers=auth_headers,
    )
    assert reset.status_code == 200

    step = client.post(
        "/simulation/step",
        json={"use_autopilot": True, "market_action": 1},
        headers=auth_headers,
    )
    assert step.status_code == 200
    payload = step.json()
    market_snapshot = payload["latest_info"]["market_snapshot"]

    assert "p2p_orders" in market_snapshot
    assert "p2p_trades" in market_snapshot
    assert isinstance(market_snapshot["p2p_orders"], list)
    assert isinstance(market_snapshot["p2p_trades"], list)


def test_topology_layout_scales_for_64_households(auth_headers) -> None:
    client = TestClient(app)

    response = client.post(
        "/simulation/reset",
        json={"seed": 123, "num_households": 64, "max_episode_steps": 8},
        headers=auth_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    topology = payload["topology"]

    nodes = topology["nodes"]
    households = [n for n in nodes if n.get("type") == "household"]

    assert len(households) == 64
    assert topology.get("layout") == "neighborhood-grid"
    assert topology.get("bounds") is not None

    bounds = topology["bounds"]
    unique_positions = set()
    for h in households:
        x, y = h.get("x"), h.get("y")
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        unique_positions.add((round(float(x), 3), round(float(y), 3)))

        # within bounds
        assert bounds["min_x"] <= x <= bounds["max_x"]
        assert bounds["min_y"] <= y <= bounds["max_y"]

    # should be fully unique for 64-house layout
    assert len(unique_positions) >= 60
