"""Integration tests for Phase 2 PPO training endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_training_run_returns_progress_artifact() -> None:
    client = TestClient(app)

    response = client.post(
        "/training/ppo/run",
        json={
            "episodes": 3,
            "steps_per_episode": 8,
            "eval_episodes": 2,
            "seed": 2026,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["run_id"].startswith("ppo-")
    assert "training" in payload
    assert "comparison" in payload

    training = payload["training"]
    reward_curve = training["reward_curve"]
    assert len(reward_curve) == 3
    assert training["episodes"] == 3
    assert training["steps_per_episode"] == 8

    comparison = payload["comparison"]
    assert "ppo" in comparison
    assert "rule" in comparison
    assert "deltas" in comparison


def test_training_latest_and_reward_curve_endpoints() -> None:
    client = TestClient(app)

    run_response = client.post(
        "/training/ppo/run",
        json={
            "episodes": 2,
            "steps_per_episode": 6,
            "eval_episodes": 2,
            "seed": 77,
        },
    )
    assert run_response.status_code == 200
    run_payload = run_response.json()

    latest_response = client.get("/training/ppo/latest")
    assert latest_response.status_code == 200
    latest_payload = latest_response.json()
    assert latest_payload["run_id"] == run_payload["run_id"]

    curve_response = client.get("/training/ppo/reward-curve")
    assert curve_response.status_code == 200
    curve_payload = curve_response.json()
    assert curve_payload["run_id"] == run_payload["run_id"]
    assert len(curve_payload["reward_curve"]) == 2


def test_training_comparison_endpoint_returns_baseline_delta() -> None:
    client = TestClient(app)

    # Ensure at least one trained model exists in service state.
    warmup_response = client.post(
        "/training/ppo/run",
        json={
            "episodes": 2,
            "steps_per_episode": 6,
            "eval_episodes": 2,
            "seed": 88,
        },
    )
    assert warmup_response.status_code == 200

    compare_response = client.post(
        "/training/ppo/compare",
        json={
            "episodes": 2,
            "steps_per_episode": 6,
            "seed": 99,
        },
    )
    assert compare_response.status_code == 200
    compare_payload = compare_response.json()

    assert "ppo" in compare_payload
    assert "rule" in compare_payload
    assert "deltas" in compare_payload
    assert "reward_delta" in compare_payload["deltas"]
