from __future__ import annotations

import inspect

from app.workers import simulation_worker, training_worker


def _unwrap_remote_call(callable_obj):
    for attribute_name in ("__wrapped__", "_function", "func"):
        original = getattr(callable_obj, attribute_name, None)
        if callable(original):
            return original
    unwrapped = inspect.unwrap(callable_obj)
    return unwrapped if callable(unwrapped) else callable_obj


def test_simulation_worker_runs_service_reset_and_run(monkeypatch):
    calls: dict[str, object] = {}

    class FakeSimulationService:
        def reset(self, **kwargs):
            calls["reset"] = kwargs
            return {"step": 0}

        def run(self, **kwargs):
            calls["run"] = kwargs
            return {
                "state": {"step": 2},
                "trajectory": [{"step": 1}, {"step": 2}],
                "metrics": {"steps_executed": 2},
            }

    monkeypatch.setattr(simulation_worker, "SimulationService", FakeSimulationService)
    worker_fn = _unwrap_remote_call(simulation_worker.run_simulation)

    result = worker_fn(
        steps=2,
        seed=11,
        num_households=4,
        max_episode_steps=16,
        weather_data_path="/tmp/weather.csv",
        use_autopilot=False,
        market_action=1,
    )

    assert calls["reset"] == {
        "seed": 11,
        "num_households": 4,
        "max_episode_steps": 16,
        "weather_data_path": "/tmp/weather.csv",
    }
    assert calls["run"] == {"steps": 2, "use_autopilot": False, "market_action": 1}
    assert result["metrics"]["steps_executed"] == 2
    assert len(result["trajectory"]) == 2


def test_training_worker_returns_training_payload_without_mlflow(monkeypatch):
    calls: dict[str, object] = {}

    class FakeModel:
        def state_dict(self):
            return {"weights": 1}

    class FakeAgent:
        def __init__(
            self, seed=None, learning_rate=None, hidden_dim=None, clip_epsilon=None
        ):
            calls["init"] = {
                "seed": seed,
                "learning_rate": learning_rate,
                "hidden_dim": hidden_dim,
                "clip_epsilon": clip_epsilon,
            }
            self.model = FakeModel()

        def train(self, episodes, steps_per_episode, num_envs, seed):
            calls["train"] = {
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "num_envs": num_envs,
                "seed": seed,
            }
            return {
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "reward_curve": [1.0, 2.0],
            }

        def compare_against_rule(self, episodes, steps_per_episode, seed):
            calls["compare"] = {
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "seed": seed,
            }
            return {"reward_delta": 0.75, "baseline_reward": 1.25}

    monkeypatch.setattr(training_worker, "PPOAgent", FakeAgent)
    monkeypatch.setattr(training_worker, "mlflow_enabled", lambda: False)
    monkeypatch.setattr(
        training_worker, "log_training_run", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        training_worker, "store_training_artifacts", lambda *args, **kwargs: None
    )
    worker_fn = _unwrap_remote_call(training_worker.run_ppo_training)

    result = worker_fn(
        run_id="ppo-test-1",
        created_at="2026-05-26T00:00:00+00:00",
        episodes=3,
        steps_per_episode=8,
        num_envs=2,
        eval_episodes=4,
        seed=7,
        learning_rate=0.001,
        hidden_dim=None,
        clip_epsilon=0.2,
    )

    assert calls["init"] == {
        "seed": 7,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "clip_epsilon": 0.2,
    }
    assert calls["train"] == {
        "episodes": 3,
        "steps_per_episode": 8,
        "num_envs": 2,
        "seed": 7,
    }
    assert calls["compare"] == {"episodes": 4, "steps_per_episode": 8, "seed": 10007}
    assert result["run_id"] == "ppo-test-1"
    assert result["created_at"] == "2026-05-26T00:00:00+00:00"
    assert result["training"]["reward_curve"] == [1.0, 2.0]
    assert result["comparison"]["reward_delta"] == 0.75
    assert result["mlflow_run_id"] is None
    assert result["model_registry"] is None
