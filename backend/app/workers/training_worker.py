from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import ray

from app.infrastructure.mlflow_tracker import log_training_run, mlflow_enabled
from app.infrastructure.model_registry import store_training_artifacts
from app.infrastructure.monitoring import record_training_job_failed
from app.models.ppo_agent import PPOAgent


def execute_ppo_training(
    run_id: str,
    created_at: str | None,
    episodes: int,
    steps_per_episode: int,
    num_envs: int,
    eval_episodes: int,
    seed: int | None,
    learning_rate: float | None,
    hidden_dim: int | None,
    clip_epsilon: float | None,
) -> dict[str, Any]:
    agent = PPOAgent(
        seed=seed,
        learning_rate=learning_rate,
        hidden_dim=128 if hidden_dim is None else int(hidden_dim),
        clip_epsilon=clip_epsilon,
    )

    training = agent.train(
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        num_envs=num_envs,
        seed=seed,
    )

    comparison = agent.compare_against_rule(
        episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        seed=None if seed is None else int(seed) + 10_000,
    )

    timestamp = created_at or datetime.now(tz=UTC).isoformat()
    mlflow_run_id = None
    registry_artifacts = None
    if mlflow_enabled():
        params = {
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "num_envs": num_envs,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "clip_epsilon": clip_epsilon,
        }
        mlflow_run_id = log_training_run(
            run_name=run_id,
            params=params,
            training_summary=training,
            comparison=comparison,
            model=agent.model,
        )
        registry_artifacts = store_training_artifacts(
            run_id=run_id,
            created_at=timestamp,
            params=params,
            training_summary=training,
            comparison=comparison,
            model=agent.model,
        )

    return {
        "run_id": run_id,
        "created_at": timestamp,
        "training": training,
        "comparison": comparison,
        "mlflow_run_id": mlflow_run_id,
        "model_registry": registry_artifacts,
    }


@ray.remote
def run_ppo_training(
    run_id: str,
    created_at: str | None,
    episodes: int,
    steps_per_episode: int,
    num_envs: int,
    eval_episodes: int,
    seed: int | None,
    learning_rate: float | None,
    hidden_dim: int | None,
    clip_epsilon: float | None,
) -> dict[str, Any]:
    try:
        return execute_ppo_training(
            run_id=run_id,
            created_at=created_at,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            num_envs=num_envs,
            eval_episodes=eval_episodes,
            seed=seed,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            clip_epsilon=clip_epsilon,
        )
    except Exception:
        record_training_job_failed()
        raise


__all__ = ["execute_ppo_training", "run_ppo_training"]
