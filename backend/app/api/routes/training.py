"""Training routes exposing PPO proof-of-life progress artifacts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.training_service import training_service

router = APIRouter(prefix="/training", tags=["training"])


class PPOTrainRequest(BaseModel):
    """Request payload for PPO training."""

    episodes: int = Field(default=20, ge=1, le=500)
    steps_per_episode: int = Field(default=24, ge=1, le=500)
    eval_episodes: int = Field(default=5, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0)
    learning_rate: float | None = Field(default=None, gt=0.0, lt=1.0)
    hidden_dim: int | None = Field(default=None, ge=32, le=1024)
    clip_epsilon: float | None = Field(default=None, gt=0.0, lt=1.0)


class PPOComparisonRequest(BaseModel):
    """Request payload for PPO vs rule comparison."""

    episodes: int = Field(default=8, ge=1, le=200)
    steps_per_episode: int = Field(default=24, ge=1, le=500)
    seed: int | None = Field(default=None, ge=0)


@router.post("/ppo/run")
def run_ppo_training(request: PPOTrainRequest) -> dict[str, Any]:
    """Train PPO and return run artifact with progress and comparison."""
    return training_service.train_ppo(
        episodes=request.episodes,
        steps_per_episode=request.steps_per_episode,
        eval_episodes=request.eval_episodes,
        seed=request.seed,
        learning_rate=request.learning_rate,
        hidden_dim=request.hidden_dim,
        clip_epsilon=request.clip_epsilon,
    )


@router.get("/ppo/latest")
def get_latest_ppo_training() -> dict[str, Any]:
    """Return latest PPO training artifact."""
    return training_service.get_latest_run()


@router.post("/ppo/compare")
def compare_ppo_and_rule(request: PPOComparisonRequest) -> dict[str, Any]:
    """Run PPO vs rule baseline comparison."""
    return training_service.compare_rule_vs_ppo(
        episodes=request.episodes,
        steps_per_episode=request.steps_per_episode,
        seed=request.seed,
    )


@router.get("/ppo/comparison/latest")
def get_latest_ppo_comparison() -> dict[str, Any]:
    """Return latest comparison artifact."""
    return training_service.get_latest_comparison()


@router.get("/ppo/reward-curve")
def get_latest_reward_curve() -> dict[str, Any]:
    """Return reward curve from latest PPO training run."""
    return training_service.get_latest_reward_curve()
