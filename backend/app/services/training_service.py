"""Training orchestration service for PPO progress and baseline comparison."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict
from uuid import uuid4

from app.models.ppo_agent import PPOAgent


class TrainingService:
    """Coordinates PPO training runs and caches latest experiment artifacts."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._agent: PPOAgent | None = None
        self._latest_run: Dict[str, Any] = {}
        self._latest_comparison: Dict[str, Any] = {}

    def train_ppo(
        self,
        episodes: int,
        steps_per_episode: int,
        eval_episodes: int,
        seed: int | None = None,
        learning_rate: float | None = None,
        hidden_dim: int | None = None,
        clip_epsilon: float | None = None,
    ) -> Dict[str, Any]:
        """Run PPO proof-of-life training and return rich progress artifact."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than zero")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be greater than zero")

        with self._lock:
            self._agent = PPOAgent(
                seed=seed,
                learning_rate=learning_rate,
                hidden_dim=128 if hidden_dim is None else int(hidden_dim),
                clip_epsilon=clip_epsilon,
            )
            training = self._agent.train(
                episodes=episodes,
                steps_per_episode=steps_per_episode,
                seed=seed,
            )

            comparison = self._agent.compare_against_rule(
                episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                seed=None if seed is None else int(seed) + 10_000,
            )

            run_id = f"ppo-{uuid4()}"
            now = datetime.now(tz=timezone.utc).isoformat()
            result = {
                "run_id": run_id,
                "created_at": now,
                "training": training,
                "comparison": comparison,
            }
            self._latest_run = result
            self._latest_comparison = {
                "run_id": run_id,
                "created_at": now,
                **comparison,
            }
            return result

    def compare_rule_vs_ppo(
        self,
        episodes: int,
        steps_per_episode: int,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """Evaluate latest PPO policy against rule baseline."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than zero")

        with self._lock:
            if self._agent is None:
                self._agent = PPOAgent(seed=seed)
                self._agent.train(
                    episodes=6,
                    steps_per_episode=steps_per_episode,
                    seed=seed,
                )

            comparison = self._agent.compare_against_rule(
                episodes=episodes,
                steps_per_episode=steps_per_episode,
                seed=seed,
            )
            self._latest_comparison = {
                "run_id": self._latest_run.get("run_id", "bootstrap"),
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                **comparison,
            }
            return dict(self._latest_comparison)

    def get_latest_run(self) -> Dict[str, Any]:
        """Return latest training payload or a bootstrap status object."""
        with self._lock:
            if not self._latest_run:
                return {
                    "status": "idle",
                    "message": "No PPO training run has been executed yet.",
                }
            return dict(self._latest_run)

    def get_latest_comparison(self) -> Dict[str, Any]:
        """Return latest comparison payload or status object."""
        with self._lock:
            if not self._latest_comparison:
                return {
                    "status": "idle",
                    "message": "No PPO comparison has been executed yet.",
                }
            return dict(self._latest_comparison)

    def get_latest_reward_curve(self) -> Dict[str, Any]:
        """Return reward curve artifact from latest training run."""
        with self._lock:
            if not self._latest_run:
                return {
                    "status": "idle",
                    "message": "No reward curve available. Run PPO training first.",
                    "reward_curve": [],
                }

            training = self._latest_run.get("training", {})
            reward_curve = training.get("reward_curve", [])
            return {
                "run_id": self._latest_run.get("run_id"),
                "created_at": self._latest_run.get("created_at"),
                "reward_curve": reward_curve,
                "episodes": training.get("episodes", len(reward_curve)),
            }


training_service = TrainingService()
