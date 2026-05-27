"""Training orchestration service for PPO progress and baseline comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from threading import RLock
from typing import Any, Dict
from uuid import uuid4

import ray  # type: ignore[import-not-found]

from app.infrastructure.ray_client import init_ray
from app.infrastructure.monitoring import (
    record_training_duration,
    record_training_job_completed,
    record_training_job_failed,
    record_training_job_started,
)
from app.models.ppo_agent import PPOAgent
from app.workers.training_worker import run_ppo_training


class TrainingService:
    """Coordinates PPO training runs and caches latest experiment artifacts."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._agent: PPOAgent | None = None
        self._latest_run: Dict[str, Any] = {}
        self._latest_comparison: Dict[str, Any] = {}
        self._jobs: Dict[str, Any] = {}
        self._job_results: Dict[str, Dict[str, Any]] = {}
        self._latest_job_id: str | None = None

    def train_ppo(
        self,
        episodes: int,
        steps_per_episode: int,
        eval_episodes: int,
        num_envs: int = 1,
        seed: int | None = None,
        learning_rate: float | None = None,
        hidden_dim: int | None = None,
        clip_epsilon: float | None = None,
        wait_for_result: bool = False,
    ) -> Dict[str, Any]:
        """Submit PPO training to Ray and return job metadata or result."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than zero")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be greater than zero")
        if num_envs <= 0:
            raise ValueError("num_envs must be greater than zero")
        init_ray()
        record_training_job_started()

        run_id = f"ppo-{uuid4()}"
        created_at = datetime.now(tz=timezone.utc).isoformat()
        ref = run_ppo_training.remote(
            run_id,
            created_at,
            episodes,
            steps_per_episode,
            num_envs,
            eval_episodes,
            seed,
            learning_rate,
            hidden_dim,
            clip_epsilon,
        )

        with self._lock:
            self._jobs[run_id] = ref
            self._latest_job_id = run_id

        if wait_for_result:
            started = time.perf_counter()
            try:
                result = ray.get(ref)
                with self._lock:
                    finalized = self._finalize_job(run_id, result, record_metrics=True)
                record_training_duration(time.perf_counter() - started, "completed")
                return finalized
            except Exception:
                record_training_job_failed()
                record_training_duration(time.perf_counter() - started, "failed")
                raise

        return {
            "status": "running",
            "job_id": run_id,
            "created_at": created_at,
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Return job status without blocking on completion."""
        with self._lock:
            if job_id in self._job_results:
                result = self._job_results[job_id]
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "created_at": result.get("created_at"),
                }

            result = self._finalize_job_if_ready(job_id)
            if result is not None:
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "created_at": result.get("created_at"),
                }

            if job_id in self._jobs:
                return {
                    "status": "running",
                    "job_id": job_id,
                }

            return {
                "status": "not_found",
                "job_id": job_id,
            }

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Return job result if completed; otherwise a status payload."""
        with self._lock:
            if job_id in self._job_results:
                return dict(self._job_results[job_id])

            result = self._finalize_job_if_ready(job_id)
            if result is not None:
                return result

            if job_id in self._jobs:
                return {
                    "status": "running",
                    "job_id": job_id,
                }

            return {
                "status": "not_found",
                "job_id": job_id,
            }

    def _finalize_job_if_ready(self, job_id: str) -> Dict[str, Any] | None:
        ref = self._jobs.get(job_id)
        if ref is None:
            return None

        init_ray()
        ready, _ = ray.wait([ref], timeout=0)
        if not ready:
            return None

        result = ray.get(ref)
        return self._finalize_job(job_id, result)

    def _finalize_job(
        self,
        job_id: str,
        result: Dict[str, Any],
        record_metrics: bool = True,
    ) -> Dict[str, Any]:
        self._jobs.pop(job_id, None)
        self._job_results[job_id] = dict(result)
        self._latest_run = dict(result)
        comparison = result.get("comparison", {})
        self._latest_comparison = {
            "run_id": result.get("run_id"),
            "created_at": result.get("created_at"),
            **comparison,
        }
        if record_metrics:
            record_training_job_completed(result.get("training", {}))
        return dict(result)

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
                if self._latest_job_id and self._latest_job_id in self._jobs:
                    return {
                        "status": "running",
                        "job_id": self._latest_job_id,
                    }
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
