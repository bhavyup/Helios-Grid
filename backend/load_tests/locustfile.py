from __future__ import annotations

import os
import random

from gevent.lock import Semaphore
from locust import HttpUser, SequentialTaskSet, between, task


def _env_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None or value == "":
		return default
	return int(value)


def _json_header(token: str | None = None) -> dict[str, str]:
	headers = {"Content-Type": "application/json"}
	if token:
		headers["Authorization"] = f"Bearer {token}"
	return headers


class BaseBackendUser(HttpUser):
	abstract = True
	host = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
	wait_time = between(
		float(os.getenv("LOCUST_MIN_WAIT", "0.1")),
		float(os.getenv("LOCUST_MAX_WAIT", "0.75")),
	)
	_shared_access_token: str | None = None
	_token_lock = Semaphore()

	def on_start(self) -> None:
		self.auth_enabled = _env_bool("AUTH_ENABLED", True)
		self.username = os.getenv("LOCUST_USERNAME", "loadtest@example.com")
		self.password = os.getenv("LOCUST_PASSWORD", "test-pass-123")
		self.access_token: str | None = None

		if self.auth_enabled:
			self.access_token = self._get_shared_access_token()

	def _get_shared_access_token(self) -> str:
		if self.__class__._shared_access_token:
			return self.__class__._shared_access_token

		with self.__class__._token_lock:
			if self.__class__._shared_access_token:
				return self.__class__._shared_access_token

			login_response = self.client.post(
				"/auth/login",
				json={"email": self.username, "password": self.password},
				headers={"Content-Type": "application/json"},
				name="POST /auth/login",
			)
			if login_response.status_code != 200:
				register_response = self.client.post(
					"/auth/register",
					json={"email": self.username, "password": self.password},
					headers={"Content-Type": "application/json"},
					name="POST /auth/register",
				)
				if register_response.status_code not in (200, 409):
					register_response.raise_for_status()

				login_response = self.client.post(
					"/auth/login",
					json={"email": self.username, "password": self.password},
					headers={"Content-Type": "application/json"},
					name="POST /auth/login",
				)

			login_response.raise_for_status()
			self.__class__._shared_access_token = login_response.json()["access_token"]
			return self.__class__._shared_access_token

	def _request_headers(self) -> dict[str, str]:
		if not self.auth_enabled:
			return {"Content-Type": "application/json"}
		return _json_header(self.access_token)

	def _post_json(self, path: str, payload: dict[str, object], name: str) -> None:
		response = self.client.post(
			path,
			json=payload,
			headers=self._request_headers(),
			name=name,
		)
		response.raise_for_status()

	def _get(self, path: str, name: str) -> None:
		response = self.client.get(
			path,
			headers=self._request_headers(),
			name=name,
		)
		response.raise_for_status()


class SimulationLoadFlow(SequentialTaskSet):
	@task
	def cycle_simulations(self) -> None:
		burst_size = _env_int("SIMULATION_BURST_SIZE", 1)
		households = _env_int("SIMULATION_HOUSEHOLDS", 32)
		max_steps = _env_int("SIMULATION_MAX_STEPS", 96)
		run_steps = _env_int("SIMULATION_RUN_STEPS", 5)

		for _ in range(burst_size):
			seed = random.randint(1, 1_000_000)
			self.user._post_json(
				"/simulation/reset",
				{
					"seed": seed,
					"num_households": households,
					"max_episode_steps": max_steps,
				},
				name="POST /simulation/reset",
			)
			self.user._post_json(
				"/simulation/step",
				{
					"market_action": 1,
					"use_autopilot": True,
				},
				name="POST /simulation/step",
			)
			self.user._post_json(
				"/simulation/run",
				{
					"steps": run_steps,
					"use_autopilot": True,
					"market_action": 1,
				},
				name="POST /simulation/run",
			)
			self.user._get("/simulation/metrics", name="GET /simulation/metrics")
			self.user._get("/simulation/history?limit=25", name="GET /simulation/history")

	@task
	def inspect_simulation_state(self) -> None:
		self.user._get("/simulation/state?include_topology=false", name="GET /simulation/state")


class TrainingLoadFlow(SequentialTaskSet):
	@task
	def submit_training_job(self) -> None:
		wait_for_result = _env_bool("TRAINING_WAIT_FOR_RESULT", False)
		response = self.user.client.post(
			"/training/ppo/run",
			json={
				"episodes": _env_int("TRAINING_EPISODES", 2),
				"steps_per_episode": _env_int("TRAINING_STEPS_PER_EPISODE", 4),
				"num_envs": _env_int("TRAINING_NUM_ENVS", 1),
				"eval_episodes": _env_int("TRAINING_EVAL_EPISODES", 1),
				"seed": random.randint(1, 1_000_000),
				"wait_for_result": wait_for_result,
			},
			headers=self.user._request_headers(),
			name="POST /training/ppo/run",
		)
		response.raise_for_status()
		payload = response.json()

		if not wait_for_result and payload.get("job_id"):
			job_id = payload["job_id"]
			status = self.user.client.get(
				f"/training/ppo/status/{job_id}",
				headers=self.user._request_headers(),
				name="GET /training/ppo/status/{job_id}",
			)
			status.raise_for_status()

	@task
	def inspect_training_results(self) -> None:
		self.user._get("/training/ppo/latest", name="GET /training/ppo/latest")
		self.user._get("/training/ppo/reward-curve", name="GET /training/ppo/reward-curve")


class SimulationLoadUser(BaseBackendUser):
	tasks = [SimulationLoadFlow]


class TrainingLoadUser(BaseBackendUser):
	tasks = [TrainingLoadFlow]