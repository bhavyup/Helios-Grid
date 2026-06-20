"""PPO proof-of-life agent for single-household training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Normal

from app.core.project_config import config
from app.domain.rewards.reward_utils import compute_house_reward
from app.envs.house_env import HouseEnv
from app.infrastructure.data_utils import get_data_paths, load_weather_data


@dataclass
class _Transition:
    state: np.ndarray
    raw_action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: float


class _ActorCritic(nn.Module):
    """Shared-backbone actor-critic used by PPOAgent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(means)
        return means, std, values


class _HouseEnvVectorAdapter(Env):
    """Gymnasium-compatible wrapper for HouseEnv."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_episode_steps: int,
        max_battery: float,
        base_price: float,
        weather_file: str,
    ) -> None:
        super().__init__()
        self._env = HouseEnv(
            max_episode_steps=max_episode_steps,
            max_battery=max_battery,
            base_price=base_price,
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._max_steps = max(1, int(max_episode_steps))
        self._step_count = 0
        self._max_battery = float(max_battery)
        self._weather_data = load_weather_data(weather_file)

    def _weather_row(self) -> object | None:
        if len(self._weather_data) == 0:
            return None
        idx = min(self._step_count, len(self._weather_data) - 1)
        return self._weather_data[idx]

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
        self._step_count = 0
        weather_row = self._weather_row()
        if weather_row is not None:
            self._env.current_weather = weather_row
        obs = self._env.reset()
        info: dict[str, object] = {}
        return obs, info

    def step(self, action):
        weather_row = self._weather_row()
        if weather_row is not None:
            self._env.current_weather = weather_row
        obs = self._env.step(action)
        reward = float(
            compute_house_reward(
                consumption=float(obs[1]),
                production=float(obs[2]),
                price=float(obs[4]),
                battery_level=float(obs[3]),
                max_battery=self._max_battery,
            )
        )
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._max_steps
        info: dict[str, object] = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        return self._env.close()


class PPOAgent:
    """Production-grade PPO proof-of-life implementation for Phase 2 demos."""

    def __init__(
        self,
        learning_rate: float | None = None,
        gamma: float | None = None,
        gae_lambda: float = 0.95,
        clip_epsilon: float | None = None,
        entropy_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        ppo_epochs: int | None = None,
        batch_size: int | None = None,
        hidden_dim: int = 128,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        self.obs_dim = self._resolve_observation_dim()
        self.action_dim = self._resolve_action_dim()

        self.learning_rate = float(
            self._resolve_value(learning_rate, "learning_rate", "lr", 3e-4)
        )
        self.gamma = float(self._resolve_value(gamma, "gamma", "gamma", 0.99))
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(
            self._resolve_value(clip_epsilon, "clip_epsilon", "clip_epsilon", 0.2)
        )
        self.entropy_coef = float(
            self._resolve_value(entropy_coef, "entropy_coef", "entropy_coef", 0.01)
        )
        self.vf_coef = float(self._resolve_value(vf_coef, "vf_coef", "vf_coef", 0.5))
        self.max_grad_norm = float(
            self._resolve_value(max_grad_norm, "max_grad_norm", "max_grad_norm", 0.5)
        )
        self.ppo_epochs = int(self._resolve_value(ppo_epochs, "epochs", "epochs", 8))
        self.batch_size = int(
            self._resolve_value(batch_size, "batch_size", "batch_size", 64)
        )
        self.hidden_dim = int(hidden_dim)

        self.seed = self._resolve_seed(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.weather_data = load_weather_data(get_data_paths()["weather_data"])
        self.weather_feature_names = [
            "solar_irradiance",
            "wind_speed",
            "temperature",
            "humidity",
        ]

        self.model = _ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self._latest_training_summary: dict[str, object] = {}

        self._seed_everything(self.seed)

    def train(
        self,
        episodes: int = 20,
        steps_per_episode: int = 24,
        seed: int | None = None,
        num_envs: int = 1,
    ) -> dict[str, object]:
        """Train PPO on HouseEnv and return a rich progress artifact."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than zero")
        if num_envs <= 0:
            raise ValueError("num_envs must be greater than zero")

        run_seed = self.seed if seed is None else int(seed)
        self._seed_everything(run_seed)

        started_at = perf_counter()
        reward_curve: list[dict[str, float]] = []

        if num_envs <= 1:
            for episode in range(episodes):
                transitions: list[_Transition] = []
                env = self._build_house_env(steps_per_episode=steps_per_episode)
                env_seed = run_seed + (episode * 97)
                env.seed(env_seed)

                state = env.reset()
                episode_reward = 0.0

                for step in range(steps_per_episode):
                    weather_row = self._weather_row(step)
                    self._attach_weather(env, weather_row)
                    model_state = self._augment_state_with_weather(state, weather_row)
                    state_tensor = self._state_tensor(model_state)
                    with torch.no_grad():
                        mean, std, value = self.model(state_tensor)
                        dist = Normal(mean, std)
                        raw_action = dist.rsample()
                        log_prob = dist.log_prob(raw_action).sum(dim=-1)
                        action = (
                            torch.sigmoid(raw_action)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                    next_state = env.step(action)
                    reward = self._compute_house_reward(next_state)
                    done = float(step == (steps_per_episode - 1))

                    transitions.append(
                        _Transition(
                            state=np.asarray(model_state, dtype=np.float32),
                            raw_action=raw_action.squeeze(0)
                            .cpu()
                            .numpy()
                            .astype(np.float32),
                            log_prob=float(log_prob.item()),
                            value=float(value.item()),
                            reward=float(reward),
                            done=done,
                        )
                    )

                    state = next_state
                    episode_reward += reward

                last_model_state = self._augment_state_with_weather(
                    state, self._weather_row(steps_per_episode)
                )
                loss_stats = self._update_policy(
                    transitions=transitions, last_state=last_model_state
                )
                moving_avg = self._moving_average(
                    [item["reward"] for item in reward_curve] + [episode_reward],
                    5,
                )

                reward_curve.append(
                    {
                        "episode": float(episode + 1),
                        "reward": float(episode_reward),
                        "moving_average_reward": float(moving_avg),
                        "policy_loss": float(loss_stats["policy_loss"]),
                        "value_loss": float(loss_stats["value_loss"]),
                        "entropy": float(loss_stats["entropy"]),
                    }
                )
        else:
            env = self._build_vector_env(
                num_envs=num_envs,
                steps_per_episode=steps_per_episode,
            )
            try:
                for episode in range(episodes):
                    seeds = [run_seed + (episode * 97) + i for i in range(num_envs)]
                    state, _ = env.reset(seed=seeds)
                    episode_rewards = np.zeros(num_envs, dtype=np.float32)

                    states_buffer: list[np.ndarray] = []
                    raw_actions_buffer: list[np.ndarray] = []
                    log_probs_buffer: list[np.ndarray] = []
                    values_buffer: list[np.ndarray] = []
                    rewards_buffer: list[np.ndarray] = []
                    dones_buffer: list[np.ndarray] = []

                    for _ in range(steps_per_episode):
                        weather_row = self._weather_row(len(states_buffer))
                        augmented_state = self._augment_state_batch_with_weather(
                            state, weather_row
                        )
                        state_tensor = self._state_tensor(augmented_state)
                        with torch.no_grad():
                            mean, std, value = self.model(state_tensor)
                            dist = Normal(mean, std)
                            raw_action = dist.rsample()
                            log_prob = dist.log_prob(raw_action).sum(dim=-1)
                            action = (
                                torch.sigmoid(raw_action)
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )

                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = np.logical_or(terminated, truncated).astype(np.float32)

                        states_buffer.append(
                            np.asarray(augmented_state, dtype=np.float32)
                        )
                        raw_actions_buffer.append(
                            raw_action.detach().cpu().numpy().astype(np.float32)
                        )
                        log_probs_buffer.append(
                            log_prob.detach().cpu().numpy().astype(np.float32)
                        )
                        values_buffer.append(
                            value.detach().cpu().numpy().astype(np.float32)
                        )
                        rewards_buffer.append(np.asarray(reward, dtype=np.float32))
                        dones_buffer.append(done)

                        episode_rewards += np.asarray(reward, dtype=np.float32)
                        state = next_state

                    last_augmented_state = self._augment_state_batch_with_weather(
                        state, self._weather_row(steps_per_episode)
                    )
                    loss_stats = self._update_policy_vectorized(
                        states=np.stack(states_buffer, axis=0),
                        raw_actions=np.stack(raw_actions_buffer, axis=0),
                        log_probs=np.stack(log_probs_buffer, axis=0),
                        values=np.stack(values_buffer, axis=0),
                        rewards=np.stack(rewards_buffer, axis=0),
                        dones=np.stack(dones_buffer, axis=0),
                        last_state=last_augmented_state,
                    )
                    episode_reward = float(np.mean(episode_rewards))
                    moving_avg = self._moving_average(
                        [item["reward"] for item in reward_curve] + [episode_reward],
                        5,
                    )

                    reward_curve.append(
                        {
                            "episode": float(episode + 1),
                            "reward": float(episode_reward),
                            "moving_average_reward": float(moving_avg),
                            "policy_loss": float(loss_stats["policy_loss"]),
                            "value_loss": float(loss_stats["value_loss"]),
                            "entropy": float(loss_stats["entropy"]),
                        }
                    )
            finally:
                env.close()

        elapsed = perf_counter() - started_at
        final_metrics = self.evaluate(
            episodes=min(5, episodes),
            steps_per_episode=steps_per_episode,
            policy_mode="ppo",
            seed=run_seed + 1000,
        )

        summary: dict[str, object] = {
            "algorithm": "ppo-proof-of-life",
            "seed": run_seed,
            "episodes": int(episodes),
            "steps_per_episode": int(steps_per_episode),
            "num_envs": int(num_envs),
            "duration_seconds": float(elapsed),
            "reward_curve": reward_curve,
            "final_training_reward": float(reward_curve[-1]["reward"]),
            "best_training_reward": float(max(item["reward"] for item in reward_curve)),
            "final_eval_metrics": final_metrics,
        }
        self._latest_training_summary = summary
        return summary

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action vector in [0, 1] for the provided state."""
        state_tensor = self._state_tensor(self._augment_state_with_weather(state, None))
        with torch.no_grad():
            mean, std, _ = self.model(state_tensor)
            if deterministic:
                raw_action = mean
            else:
                raw_action = Normal(mean, std).sample()
            action = (
                torch.sigmoid(raw_action).squeeze(0).cpu().numpy().astype(np.float32)
            )
        return np.clip(action, 0.0, 1.0)

    def evaluate(
        self,
        episodes: int = 5,
        steps_per_episode: int = 24,
        policy_mode: str = "ppo",
        seed: int | None = None,
    ) -> dict[str, float | int | str]:
        """Evaluate PPO policy or rule baseline on HouseEnv."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")

        run_seed = self.seed if seed is None else int(seed)

        total_reward = 0.0
        total_consumption = 0.0
        total_production = 0.0
        total_price = 0.0
        total_grid_import = 0.0
        total_steps = 0

        for episode in range(episodes):
            env = self._build_house_env(steps_per_episode=steps_per_episode)
            env.seed(run_seed + (episode * 41))

            state = env.reset()
            for _ in range(steps_per_episode):
                weather_row = self._weather_row(total_steps)
                self._attach_weather(env, weather_row)
                model_state = self._augment_state_with_weather(state, weather_row)
                if policy_mode == "ppo":
                    action = self.predict(model_state, deterministic=True)
                elif policy_mode == "rule":
                    action = self._rule_action(model_state)
                else:
                    raise ValueError(f"Unsupported policy_mode: {policy_mode}")

                next_state = env.step(action)
                reward = self._compute_house_reward(next_state)

                total_reward += reward
                total_consumption += float(next_state[1])
                total_production += float(next_state[2])
                total_price += float(next_state[4])
                total_grid_import += float(next_state[5])
                total_steps += 1
                state = next_state

        divisor = max(total_steps, 1)
        return {
            "policy_mode": policy_mode,
            "episodes": int(episodes),
            "steps_per_episode": int(steps_per_episode),
            "average_reward": float(total_reward / divisor),
            "average_consumption": float(total_consumption / divisor),
            "average_production": float(total_production / divisor),
            "average_price": float(total_price / divisor),
            "average_grid_import": float(total_grid_import / divisor),
        }

    def compare_against_rule(
        self,
        episodes: int = 5,
        steps_per_episode: int = 24,
        seed: int | None = None,
    ) -> dict[str, object]:
        """Compute side-by-side metrics for rule baseline and PPO policy."""
        ppo_metrics = self.evaluate(
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            policy_mode="ppo",
            seed=seed,
        )
        rule_metrics = self.evaluate(
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            policy_mode="rule",
            seed=seed,
        )

        deltas = {
            "reward_delta": float(
                ppo_metrics["average_reward"] - rule_metrics["average_reward"]
            ),
            "grid_import_delta": float(
                ppo_metrics["average_grid_import"] - rule_metrics["average_grid_import"]
            ),
            "price_delta": float(
                ppo_metrics["average_price"] - rule_metrics["average_price"]
            ),
        }

        return {
            "ppo": ppo_metrics,
            "rule": rule_metrics,
            "deltas": deltas,
        }

    @property
    def latest_training_summary(self) -> dict[str, object]:
        return dict(self._latest_training_summary)

    def _update_policy(
        self,
        transitions: list[_Transition],
        last_state: np.ndarray,
    ) -> dict[str, float]:
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states = torch.as_tensor(
            np.asarray([item.state for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        raw_actions = torch.as_tensor(
            np.asarray([item.raw_action for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            np.asarray([item.log_prob for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        values = np.asarray([item.value for item in transitions], dtype=np.float32)
        rewards = np.asarray([item.reward for item in transitions], dtype=np.float32)
        dones = np.asarray([item.done for item in transitions], dtype=np.float32)

        with torch.no_grad():
            _, _, last_value_tensor = self.model(self._state_tensor(last_state))
            last_value = float(last_value_tensor.item())

        returns_np, advantages_np = self._compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
        )

        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )

        batch_size = min(self.batch_size, len(transitions))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(len(transitions), device=self.device)
            for start in range(0, len(transitions), batch_size):
                batch_indices = permutation[start : start + batch_size]

                means, std, value_preds = self.model(states[batch_indices])
                dist = Normal(means, std)
                new_log_probs = dist.log_prob(raw_actions[batch_indices]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                unclipped = ratio * advantages[batch_indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * advantages[batch_indices]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value_preds, returns[batch_indices])

                loss = (
                    policy_loss
                    + (self.vf_coef * value_loss)
                    - (self.entropy_coef * entropy)
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _update_policy_vectorized(
        self,
        states: np.ndarray,
        raw_actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        last_state: np.ndarray,
    ) -> dict[str, float]:
        if states.size == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        steps, num_envs, _ = states.shape
        states_tensor = torch.as_tensor(
            states.reshape(steps * num_envs, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        raw_actions_tensor = torch.as_tensor(
            raw_actions.reshape(steps * num_envs, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            log_probs.reshape(steps * num_envs),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            _, _, last_values_tensor = self.model(self._state_tensor(last_state))
            last_values = last_values_tensor.detach().cpu().numpy().astype(np.float32)

        returns_np, advantages_np = self._compute_gae_vectorized(
            rewards=rewards.reshape(steps, num_envs),
            values=values.reshape(steps, num_envs),
            dones=dones.reshape(steps, num_envs),
            last_values=last_values,
        )

        returns = torch.as_tensor(
            returns_np.reshape(steps * num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            advantages_np.reshape(steps * num_envs),
            dtype=torch.float32,
            device=self.device,
        )

        total_samples = int(steps * num_envs)
        batch_size = min(self.batch_size, total_samples)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, batch_size):
                batch_indices = permutation[start : start + batch_size]

                means, std, value_preds = self.model(states_tensor[batch_indices])
                dist = Normal(means, std)
                new_log_probs = dist.log_prob(raw_actions_tensor[batch_indices]).sum(
                    dim=-1
                )
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                unclipped = ratio * advantages[batch_indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * advantages[batch_indices]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value_preds, returns[batch_indices])

                loss = (
                    policy_loss
                    + (self.vf_coef * value_loss)
                    - (self.entropy_coef * entropy)
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for timestep in reversed(range(len(rewards))):
            if timestep == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = float(values[timestep + 1])

            mask = 1.0 - float(dones[timestep])
            delta = (
                float(rewards[timestep])
                + (self.gamma * next_value * mask)
                - float(values[timestep])
            )
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages[timestep] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.astype(np.float32), advantages.astype(np.float32)

    def _compute_gae_vectorized(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)

        for timestep in reversed(range(rewards.shape[0])):
            if timestep == rewards.shape[0] - 1:
                next_values = last_values
            else:
                next_values = values[timestep + 1]

            mask = 1.0 - dones[timestep]
            delta = (
                rewards[timestep] + (self.gamma * next_values * mask) - values[timestep]
            )
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages[timestep] = gae

        returns = advantages + values
        flat_adv = advantages.reshape(-1)
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        return returns.astype(np.float32), advantages.astype(np.float32)

    def _build_house_env(self, steps_per_episode: int) -> HouseEnv:
        return HouseEnv(
            max_episode_steps=steps_per_episode,
            max_battery=self._resolve_max_battery(),
            base_price=self._resolve_default_price(),
        )

    def _build_vector_env(
        self,
        num_envs: int,
        steps_per_episode: int,
    ) -> AsyncVectorEnv:
        max_battery = self._resolve_max_battery()
        base_price = self._resolve_default_price()
        env_fns = [
            partial(
                _HouseEnvVectorAdapter,
                max_episode_steps=steps_per_episode,
                max_battery=max_battery,
                base_price=base_price,
                weather_file=get_data_paths()["weather_data"],
            )
            for _ in range(num_envs)
        ]
        return AsyncVectorEnv(env_fns)

    def _weather_row(self, step_index: int) -> object | None:
        if len(self.weather_data) == 0:
            return None
        idx = min(max(step_index, 0), len(self.weather_data) - 1)
        return self.weather_data[idx]

    def _attach_weather(self, env: HouseEnv, weather_row: object | None) -> None:
        if weather_row is None:
            return
        try:
            env.current_weather = weather_row
        except Exception:
            pass

    def _weather_features(self, weather_row: object | None) -> np.ndarray:
        features = np.zeros(4, dtype=np.float32)
        if weather_row is None:
            return features

        try:
            getter = getattr(weather_row, "get", None)
            if callable(getter):
                features[0] = float(getter("solar_irradiance", 0.0) or 0.0)
                features[1] = float(getter("wind_speed", 0.0) or 0.0)
                features[2] = float(getter("temperature", 20.0) or 20.0)
                features[3] = float(getter("humidity", 50.0) or 50.0)
        except Exception:
            return features
        return features

    def _augment_state_with_weather(
        self, state: np.ndarray, weather_row: object | None
    ) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        return np.concatenate(
            [state_array, self._weather_features(weather_row)]
        ).astype(np.float32)

    def _augment_state_batch_with_weather(
        self,
        state: np.ndarray,
        weather_row: object | None,
    ) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        weather_features = self._weather_features(weather_row)
        weather_block = np.tile(weather_features, (state_array.shape[0], 1)).astype(
            np.float32
        )
        return np.concatenate([state_array, weather_block], axis=1)

    def _compute_house_reward(self, state: np.ndarray) -> float:
        return float(
            compute_house_reward(
                consumption=float(state[1]),
                production=float(state[2]),
                price=float(state[4]),
                battery_level=float(state[3]),
                max_battery=self._resolve_max_battery(),
            )
        )

    def _rule_action(self, state: np.ndarray) -> np.ndarray:
        max_battery = self._resolve_max_battery()
        price_ceiling = max(self._resolve_price_max(), 1e-6)
        default_price = self._resolve_default_price()

        consumption = float(state[1])
        production = float(state[2])
        battery_level = float(state[3])
        price = float(state[4])
        net_balance = float(state[9])

        demand_response = np.clip(1.0 - (price / price_ceiling), 0.0, 1.0)
        charge_signal = 0.75 if battery_level < (0.45 * max_battery) else 0.2
        discharge_signal = (
            0.65
            if (battery_level > (0.60 * max_battery) and price > default_price)
            else 0.1
        )
        buy_signal = np.clip(-net_balance, 0.0, 1.0)
        sell_signal = np.clip(net_balance, 0.0, 1.0)
        grid_import = np.clip(max(consumption - production, 0.0), 0.0, 1.0)

        return np.asarray(
            [
                float(demand_response),
                float(charge_signal),
                float(discharge_signal),
                float(buy_signal),
                float(sell_signal),
                float(grid_import),
            ],
            dtype=np.float32,
        )

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 1:
            state_array = state_array[None, :]
        return torch.as_tensor(state_array, dtype=torch.float32, device=self.device)

    @staticmethod
    def _moving_average(values: list[float], window: int) -> float:
        if not values:
            return 0.0
        effective_window = max(1, min(window, len(values)))
        return float(np.mean(values[-effective_window:]))

    def _seed_everything(self, seed: int) -> None:
        safe_seed = int(seed) % (2**31 - 1)
        random.seed(safe_seed)
        np.random.seed(safe_seed)
        torch.manual_seed(safe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(safe_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        if seed is not None:
            return int(seed)

        reproducibility = config.get("reproducibility", {})
        if hasattr(reproducibility, "get"):
            resolved = reproducibility.get("seed", None)
            if resolved is not None:
                return int(resolved)

        return int(config.get("seed", 42))

    @staticmethod
    def _resolve_observation_dim() -> int:
        env_cfg = config.get("env", {})
        if hasattr(env_cfg, "get"):
            return max(int(env_cfg.get("observation_dim", 14)), 14)
        return 14

    @staticmethod
    def _resolve_action_dim() -> int:
        env_cfg = config.get("env", {})
        if hasattr(env_cfg, "get"):
            return int(env_cfg.get("action_dim", 6))
        return 6

    @staticmethod
    def _resolve_max_battery() -> float:
        env_cfg = config.get("env", {})
        env_value = (
            env_cfg.get("max_battery", None) if hasattr(env_cfg, "get") else None
        )
        top_level = config.get("max_battery", None)
        if top_level is not None:
            return float(top_level)
        return float(env_value if env_value is not None else 10.0)

    @staticmethod
    def _resolve_default_price() -> float:
        market_cfg = config.get("market", {})
        if hasattr(market_cfg, "get"):
            return float(market_cfg.get("default_price", 0.3))
        return 0.3

    @staticmethod
    def _resolve_price_max() -> float:
        market_cfg = config.get("market", {})
        if hasattr(market_cfg, "get"):
            return float(market_cfg.get("price_max", 1.0))
        return 1.0

    @staticmethod
    def _resolve_value(
        value: float | None,
        canonical_key: str,
        legacy_key: str,
        default: float,
    ) -> float:
        if value is not None:
            return float(value)

        ppo_cfg = config.get("ppo", {})
        if hasattr(ppo_cfg, "get"):
            resolved = ppo_cfg.get(canonical_key, None)
            if resolved is not None:
                return float(resolved)
            resolved = ppo_cfg.get(legacy_key, None)
            if resolved is not None:
                return float(resolved)

        agents_cfg = config.get("agents", {})
        if hasattr(agents_cfg, "get"):
            resolved = agents_cfg.get(canonical_key, None)
            if resolved is not None:
                return float(resolved)
            resolved = agents_cfg.get(legacy_key, None)
            if resolved is not None:
                return float(resolved)

        return float(default)
