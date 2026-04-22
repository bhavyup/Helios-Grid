"""PPO proof-of-life agent for single-household training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from app.core.project_config import config
from app.envs.house_env import HouseEnv
from app.utils.reward_utils import compute_house_reward


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

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(means)
        return means, std, values


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
        self.batch_size = int(self._resolve_value(batch_size, "batch_size", "batch_size", 64))
        self.hidden_dim = int(hidden_dim)

        self.seed = self._resolve_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = _ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self._latest_training_summary: Dict[str, object] = {}

        self._seed_everything(self.seed)

    def train(
        self,
        episodes: int = 20,
        steps_per_episode: int = 24,
        seed: int | None = None,
    ) -> Dict[str, object]:
        """Train PPO on HouseEnv and return a rich progress artifact."""
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be greater than zero")

        run_seed = self.seed if seed is None else int(seed)
        self._seed_everything(run_seed)

        started_at = perf_counter()
        reward_curve: List[Dict[str, float]] = []

        for episode in range(episodes):
            transitions: List[_Transition] = []
            env = self._build_house_env(steps_per_episode=steps_per_episode)
            env_seed = run_seed + (episode * 97)
            env.seed(env_seed)

            state = env.reset()
            episode_reward = 0.0

            for step in range(steps_per_episode):
                state_tensor = self._state_tensor(state)
                with torch.no_grad():
                    mean, std, value = self.model(state_tensor)
                    dist = Normal(mean, std)
                    raw_action = dist.rsample()
                    log_prob = dist.log_prob(raw_action).sum(dim=-1)
                    action = torch.sigmoid(raw_action).squeeze(0).cpu().numpy().astype(np.float32)

                next_state = env.step(action)
                reward = self._compute_house_reward(next_state)
                done = float(step == (steps_per_episode - 1))

                transitions.append(
                    _Transition(
                        state=np.asarray(state, dtype=np.float32),
                        raw_action=raw_action.squeeze(0).cpu().numpy().astype(np.float32),
                        log_prob=float(log_prob.item()),
                        value=float(value.item()),
                        reward=float(reward),
                        done=done,
                    )
                )

                state = next_state
                episode_reward += reward

            loss_stats = self._update_policy(transitions=transitions, last_state=state)
            moving_avg = self._moving_average([item["reward"] for item in reward_curve] + [episode_reward], 5)

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

        elapsed = perf_counter() - started_at
        final_metrics = self.evaluate(
            episodes=min(5, episodes),
            steps_per_episode=steps_per_episode,
            policy_mode="ppo",
            seed=run_seed + 1000,
        )

        summary: Dict[str, object] = {
            "algorithm": "ppo-proof-of-life",
            "seed": run_seed,
            "episodes": int(episodes),
            "steps_per_episode": int(steps_per_episode),
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
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            mean, std, _ = self.model(state_tensor)
            if deterministic:
                raw_action = mean
            else:
                raw_action = Normal(mean, std).sample()
            action = torch.sigmoid(raw_action).squeeze(0).cpu().numpy().astype(np.float32)
        return np.clip(action, 0.0, 1.0)

    def evaluate(
        self,
        episodes: int = 5,
        steps_per_episode: int = 24,
        policy_mode: str = "ppo",
        seed: int | None = None,
    ) -> Dict[str, float | int | str]:
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
                if policy_mode == "ppo":
                    action = self.predict(state, deterministic=True)
                elif policy_mode == "rule":
                    action = self._rule_action(state)
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
    ) -> Dict[str, object]:
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
            "reward_delta": float(ppo_metrics["average_reward"] - rule_metrics["average_reward"]),
            "grid_import_delta": float(
                ppo_metrics["average_grid_import"] - rule_metrics["average_grid_import"]
            ),
            "price_delta": float(ppo_metrics["average_price"] - rule_metrics["average_price"]),
        }

        return {
            "ppo": ppo_metrics,
            "rule": rule_metrics,
            "deltas": deltas,
        }

    @property
    def latest_training_summary(self) -> Dict[str, object]:
        return dict(self._latest_training_summary)

    def _update_policy(
        self,
        transitions: List[_Transition],
        last_state: np.ndarray,
    ) -> Dict[str, float]:
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
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)

        batch_size = min(self.batch_size, len(transitions))
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []

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
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages[batch_indices]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value_preds, returns[batch_indices])

                loss = policy_loss + (self.vf_coef * value_loss) - (self.entropy_coef * entropy)

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
            delta = float(rewards[timestep]) + (self.gamma * next_value * mask) - float(values[timestep])
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages[timestep] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.astype(np.float32), advantages.astype(np.float32)

    def _build_house_env(self, steps_per_episode: int) -> HouseEnv:
        return HouseEnv(
            max_episode_steps=steps_per_episode,
            max_battery=self._resolve_max_battery(),
            base_price=self._resolve_default_price(),
        )

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
        discharge_signal = 0.65 if (battery_level > (0.60 * max_battery) and price > default_price) else 0.1
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
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    @staticmethod
    def _moving_average(values: List[float], window: int) -> float:
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
            return int(env_cfg.get("observation_dim", 10))
        return 10

    @staticmethod
    def _resolve_action_dim() -> int:
        env_cfg = config.get("env", {})
        if hasattr(env_cfg, "get"):
            return int(env_cfg.get("action_dim", 6))
        return 6

    @staticmethod
    def _resolve_max_battery() -> float:
        env_cfg = config.get("env", {})
        env_value = env_cfg.get("max_battery", None) if hasattr(env_cfg, "get") else None
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
        explicit: float | int | None,
        canonical_key: str,
        legacy_key: str,
        default: float | int,
    ) -> float | int:
        if explicit is not None:
            return explicit

        ppo_cfg = config.get("ppo", {})
        if hasattr(ppo_cfg, "get"):
            value = ppo_cfg.get(canonical_key, None)
            if value is not None:
                return value
            legacy_value = ppo_cfg.get(legacy_key, None)
            if legacy_value is not None:
                return legacy_value

        nested_cfg = config.get("agents", {})
        if hasattr(nested_cfg, "get"):
            agents_ppo = nested_cfg.get("ppo", {})
            if hasattr(agents_ppo, "get"):
                value = agents_ppo.get(legacy_key, None)
                if value is not None:
                    return value

        return default
