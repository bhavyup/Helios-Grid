"""GridEnv orchestrates grid simulation engines and household envs."""

from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict as GymDict, Box, Discrete
from gymnasium.utils import seeding

from app.core.project_config import config
from app.domain.models.gnn_coordinator import GNNCoordinator
from app.infrastructure.logging_utils import log_env_info
from app.simulations.household_manager import HouseholdManager
from app.simulations.market_engine import MarketEngine
from app.simulations.reward_engine import RewardEngine
from app.simulations.topology_engine import TopologyEngine
from app.simulations.weather_engine import WeatherEngine


class GridEnv(Env):
    """
    Top-level grid environment that orchestrates household sub-environments,
    coordination, weather data, and market actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_topology_file: str = "data/grid_topology/sample_grid.json",
        weather_file: str = "data/weather_data/sample_weather.csv",
        num_households: int = config["num_households"],
        max_episode_steps: int = config["training_steps"],
    ):
        super().__init__()

        self.grid_topology_file = grid_topology_file
        self.weather_file = weather_file
        self.num_households = num_households
        self.max_episode_steps = max_episode_steps

        self.topology_engine = TopologyEngine(
            grid_topology_file=grid_topology_file,
            num_households=num_households,
        )
        self.graph = self.topology_engine.graph
        self.nodes = self.topology_engine.nodes
        self.edges = self.topology_engine.edges

        self.weather_engine = WeatherEngine(weather_file=weather_file)
        self.weather_data = self.weather_engine.weather_data
        self.current_time = 0

        self.household_manager = HouseholdManager(num_households=num_households)
        self.house_environments = self.household_manager.house_environments

        self.gnn_coordinator = GNNCoordinator(self.graph)

        self.market_engine = MarketEngine()
        self.market_model = self.market_engine.market_model
        self.last_market_snapshot = self.market_engine.reset()

        self.reward_engine = RewardEngine()
        self.last_coordination_summary = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

        self.action_space = GymDict({
            "house_actions": Box(
                low=0, high=1, shape=(num_households, 6), dtype=np.float32
            ),
            "market_actions": Discrete(2),
        })

        self.observation_space = GymDict({
            "house_states": Box(
                low=-np.inf, high=np.inf,
                shape=(num_households, 10), dtype=np.float32,
            ),
            "grid_state": Box(
                low=-np.inf, high=np.inf,
                shape=(num_households, 10), dtype=np.float32,
            ),
            "market_state": Box(
                low=-np.inf, high=np.inf,
                shape=(num_households, 2), dtype=np.float32,
            ),
        })

        self.episode_count = 0
        self.episode_length = 0
        self.total_reward = 0.0

        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        """Seed all internal RNGs for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        base_seed = int(seed) % (2**31 - 1)

        self.household_manager.seed(base_seed)

        if hasattr(self.gnn_coordinator, "seed_everything"):
            self.gnn_coordinator.seed_everything(base_seed)

        return [seed]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to the initial state."""
        self.current_time = 0
        self.episode_count += 1
        self.episode_length = 0
        self.total_reward = 0.0

        self.household_manager.reset()
        self.gnn_coordinator.reset()
        self.last_market_snapshot = self.market_engine.reset()
        self.last_coordination_summary = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

        return self._get_observation()

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Execute one timestep."""
        self.episode_length += 1

        weather_datum, weather_idx = self.weather_engine.get_weather_at(
            self.current_time
        )

        house_actions = actions["house_actions"]
        market_actions = actions["market_actions"]

        house_step_results = self.household_manager.step(house_actions)
        extracted_states = [
            self.household_manager.extract_state(result)
            for result in house_step_results
        ]

        coordination_signals = self.gnn_coordinator.compute_coordination_signals(
            house_step_results, self.graph, weather_datum
        )
        self.last_coordination_summary = self._summarize_coordination_signals(
            coordination_signals
        )

        aggregate = self.household_manager.aggregate_states(extracted_states)
        self.last_market_snapshot = self.market_engine.step(
            supply=aggregate["supply"],
            demand=aggregate["demand"],
            market_action=int(market_actions),
            solar=aggregate["supply"],
            wind=0.0,
        )

        self.current_time += 1

        step_reward = self.reward_engine.compute(self.last_market_snapshot)
        self.total_reward += step_reward

        done = self.episode_length >= self.max_episode_steps

        log_env_info(
            self.episode_count,
            self.episode_length,
            self.current_time,
            step_reward,
        )

        info = {
            "episode_count": self.episode_count,
            "episode_length": self.episode_length,
            "current_time": self.current_time,
            "step_reward": step_reward,
            "total_reward": self.total_reward,
            "weather_index_used": weather_idx,
            "coordination_summary": self.last_coordination_summary,
            "market_snapshot": self.last_market_snapshot,
        }

        return self._get_observation(), step_reward, done, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, np.ndarray]:
        house_states = self.household_manager.get_states()
        return {
            "house_states": np.array(house_states, dtype=np.float32),
            "grid_state": self.household_manager.get_grid_state(
                self.current_time, self.max_episode_steps
            ),
            "market_state": self.market_engine.get_market_state(
                self.num_households
            ),
        }

    @staticmethod
    def _summarize_coordination_signals(signals: Any) -> Dict[str, float]:
        array = np.asarray(signals, dtype=np.float32)
        if array.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    # ------------------------------------------------------------------
    # Rendering / lifecycle
    # ------------------------------------------------------------------

    def render(self, mode="human"):
        """Render the environment (for visualization / debugging)."""
        print(
            f"Episode: {self.episode_count}, "
            f"Time: {self.current_time}, "
            f"Reward: {self.total_reward:.4f}"
        )

    def close(self):
        """Close the environment (release resources)."""
        pass
