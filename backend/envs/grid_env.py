import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import Env
from gymnasium.spaces import Dict as GymDict, Box, Discrete
from gymnasium.utils import seeding

from config import config
from envs.house_env import HouseEnv
from models.gnn_coordinator import GNNCoordinator
from utils.data_utils import load_weather_data
from utils.graph_utils import build_grid_graph, create_grid_graph
from utils.logging_utils import log_env_info
from utils.reward_utils import compute_grid_reward


class GridEnv(Env):
    """
    Top-level grid environment that orchestrates household sub-environments,
    a GNN coordinator, weather data, and market actions.
    """

    def __init__(
        self,
        grid_topology_file: str = "",
        weather_file: str = "",
        num_households: int = config["num_households"],
        max_episode_steps: int = config["env"]["max_episode_steps"],
    ):
        """
        Initialize the main grid environment.

        Args:
            grid_topology_file: Optional path to grid topology file (JSON).
                If empty, uses an in-memory default graph topology.
            weather_file: Optional path to weather data file (CSV).
                If empty or missing, synthetic default weather is used.
            num_households: Number of households in the grid.
            max_episode_steps: Maximum number of timesteps per episode.
                NOTE: sourced from canonical config["env"]["max_episode_steps"].
        """
        super().__init__()

        self.grid_topology_file = grid_topology_file
        self.weather_file = weather_file
        self.num_households = num_households
        self.max_episode_steps = max_episode_steps

        # Load grid topology and build graph
        if grid_topology_file:
            self.graph = build_grid_graph(grid_topology_file, num_households)
        else:
            self.graph = create_grid_graph(num_households=num_households)
        self.nodes = list(self.graph.nodes)
        self.edges = list(self.graph.edges)

        # Load weather data (solar irradiance, wind speed, etc.)
        # ASSUMPTION: load_weather_data returns an object that supports
        #   integer indexing for rows (list, ndarray, or DataFrame with
        #   positional access).  If it returns a DataFrame, callers may
        #   need to ensure .iloc is used downstream.
        self.weather_data = load_weather_data(weather_file)
        self.current_time = 0

        # Initialize household environments
        self.house_environments = [HouseEnv() for _ in range(num_households)]

        # Initialize GNN coordinator
        self.gnn_coordinator = GNNCoordinator(self.graph)

        # ------------------------------------------------------------------
        # Action and observation spaces
        # ------------------------------------------------------------------
        self.action_space = GymDict({
            "house_actions": Box(
                low=-1.0, high=1.0, shape=(num_households, 6), dtype=np.float32
            ),
            "market_actions": Discrete(2),  # 0 = hold, 1 = trade
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
        self._grid_state = np.zeros((self.num_households, 10), dtype=np.float32)
        self._market_state = np.zeros((self.num_households, 2), dtype=np.float32)

        # Deterministic RNG — seeded lazily; call seed() to set explicitly
        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        """Seed all internal RNGs for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        base_seed = int(seed)

        # Reseed household sub-environments deterministically.
        for idx, house in enumerate(self.house_environments):
            house.seed(base_seed + idx + 1)

        # Reseed coordinator RNGs deterministically.
        self.gnn_coordinator.seed = base_seed + len(self.house_environments) + 1
        self.gnn_coordinator.seed_everything(self.gnn_coordinator.seed)

        return [seed]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        # Retained for Gymnasium API compatibility.
        _ = options
        if seed is not None:
            self.seed(seed)

        self.current_time = 0
        self.episode_count += 1
        self.episode_length = 0
        self.total_reward = 0.0

        for house in self.house_environments:
            house.reset()

        self.gnn_coordinator.reset()
        self._update_grid_state()
        self._update_market_state()

        observation = self._get_observation()
        info = {
            "episode_count": self.episode_count,
            "episode_length": self.episode_length,
            "current_time": self.current_time,
            "total_reward": self.total_reward,
        }
        return observation, info

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            actions: Dict with keys ``"house_actions"`` and ``"market_actions"``.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.episode_length += 1

        # --- weather index: use current time, then advance ----------------
        weather_len = len(self.weather_data)
        if weather_len == 0:
            raise RuntimeError("weather_data is empty; cannot index")
        weather_idx = min(self.current_time, weather_len - 1)

        # --- extract actions ----------------------------------------------
        house_actions = actions["house_actions"]
        market_actions = actions["market_actions"]

        # --- step each household ------------------------------------------
        # HouseEnv.step follows Gymnasium convention and returns
        # (obs, reward, terminated, truncated, info). We only need obs here.
        house_step_results = []
        for i, house in enumerate(self.house_environments):
            result = house.step(house_actions[i])
            house_obs = result[0] if isinstance(result, tuple) else result
            house_step_results.append(house_obs)

        # --- GNN coordination --------------------------------------------
        # TODO: coordination_signals are computed but not yet consumed.
        #       Wire into house sub-envs or include in observation when the
        #       coordination protocol is defined.
        # Use .iloc for positional row selection if weather_data is a DataFrame
        weather_datum = (
            self.weather_data.iloc[weather_idx]
            if hasattr(self.weather_data, 'iloc')
            else self.weather_data[weather_idx]
        )
        coordination_signals = self.gnn_coordinator.compute_coordination_signals(
            house_step_results, self.graph, weather_datum
        )

        # Advance time *after* consuming the current weather datum
        self.current_time += 1

        # --- reward -------------------------------------------------------
        market_reward = 0.0
        if market_actions == 1:
            # Compute aggregate supply, demand, and price
            # ASSUMPTION: house_step_results contain supply/demand info
            # For now, use placeholder aggregation
            total_supply = sum(
                getattr(env, 'supply', 0.0) for env in self.house_environments
            )
            total_demand = sum(
                getattr(env, 'demand', 0.0) for env in self.house_environments
            )
            # Price could come from market state or be computed
            current_price = getattr(self, 'current_price', 0.3)
            market_reward = float(
                compute_grid_reward(total_supply, total_demand, current_price)
            )

        step_reward = market_reward
        self.total_reward += step_reward
        self._update_grid_state()
        self._update_market_state()

        # --- termination --------------------------------------------------
        terminated = False
        truncated = self.episode_length >= self.max_episode_steps

        # --- logging ------------------------------------------------------
        log_env_info(
            self.episode_count,
            self.episode_length,
            self.current_time,
            step_reward,
        )

        # --- info dict (diagnostics) --------------------------------------
        info = {
            "episode_count": self.episode_count,
            "episode_length": self.episode_length,
            "current_time": self.current_time,
            "step_reward": step_reward,
            "total_reward": self.total_reward,
            "weather_index_used": weather_idx,
            "coordination_signals": coordination_signals,
        }

        return self._get_observation(), step_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the current observation dict.

        ASSUMPTION: house.get_state() exists and returns an array-like of
        shape ``(10,)`` matching ``observation_space["house_states"]``.
        """
        house_states = [house.get_state() for house in self.house_environments]

        return {
            "house_states": np.array(house_states, dtype=np.float32),
            "grid_state": self._get_grid_state(),
            "market_state": self._get_market_state(),
        }

    def _get_grid_state(self) -> np.ndarray:
        """
        Get the current grid state (e.g., voltage, line losses).

        Returns:
            np.ndarray of shape ``(num_households, 10)``

        NOTE: placeholder — replace with actual grid-state computation.
        """
        return self._grid_state.copy()

    def _get_market_state(self) -> np.ndarray:
        """
        Get the current market state (e.g., energy prices, demand, supply).

        Returns:
            np.ndarray of shape ``(num_households, 2)``

        NOTE: placeholder — replace with actual market-state computation.
        """
        return self._market_state.copy()

    def _update_grid_state(self) -> None:
        """Update cached grid state when environment state changes."""
        self._grid_state = self.np_random.uniform(
            size=(self.num_households, 10)
        ).astype(np.float32)

    def _update_market_state(self) -> None:
        """Update cached market state when environment state changes."""
        self._market_state = self.np_random.uniform(
            size=(self.num_households, 2)
        ).astype(np.float32)

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
