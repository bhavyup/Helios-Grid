import numpy as np
from typing import Dict, List, Tuple, Any

from gymnasium import Env
from gymnasium.spaces import Dict as GymDict, Box, Discrete
from gymnasium.utils import seeding

from app.core.project_config import config
from app.envs.house_env import HouseEnv
from app.models.gnn_coordinator import GNNCoordinator
from app.models.market_model import MarketModel
from app.utils.data_utils import load_weather_data
from app.utils.graph_utils import build_grid_graph
from app.utils.logging_utils import log_env_info
from app.utils.reward_utils import compute_grid_reward


class GridEnv(Env):
    """
    Top-level grid environment that orchestrates household sub-environments,
    a GNN coordinator, weather data, and market actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_topology_file: str = "data/grid_topology/sample_grid.json",
        weather_file: str = "data/weather_data/sample_weather.csv",
        num_households: int = config["num_households"],
        max_episode_steps: int = config["training_steps"],
    ):
        """
        Initialize the main grid environment.

        Args:
            grid_topology_file: Path to grid topology file (JSON).
            weather_file: Path to weather data file (CSV).
            num_households: Number of households in the grid.
            max_episode_steps: Maximum number of timesteps per episode.
                NOTE: sourced from config["training_steps"] by default.
                The config key name is preserved for backward compatibility.
        """
        super().__init__()

        self.grid_topology_file = grid_topology_file
        self.weather_file = weather_file
        self.num_households = num_households
        self.max_episode_steps = max_episode_steps

        # Load grid topology and build graph
        self.graph = build_grid_graph(grid_topology_file, num_households)
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

        market_cfg = config.get("market", {})
        default_price = float(
            market_cfg.get("default_price", 0.3)
            if hasattr(market_cfg, "get")
            else 0.3
        )
        price_min = float(
            market_cfg.get("price_min", 0.1)
            if hasattr(market_cfg, "get")
            else 0.1
        )
        price_max = float(
            market_cfg.get("price_max", 1.0)
            if hasattr(market_cfg, "get")
            else 1.0
        )
        self.market_model = MarketModel(
            default_price=default_price,
            price_min=price_min,
            price_max=price_max,
        )
        self.last_market_snapshot = self.market_model.reset()
        self.last_coordination_summary = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

        # ------------------------------------------------------------------
        # Action and observation spaces
        # ------------------------------------------------------------------
        self.action_space = GymDict({
            "house_actions": Box(
                low=0, high=1, shape=(num_households, 6), dtype=np.float32
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

        # Deterministic RNG — seeded lazily; call seed() to set explicitly
        self.np_random = None
        self.seed()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed(self, seed=None):
        """Seed all internal RNGs for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        # Gymnasium may return very large seed integers; normalize to keep
        # downstream libraries (notably torch) in a safe integer range.
        base_seed = int(seed) % (2**31 - 1)

        for index, house in enumerate(self.house_environments):
            house.seed(base_seed + index + 1)

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

        for house in self.house_environments:
            house.reset()

        self.gnn_coordinator.reset()
        self.last_market_snapshot = self.market_model.reset()
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
        """
        Execute one timestep.

        Args:
            actions: Dict with keys ``"house_actions"`` and ``"market_actions"``.

        Returns:
            (observation, reward, done, info)
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
        # ASSUMPTION: HouseEnv.step() returns a state array (not a standard
        #   gym 4-tuple).  If HouseEnv follows the gym convention
        #   (obs, reward, done, info), this line must unpack the tuple and
        #   house-level rewards should be accumulated.
        house_step_results = []
        for i, house in enumerate(self.house_environments):
            result = house.step(house_actions[i])
            house_step_results.append(result)

        extracted_states = [
            self._extract_house_state(result)
            for result in house_step_results
        ]

        # --- GNN coordination --------------------------------------------
        # TODO: coordination_signals are computed but not yet consumed.
        #       Wire into house sub-envs or include in observation when the
        #       coordination protocol is defined.
        coordination_signals = self.gnn_coordinator.compute_coordination_signals(
            house_step_results, self.graph, self.weather_data[weather_idx]
        )
        self.last_coordination_summary = self._summarize_coordination_signals(
            coordination_signals
        )

        aggregate = self._aggregate_house_states(extracted_states)
        self.last_market_snapshot = self.market_model.step(
            supply=aggregate["supply"],
            demand=aggregate["demand"],
            market_action=int(market_actions),
            solar=aggregate["supply"],
            wind=0.0,
        )

        # Advance time *after* consuming the current weather datum
        self.current_time += 1

        # --- reward -------------------------------------------------------
        step_reward = float(
            compute_grid_reward(
                supply=float(self.last_market_snapshot["effective_supply"]),
                demand=float(self.last_market_snapshot["effective_demand"]),
                price=float(self.last_market_snapshot["clearing_price"]),
            )
        )
        self.total_reward += step_reward

        # --- termination --------------------------------------------------
        done = self.episode_length >= self.max_episode_steps

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
            "coordination_summary": self.last_coordination_summary,
            "market_snapshot": self.last_market_snapshot,
        }

        return self._get_observation(), step_reward, done, info

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

    @staticmethod
    def _extract_house_state(step_result: Any) -> np.ndarray:
        """Normalize a house step result into a (10,) float32 state vector."""
        if isinstance(step_result, tuple):
            step_result = step_result[0]
        return np.asarray(step_result, dtype=np.float32)

    def _get_grid_state(self) -> np.ndarray:
        """
        Get the current grid state (e.g., voltage, line losses).

        Returns:
            np.ndarray of shape ``(num_households, 10)``

        Returns deterministic aggregate metrics derived from current
        household states, repeated per household for backward compatibility
        with the current observation space shape.
        """
        if not self.house_environments:
            return np.zeros((self.num_households, 10), dtype=np.float32)

        house_states = np.asarray(
            [house.get_state() for house in self.house_environments],
            dtype=np.float32,
        )
        if house_states.size == 0:
            return np.zeros((self.num_households, 10), dtype=np.float32)

        total_energy = float(np.sum(house_states[:, 0]))
        total_demand = float(np.sum(house_states[:, 1]))
        total_supply = float(np.sum(house_states[:, 2]))
        avg_battery = float(np.mean(house_states[:, 3]))
        avg_price = float(np.mean(house_states[:, 4]))
        total_grid_import = float(np.sum(house_states[:, 5]))
        total_p2p_buy = float(np.sum(house_states[:, 6]))
        total_p2p_sell = float(np.sum(house_states[:, 7]))
        net_balance = float(np.sum(house_states[:, 9]))
        renewable_utilization = 0.0
        if total_demand > 0.0:
            renewable_utilization = float(
                np.clip(total_supply / total_demand, 0.0, 1.0)
            )

        timestep_norm = float(
            min(self.current_time, self.max_episode_steps)
            / max(self.max_episode_steps, 1)
        )

        aggregate_vector = np.array(
            [
                total_energy,
                total_demand,
                total_supply,
                avg_battery,
                avg_price,
                total_grid_import,
                total_p2p_buy,
                total_p2p_sell,
                timestep_norm,
                net_balance + renewable_utilization,
            ],
            dtype=np.float32,
        )
        return np.tile(aggregate_vector, (self.num_households, 1))

    def _get_market_state(self) -> np.ndarray:
        """
        Get the current market state (e.g., energy prices, demand, supply).

        Returns:
            np.ndarray of shape ``(num_households, 2)``

        Returns deterministic market metrics repeated per household for
        backward compatibility with current observation shape.
        """
        price = float(self.last_market_snapshot.get("clearing_price", 0.0))
        imbalance = float(self.last_market_snapshot.get("imbalance", 0.0))
        market_vector = np.array([price, imbalance], dtype=np.float32)
        return np.tile(market_vector, (self.num_households, 1))

    @staticmethod
    def _aggregate_house_states(house_states: List[np.ndarray]) -> Dict[str, float]:
        if not house_states:
            return {"demand": 0.0, "supply": 0.0}

        states = np.asarray(house_states, dtype=np.float32)
        return {
            "demand": float(np.sum(states[:, 1])),
            "supply": float(np.sum(states[:, 2])),
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