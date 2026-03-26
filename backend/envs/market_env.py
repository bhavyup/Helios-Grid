"""
MarketEnv — Energy market environment for RL training.

Simulates a simplified energy market where an agent decides to buy
or sell energy at each timestep.  Supply, demand, and price are
grounded in loaded market data when available, with the agent's
action influencing the effective supply/demand balance.

ASSUMPTIONS (unverified — require source files)
================================================
* ``load_market_data(path)`` returns an object that supports:
    - ``len(data)`` → number of timesteps
    - integer row-indexing ``data[t]`` → a dict-like or object with
      keys/attributes ``"supply"``, ``"demand"``, ``"price"``.
  If the return type differs, ``_read_market_row()`` must be adapted.

* ``compute_market_reward(supply, demand, price)`` accepts three
  floats and returns a float.

* ``config`` is importable and provides no keys used directly by
  this file (only passed through to dependencies).
"""

import numpy as np
from typing import Tuple

from gym import Env
from gym.spaces import Box, Discrete
from gym.utils import seeding

from utils.data_utils import load_market_data
from utils.reward_utils import compute_market_reward


class MarketEnv(Env):
    """
    Gym-compatible energy market environment.

    Action space:
        ``Discrete(3)``
            0 — hold  (no market participation)
            1 — buy   (increase demand)
            2 — sell   (increase supply)

    Observation space:
        ``Box(shape=(5,), dtype=np.float32)``
            [supply, demand, energy_price, price_change, net_position]
    """

    metadata = {"render.modes": ["human"]}

    # How much a single buy/sell action shifts supply or demand.
    # Exposed as class attribute so tests / configs can override.
    ACTION_EFFECT_SIZE: float = 5.0

    def __init__(
        self,
        market_data_file: str = "data/market_prices/sample_prices.csv",
    ):
        """
        Args:
            market_data_file: Path to market data CSV.
        """
        super().__init__()

        # ASSUMPTION: load_market_data returns a sequence supporting
        # len() and integer indexing.  See module docstring.
        self.market_data = load_market_data(market_data_file)
        self._max_steps: int = len(self.market_data)

        if self._max_steps == 0:
            raise ValueError(
                f"market_data is empty after loading: {market_data_file}"
            )

        # --- state variables (set properly in reset) ---------------------
        self.current_time: int = 0
        self.total_supply: float = 0.0
        self.total_demand: float = 0.0
        self.energy_price: float = 0.0
        self.prev_price: float = 0.0
        self.net_position: float = 0.0   # cumulative buy(+) / sell(-)
        self.total_reward: float = 0.0

        # --- spaces ------------------------------------------------------
        self.num_actions: int = 3  # hold, buy, sell
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32,
        )

        # --- deterministic RNG -------------------------------------------
        self.np_random = None
        self.seed()

    # ==================================================================
    # Seeding
    # ==================================================================

    def seed(self, seed=None):
        """Seed internal RNG for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ==================================================================
    # Core gym interface
    # ==================================================================

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state."""
        self.current_time = 0
        self.total_supply = 0.0
        self.total_demand = 0.0
        self.energy_price = 0.0
        self.prev_price = 0.0
        self.net_position = 0.0
        self.total_reward = 0.0

        # Read initial market state from data
        self._apply_market_data()

        return self._get_observation()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one timestep.

        Args:
            action: 0 = hold, 1 = buy (increases demand),
                    2 = sell (increases supply).

        Returns:
            (observation, reward, done, info)
        """
        assert self.action_space.contains(action), (
            f"Invalid action {action!r}; expected value in "
            f"{{0, ..., {self.num_actions - 1}}}"
        )

        self.prev_price = self.energy_price

        # Advance time
        self.current_time += 1

        # --- ground supply / demand from market data ---------------------
        self._apply_market_data()

        # --- agent action affects effective supply / demand --------------
        self._apply_action(action)

        # --- price emerges from supply / demand balance ------------------
        self.energy_price = self._compute_price()

        # --- reward ------------------------------------------------------
        # ASSUMPTION: compute_market_reward(supply, demand, price) → float
        reward = float(
            compute_market_reward(
                self.total_supply,
                self.total_demand,
                self.energy_price,
            )
        )
        self.total_reward += reward

        # --- termination -------------------------------------------------
        done = self.current_time >= self._max_steps

        # --- info --------------------------------------------------------
        info = {
            "current_time": self.current_time,
            "action": action,
            "supply": self.total_supply,
            "demand": self.total_demand,
            "energy_price": self.energy_price,
            "price_change": self.energy_price - self.prev_price,
            "net_position": self.net_position,
            "step_reward": reward,
            "total_reward": self.total_reward,
        }

        return self._get_observation(), reward, done, info

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_observation(self) -> np.ndarray:
        """
        Build the current observation vector.

        Contents:
            [0] total_supply
            [1] total_demand
            [2] energy_price
            [3] price_change  (current − previous; 0.0 at t=0)
            [4] net_position  (cumulative buy − sell actions)
        """
        return np.array(
            [
                self.total_supply,
                self.total_demand,
                self.energy_price,
                self.energy_price - self.prev_price,
                self.net_position,
            ],
            dtype=np.float32,
        )

    # ==================================================================
    # Market mechanics
    # ==================================================================

    def _apply_market_data(self) -> None:
        """
        Read base supply, demand, and price from loaded market data
        for the current timestep.

        Falls back to seeded random values if the data row cannot be
        parsed, so the environment never crashes mid-episode.

        ASSUMPTION: each row of ``self.market_data`` supports
        dict-style access with keys ``"supply"``, ``"demand"``,
        ``"price"``.  Adjust ``_read_market_row`` if the format
        differs.
        """
        idx = min(self.current_time, self._max_steps - 1)
        row = self._read_market_row(idx)

        self.total_supply = row["supply"]
        self.total_demand = row["demand"]
        self.energy_price = row["price"]

    def _read_market_row(self, idx: int) -> dict:
        """
        Extract a single timestep from market data.

        Returns a dict with keys ``supply``, ``demand``, ``price``.

        ASSUMPTION: ``self.market_data[idx]`` is dict-like with those
        keys.  If load_market_data returns a DataFrame, bracket
        indexing selects a column — use ``.iloc[idx]`` instead.
        Adapt this method once ``data_utils.py`` is reviewed.
        """
        try:
            row = self.market_data[idx]
            return {
                "supply": float(row["supply"]),
                "demand": float(row["demand"]),
                "price": float(row["price"]),
            }
        except (KeyError, TypeError, IndexError, ValueError):
            # Fallback: seeded random values so the env stays alive
            # and deterministic even if data format is unexpected.
            return {
                "supply": float(self.np_random.uniform(50, 100)),
                "demand": float(self.np_random.uniform(40, 80)),
                "price": float(self.np_random.uniform(0.3, 1.5)),
            }

    def _apply_action(self, action: int) -> None:
        """
        Modify effective supply / demand based on the agent's action.

        0 — hold:  no change
        1 — buy:   increases demand (agent wants energy)
        2 — sell:  increases supply (agent offers energy)
        """
        if action == 1:
            self.total_demand += self.ACTION_EFFECT_SIZE
            self.net_position += 1.0
        elif action == 2:
            self.total_supply += self.ACTION_EFFECT_SIZE
            self.net_position -= 1.0
        # action == 0: hold — no modification

    def _compute_price(self) -> float:
        """
        Derive energy price from current supply / demand balance.

        Uses a simple ratio; replace with a more realistic clearing
        mechanism when the market model matures.
        """
        if self.total_supply <= 0:
            return float(self.np_random.uniform(0.5, 1.5))
        return self.total_demand / self.total_supply

    # ==================================================================
    # Rendering / lifecycle
    # ==================================================================

    def render(self, mode: str = "human") -> None:
        """Render the environment for debugging."""
        print(
            f"Time: {self.current_time}, "
            f"Supply: {self.total_supply:.2f}, "
            f"Demand: {self.total_demand:.2f}, "
            f"Price: {self.energy_price:.4f}, "
            f"Position: {self.net_position:.1f}, "
            f"Reward: {self.total_reward:.4f}"
        )

    def close(self) -> None:
        """Release resources (currently a no-op)."""
        pass