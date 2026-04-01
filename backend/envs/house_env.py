"""
HouseEnv — Household-level environment for RL training.

ARCHITECTURAL NOTE
==================
This environment represents a single household agent in the Helios-Grid
system. It manages battery state, consumption, and production for one
household.

The environment is used by GridEnv to instantiate multiple household
sub-environments.

IMPORT NOTE
===========
CommunicationLayer has been moved to infra/communication.py.
Import it from there if needed:
    from infra.communication import CommunicationLayer
"""

import logging
import numpy as np
from typing import Any, Dict, Tuple
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)


class HouseEnv(Env):
    """
    Gym-compatible household environment.

    Action space:
        Box(shape=(6,), dtype=np.float32)
            [battery_charge, consumption_adjust, unused, unused, unused, unused]
            NOTE: step() currently consumes action[0] and action[1] only;
            action[2:] are accepted by the Box space but ignored.

    Observation space:
        Box(shape=(10,), dtype=np.float32)
            [battery_level, consumption, production, price, ...]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        house_id: int = 0,
        max_battery: float = 10.0,
        initial_battery: float = 5.0,
    ):
        """
        Args:
            house_id: Unique identifier for this household.
            max_battery: Battery capacity (kWh).
            initial_battery: Initial battery charge (kWh).
        """
        super().__init__()
        self.house_id = house_id
        self.max_battery = max_battery
        self.initial_battery = initial_battery

        # State variables
        self.battery_level = initial_battery
        self.consumption = 0.0
        self.production = 0.0
        self.supply = 0.0
        self.demand = 0.0
        self.current_price = 0.3
        self.timestep = 0

        # Action space: 6-dimensional continuous actions
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Observation space: 10-dimensional state vector
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # RNG for reproducibility
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """Seed the environment's RNG."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state (Gymnasium API)."""
        _ = options
        if seed is not None:
            self.seed(seed)
        self.battery_level = self.initial_battery
        self.consumption = 0.0
        self.production = 0.0
        self.supply = 0.0
        self.demand = 0.0
        self.current_price = 0.3
        self.timestep = 0
        observation = self._get_observation()
        info = {"house_id": self.house_id, "timestep": self.timestep}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: Action vector (6-dimensional), where step() currently uses
                action[0] as battery_charge and action[1] as consumption_adjust.
                action[2:] are currently unused.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.timestep += 1

        # Parse action (simplified placeholder)
        battery_charge = np.clip(action[0], -1.0, 1.0)
        consumption_adjust = np.clip(action[1], -1.0, 1.0)

        # Update battery
        self.battery_level = np.clip(
            self.battery_level + battery_charge,
            0.0,
            self.max_battery,
        )

        # Update consumption (placeholder)
        base_consumption = float(self.np_random.uniform(1.0, 5.0))
        self.consumption = base_consumption * (1.0 + consumption_adjust * 0.2)

        # Update production (placeholder)
        self.production = float(self.np_random.uniform(0.0, 3.0))

        # Compute supply/demand
        self.supply = self.production
        self.demand = self.consumption

        # Compute reward (placeholder)
        reward = self.production - self.consumption

        # Episode terminates after a fixed horizon (handled by GridEnv)
        terminated = False
        truncated = False

        info = {
            "house_id": self.house_id,
            "timestep": self.timestep,
            "battery_level": self.battery_level,
            "consumption": self.consumption,
            "production": self.production,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build the current observation vector.

        Returns:
            np.ndarray of shape (10,)
        """
        return np.array(
            [
                self.battery_level,
                self.consumption,
                self.production,
                self.current_price,
                self.supply,
                self.demand,
                self.timestep,
                0.0,  # placeholder
                0.0,  # placeholder
                0.0,  # placeholder
            ],
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        """
        Return the current state (alias for _get_observation for compatibility).

        Used by GridEnv when building observations.
        """
        return self._get_observation()

    def render(self, mode: str = "human") -> None:
        """Render the environment for debugging."""
        print(
            f"House {self.house_id} — Step {self.timestep} — "
            f"Battery: {self.battery_level:.2f}/{self.max_battery:.2f}, "
            f"Consumption: {self.consumption:.2f}, "
            f"Production: {self.production:.2f}"
        )

    def close(self) -> None:
        """Release resources (currently a no-op)."""
        pass
