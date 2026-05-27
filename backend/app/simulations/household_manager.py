"""HouseholdManager coordinates per-household environment instances."""

from typing import Any, Callable, Dict, List

import numpy as np

from app.envs.house_env import HouseEnv


class HouseholdManager:
    """Manage HouseEnv instances and aggregate their outputs."""

    def __init__(
        self,
        num_households: int,
        house_env_factory: Callable[[], HouseEnv] = HouseEnv,
    ) -> None:
        self.num_households = num_households
        self._house_env_factory = house_env_factory
        self.house_environments = [
            house_env_factory() for _ in range(num_households)
        ]

    def seed(self, base_seed: int) -> None:
        for index, house in enumerate(self.house_environments):
            house.seed(base_seed + index + 1)

    def reset(self) -> None:
        for house in self.house_environments:
            house.reset()

    def step(self, house_actions: np.ndarray) -> List[Any]:
        results: List[Any] = []
        for i, house in enumerate(self.house_environments):
            results.append(house.step(house_actions[i]))
        return results

    def get_states(self) -> List[np.ndarray]:
        return [house.get_state() for house in self.house_environments]

    def get_grid_state(
        self, current_time: int, max_episode_steps: int
    ) -> np.ndarray:
        if not self.house_environments:
            return np.zeros((self.num_households, 10), dtype=np.float32)

        house_states = np.asarray(self.get_states(), dtype=np.float32)
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
            min(current_time, max_episode_steps) / max(max_episode_steps, 1)
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

    @staticmethod
    def extract_state(step_result: Any) -> np.ndarray:
        if isinstance(step_result, tuple):
            step_result = step_result[0]
        return np.asarray(step_result, dtype=np.float32)

    @staticmethod
    def aggregate_states(house_states: List[np.ndarray]) -> Dict[str, float]:
        if not house_states:
            return {"demand": 0.0, "supply": 0.0}

        states = np.asarray(house_states, dtype=np.float32)
        return {
            "demand": float(np.sum(states[:, 1])),
            "supply": float(np.sum(states[:, 2])),
        }
