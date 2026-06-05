"""Household environment used by GridEnv for per-agent local dynamics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from app.core.project_config import config


class HouseEnv(Env):
    """Simple, deterministic-under-seed household environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_energy: float = 100.0,
        max_battery: float = float(config.get("max_battery", 10.0)),
        base_price: float = float(config.market.default_price),
        max_episode_steps: int = int(config.get("training_steps", 1000)),
    ):
        super().__init__()

        self.initial_energy = float(initial_energy)
        self.max_battery = float(max_battery)
        self.base_price = float(base_price)
        self.max_episode_steps = max(1, int(max_episode_steps))

        self.max_charge_rate = 1.0
        self.max_discharge_rate = 1.0

        self.action_space = Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        self.np_random = None
        self.seed()

        self.current_time = 0
        self.energy = self.initial_energy
        self.battery_level = self.max_battery * 0.5
        self.last_price = self.base_price
        self.last_consumption = 0.0
        self.last_production = 0.0
        self.last_grid_import = 0.0
        self.last_p2p_buy = 0.0
        self.last_p2p_sell = 0.0
        self._state = np.zeros((10,), dtype=np.float32)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> np.ndarray:
        self.current_time = 0
        self.energy = self.initial_energy
        self.battery_level = self.max_battery * 0.5
        self.last_price = self.base_price
        self.last_consumption = 0.0
        self.last_production = 0.0
        self.last_grid_import = 0.0
        self.last_p2p_buy = 0.0
        self.last_p2p_sell = 0.0
        self._state = self._build_state()
        return self._state.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        action_vec = np.asarray(action, dtype=np.float32).reshape(6)
        action_vec = np.clip(action_vec, 0.0, 1.0)

        self.current_time += 1

        base_load = float(self.np_random.uniform(0.5, 2.0))

        # Optional external household demand signal (per-house baseline)
        try:
            if hasattr(self, "current_consumption_target"):
                target = getattr(self, "current_consumption_target")
                if target is not None:
                    base_load = float(max(0.0, target))
        except Exception:
            pass

        # Prefer weather-driven signals when available on the instance.
        solar_gen = None
        wind_gen = None
        try:
            if hasattr(self, "current_weather") and self.current_weather is not None:
                # Row-like object (pd.Series) supports .get(key)
                try:
                    solar_val = self.current_weather.get("pv_power", None)
                except Exception:
                    solar_val = self.current_weather.get("solar_irradiance", None)

                if solar_val is not None:
                    solar_gen = float(solar_val)

                try:
                    wind_val = self.current_weather.get("wind_speed", None)
                    if wind_val is not None:
                        wind_gen = float(wind_val)
                except Exception:
                    wind_gen = None
        except Exception:
            solar_gen = None
            wind_gen = None

        # Fallback to simple deterministic scaling if weather fields are normalized
        if solar_gen is None:
            # previous random range ~ [0, 1.5]
            solar_gen = float(self.np_random.uniform(0.0, 1.5))
        if wind_gen is None:
            wind_gen = float(self.np_random.uniform(0.0, 0.5))
        production = solar_gen + wind_gen

        buy_signal = float(action_vec[3])
        sell_signal = float(action_vec[4])

        price_ref = float(self.base_price)
        try:
            if hasattr(self, "current_market") and self.current_market is not None:
                # pd.Series-like supports .get
                try:
                    ext_price = self.current_market.get("price", None)
                except Exception:
                    ext_price = None
                if ext_price is None:
                    try:
                        ext_price = self.current_market.get("clearing_price", None)
                    except Exception:
                        ext_price = None
                if ext_price is not None:
                    price_ref = float(ext_price)
        except Exception:
            pass

        # keep it in plausible bounds
        price_ref = float(np.clip(price_ref, 0.05, 2.0))

        price_noise = float(self.np_random.uniform(-0.05, 0.05))
        self.last_price = float(
            np.clip(
                price_ref + price_noise + 0.2 * (buy_signal - sell_signal),
                0.05,
                2.0,
            )
        )

        desired_consumption = float(base_load * (0.5 + action_vec[0]))
        battery_charge = float(action_vec[1] * self.max_charge_rate)
        battery_discharge = float(action_vec[2] * self.max_discharge_rate)

        self.battery_level = float(
            np.clip(
                self.battery_level + battery_charge - battery_discharge,
                0.0,
                self.max_battery,
            )
        )

        available_energy = float(self.energy + production + battery_discharge)
        consumption = float(min(desired_consumption, max(available_energy, 0.0)))

        p2p_buy = buy_signal
        p2p_sell = float(min(sell_signal, production))
        grid_import = float(
            max(consumption - (production + battery_discharge), 0.0) + action_vec[5]
        )

        self.energy = float(
            max(
                self.energy
                + production
                + grid_import
                + p2p_buy
                - consumption
                - p2p_sell,
                0.0,
            )
        )

        self.last_consumption = consumption
        self.last_production = production
        self.last_grid_import = grid_import
        self.last_p2p_buy = p2p_buy
        self.last_p2p_sell = p2p_sell

        self._state = self._build_state()
        return self._state.copy()

    def _build_state(self) -> np.ndarray:
        net_balance = self.last_production - self.last_consumption
        time_norm = float(
            min(self.current_time, self.max_episode_steps) / self.max_episode_steps
        )

        return np.array(
            [
                self.energy,
                self.last_consumption,
                self.last_production,
                self.battery_level,
                self.last_price,
                self.last_grid_import,
                self.last_p2p_buy,
                self.last_p2p_sell,
                time_norm,
                net_balance,
            ],
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def render(self, mode="human"):
        if mode == "human":
            print(
                f"t={self.current_time} energy={self.energy:.2f} "
                f"cons={self.last_consumption:.2f} prod={self.last_production:.2f} "
                f"price={self.last_price:.3f}"
            )

    def close(self):
        return None
