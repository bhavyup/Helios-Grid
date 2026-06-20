"""
HouseAgent -- Rule-based household decision agent.
"""

import logging
from typing import Any

import numpy as np

from app.core.project_config import config

logger = logging.getLogger(__name__)


class HouseAgent:
    """Rule-based household agent."""

    def __init__(
        self,
        house_id: int = 1,
        initial_energy: float = 100.0,
        max_consumption: float = 5.0,
        price_ceiling: float = 10.0,
        log_dir: str = config.LOG_DIR,
        comm_layer: Any | None = None,
        seed: int | None = None,
    ):
        self.house_id = house_id
        self.initial_energy = initial_energy
        self.energy = initial_energy
        self.max_consumption = max_consumption
        self.price_ceiling = price_ceiling
        self.log_dir = log_dir
        self.running: bool = False

        self.consumption_history: list[tuple[int, float]] = []
        self.price_history: list[tuple[int, float]] = []

        self.rng = np.random.RandomState(seed)
        self._comm_layer = comm_layer

    def update_consumption(self, price: float, time_step: int) -> float:
        raw = self.max_consumption * (1.0 - price / self.price_ceiling)
        upper = min(self.max_consumption, max(self.energy, 0.0))
        consumption = float(np.clip(raw, 0.0, upper))

        self.energy -= consumption
        self.consumption_history.append((time_step, consumption))
        self.price_history.append((time_step, float(price)))

        return consumption

    def make_decision(self, price: float, time_step: int) -> dict[str, Any]:
        consumption = self.update_consumption(price, time_step)
        return {
            "house_id": self.house_id,
            "time": time_step,
            "consumption": consumption,
            "price": float(price),
            "energy": self.energy,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "house_id": self.house_id,
            "current_energy": self.energy,
            "consumption_history": list(self.consumption_history),
            "price_history": list(self.price_history),
        }

    def get_consumption_history(self) -> list[tuple[int, float]]:
        return list(self.consumption_history)

    def reset(self) -> None:
        self.energy = self.initial_energy
        self.consumption_history.clear()
        self.price_history.clear()

    def communicate(self, message: dict[str, Any]) -> None:
        if message.get("component_type") != "grid":
            return

        price = message.get("price")
        time_step = message.get("time_step")

        if price is None or time_step is None:
            logger.warning(
                "House %d: communicate() received incomplete message: %s",
                self.house_id,
                message,
            )
            return

        decision = self.make_decision(float(price), int(time_step))

        if self._comm_layer is not None:
            self._comm_layer.send_message(
                {
                    "component_type": "agent",
                    "agent_id": self.house_id,
                    "time_step": time_step,
                    "price": decision["price"],
                    "consumption": decision["consumption"],
                    "energy": decision["energy"],
                }
            )

    def run(self, num_steps: int = 100) -> None:
        self.running = True

        try:
            for step in range(num_steps):
                if not self.running:
                    logger.info(
                        "House %d: run interrupted at step %d.",
                        self.house_id,
                        step,
                    )
                    break

                price = float(self.rng.uniform(0.1, 0.5))
                decision = self.make_decision(price, step)

                if self._comm_layer is not None:
                    self._comm_layer.send_message(
                        {
                            "component_type": "agent",
                            "agent_id": self.house_id,
                            "time_step": step,
                            "price": decision["price"],
                            "consumption": decision["consumption"],
                            "energy": decision["energy"],
                        }
                    )

                logger.info(
                    "House %d -- Step %d/%d -- "
                    "Consumption: %.2f, Energy: %.2f, Price: %.3f",
                    self.house_id,
                    step + 1,
                    num_steps,
                    decision["consumption"],
                    self.energy,
                    price,
                )
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False
        logger.info("House Agent %d stopped.", self.house_id)
