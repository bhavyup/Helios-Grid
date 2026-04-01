"""
HouseAgent — Rule-based household decision agent.

ARCHITECTURAL NOTE
==================
This class combines agent-like behavior (``make_decision``) with
environment-like state transitions (``update_consumption`` depleting
an energy budget).

If the project also provides ``HouseEnv`` (used by ``grid_env.py``),
this class's internal energy tracking will diverge from the
environment's state.  A future refactor should either:

* make ``HouseAgent`` a pure policy that acts on a ``HouseEnv``, or
* make ``HouseEnv`` delegate transition logic to ``HouseAgent``.

Both should NOT independently track energy for the same household.

DEPENDENCY ASSUMPTIONS (unverified)
====================================
* ``CommunicationLayer`` is injectable.  If *None*, the agent runs
  without TCP infrastructure.
* ``log_simulation_data(log_dir, timestamp, ...)`` exists at
  ``utils.logging_utils``.  Signature unverified.
* ``config.LOG_DIR`` — attribute-style access.  Unverified.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.project_config import config

# Only imported if the agent needs to log independently.
# In a well-integrated system the orchestrator handles logging;
# the agent should not need this.  Retained for backward compat.
from app.utils.logging_utils import log_simulation_data

logger = logging.getLogger(__name__)


class HouseAgent:
    """
    Rule-based household agent.

    Policy: consumption decreases linearly as price increases::

        consumption = max_consumption × (1 − price / price_ceiling)

    Energy is bounded at zero — the agent cannot consume more than
    its remaining energy.

    Parameters
    ----------
    house_id : int
        Unique identifier for this household.
    initial_energy : float
        Starting energy budget (kWh or abstract units).
    max_consumption : float
        Maximum consumption per timestep.
    price_ceiling : float
        Price at which consumption drops to zero.
    log_dir : str
        Directory for simulation logs.
    comm_layer : optional
        Pre-built communication layer.  If *None*, the agent operates
        without TCP infrastructure.
    seed : int, optional
        RNG seed for deterministic behavior in standalone ``run()``.
    """

    def __init__(
        self,
        house_id: int = 1,
        initial_energy: float = 100.0,
        max_consumption: float = 5.0,
        price_ceiling: float = 10.0,
        log_dir: str = config.LOG_DIR,
        comm_layer: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        self.house_id = house_id
        self.initial_energy = initial_energy
        self.energy = initial_energy
        self.max_consumption = max_consumption
        self.price_ceiling = price_ceiling
        self.log_dir = log_dir
        self.running: bool = False

        self.consumption_history: List[Tuple[int, float]] = []
        self.price_history: List[Tuple[int, float]] = []

        # --- deterministic RNG -------------------------------------------
        self.rng = np.random.RandomState(seed)

        # --- communication (optional) ------------------------------------
        self._comm_layer = comm_layer

    # ==================================================================
    # Core decision / transition logic
    # ==================================================================

    def update_consumption(self, price: float, time_step: int) -> float:
        """
        Compute consumption for this step and deplete energy.

        Consumption is clipped to ``[0, min(max_consumption, energy)]``
        so energy never goes negative.

        Args:
            price: Current energy price.
            time_step: Integer timestep index.

        Returns:
            Actual consumption (float ≥ 0).
        """
        raw = self.max_consumption * (1.0 - price / self.price_ceiling)
        # Upper-bound: cannot consume more than remaining energy
        upper = min(self.max_consumption, max(self.energy, 0.0))
        consumption = float(np.clip(raw, 0.0, upper))

        self.energy -= consumption
        self.consumption_history.append((time_step, consumption))
        self.price_history.append((time_step, float(price)))

        return consumption

    def make_decision(
        self, price: float, time_step: int
    ) -> Dict[str, Any]:
        """
        Decide how much to consume at the given price.

        This is the primary agent interface.  It calls
        ``update_consumption`` internally — callers should NOT call
        both.

        Args:
            price: Current energy price.
            time_step: Integer timestep index.

        Returns:
            Dict with keys ``house_id``, ``time``, ``consumption``,
            ``price``, ``energy``.
        """
        consumption = self.update_consumption(price, time_step)
        return {
            "house_id": self.house_id,
            "time": time_step,
            "consumption": consumption,
            "price": float(price),
            "energy": self.energy,
        }

    # ==================================================================
    # State query
    # ==================================================================

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the agent's current state."""
        return {
            "house_id": self.house_id,
            "current_energy": self.energy,
            "consumption_history": list(self.consumption_history),
            "price_history": list(self.price_history),
        }

    def get_consumption_history(self) -> List[Tuple[int, float]]:
        """Return the full consumption history as (timestep, value) pairs."""
        return list(self.consumption_history)

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.energy = self.initial_energy
        self.consumption_history.clear()
        self.price_history.clear()

    # ==================================================================
    # Communication (optional)
    # ==================================================================

    def communicate(self, message: Dict[str, Any]) -> None:
        """
        Handle an inbound message.

        WARNING: this calls ``make_decision`` and therefore **mutates
        agent state** (energy, histories).  It is a side effect of
        receiving a message.

        Args:
            message: Expected keys ``"component_type"``, ``"price"``,
                     ``"time_step"``.
        """
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
            self._comm_layer.send_message({
                "component_type": "agent",
                "agent_id": self.house_id,
                "time_step": time_step,
                "price": decision["price"],
                "consumption": decision["consumption"],
                "energy": decision["energy"],
            })

    # ==================================================================
    # Standalone run loop
    # ==================================================================

    def run(self, num_steps: int = 100) -> None:
        """
        Self-contained simulation loop (for standalone testing).

        Uses seeded RNG for deterministic price generation.
        Checks ``self.running`` each iteration so ``stop()``
        can interrupt from another thread.

        NOTE: in a full simulation, the orchestrator or environment
        should drive the agent — this method is for isolated testing.
        """
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

                # --- send via comm layer if available --------------------
                if self._comm_layer is not None:
                    self._comm_layer.send_message({
                        "component_type": "agent",
                        "agent_id": self.house_id,
                        "time_step": step,
                        "price": decision["price"],
                        "consumption": decision["consumption"],
                        "energy": decision["energy"],
                    })

                logger.info(
                    "House %d — Step %d/%d — "
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

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self.running = False
        logger.info("House Agent %d stopped.", self.house_id)