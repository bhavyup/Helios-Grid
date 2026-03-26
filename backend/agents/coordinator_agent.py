"""
CoordinatorAgent — Grid simulation orchestrator.

ARCHITECTURAL NOTE
==================
Despite the class name, this is a **simulation orchestrator /
controller**, not an RL agent.  It:

* owns and initialises subsystems (GNN coordinator, market model)
* optionally starts a communication layer
* runs a fixed-length simulation loop
* applies a rule-based decision heuristic
* logs simulation state

It does NOT:

* observe an environment through a gym-style interface
* maintain a learned policy trained via rewards
* participate in MARL as a peer agent

A future refactor should consider renaming this to
``SimulationOrchestrator`` or ``GridController`` and moving it
out of ``agents/``.

DEPENDENCY ASSUMPTIONS (unverified — require source files)
==========================================================
* ``GNNCoordinator(num_households, num_solar_panels,
  num_wind_turbines, log_dir)`` — constructor signature.
  NOTE: the refactored ``GNNCoordinator`` may instead accept a
  NetworkX ``graph`` as its first positional arg.  Verify
  ``models/gnn_coordinator.py``.

* ``MarketModel(num_households, num_solar_panels,
  num_wind_turbines, log_dir)`` — constructor and
  ``.step(households, solar, wind)`` returning a dict with keys
  ``grid_balance``, ``market_balance``, ``household_consumption``,
  ``solar_production``, ``wind_production``.
  Unverified — requires ``market_model.py``.

* ``MarketModel`` attribute access (``.grid_balance`` etc.) used
  in ``get_grid_state()``.  Unverified.

* ``config.LOG_DIR`` — attribute-style access.  Other files use
  ``config['key']`` (dict).  Verify ``config.py``.

* ``log_simulation_data`` keyword-arg signature.
  Unverified — requires ``utils/logging_utils.py``.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from config import config

# ASSUMPTION: import paths aligned with grid_env.py convention.
# Adjust if the actual package layout differs.
from utils.logging_utils import log_simulation_data
from models.gnn_coordinator import GNNCoordinator

# ASSUMPTION: MarketModel lives here.  Unverified.
from models.market_model import MarketModel

# CommunicationLayer is optional infrastructure.
# Import is guarded so the module loads even if the comm layer
# has not been implemented or has been moved.
try:
    from infra.communication import CommunicationLayer  # ASSUMPTION: path
except ImportError:
    CommunicationLayer = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Grid simulation orchestrator (see module docstring).

    Parameters
    ----------
    num_households : int
    num_solar_panels : int
    num_wind_turbines : int
    log_dir : str
        Directory for simulation logs.
    comm_layer : optional
        Pre-built ``CommunicationLayer``.  If *None* (default), the
        simulation runs without TCP infrastructure.
    seed : int, optional
        RNG seed for deterministic simulation inputs.
    """

    def __init__(
        self,
        num_households: int = 10,
        num_solar_panels: int = 5,
        num_wind_turbines: int = 3,
        log_dir: str = config.LOG_DIR,
        comm_layer: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        self.num_households = num_households
        self.num_solar_panels = num_solar_panels
        self.num_wind_turbines = num_wind_turbines
        self.log_dir = log_dir
        self.current_time: int = 0
        self.running: bool = False

        # --- deterministic RNG -------------------------------------------
        self.rng = np.random.RandomState(seed)

        # --- subsystems --------------------------------------------------
        # ASSUMPTION: constructor signatures match.
        self.gnn_coordinator = GNNCoordinator(
            num_households=num_households,
            num_solar_panels=num_solar_panels,
            num_wind_turbines=num_wind_turbines,
            log_dir=log_dir,
        )

        self.market_model = MarketModel(
            num_households=num_households,
            num_solar_panels=num_solar_panels,
            num_wind_turbines=num_wind_turbines,
            log_dir=log_dir,
        )

        # Communication layer is optional — avoids coupling the
        # simulation loop to TCP infrastructure when it is not needed.
        self._comm_layer = comm_layer

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(
        self,
        num_epochs: int = 100,
        num_steps: int = 100,
    ) -> None:
        """
        Run GNN pre-training followed by the simulation loop.

        The communication layer (if present) is started before and
        stopped after, even on error.
        """
        self.running = True

        if self._comm_layer is not None:
            self._comm_layer.start()

        try:
            # GNN trains to completion before simulation.  If
            # interleaved training is required, restructure later.
            self.gnn_coordinator.run(num_epochs=num_epochs)
            self._simulate_grid(num_steps=num_steps)
        finally:
            self.stop()

    # ==================================================================
    # Simulation loop
    # ==================================================================

    def _simulate_grid(self, num_steps: int = 100) -> None:
        """
        Fixed-length simulation loop.

        Checks ``self.running`` each iteration so ``stop()`` can
        interrupt from another thread.
        """
        for step in range(num_steps):
            if not self.running:
                logger.info("Simulation interrupted at step %d.", step)
                break

            self.current_time = step

            # --- generate inputs (seeded) --------------------------------
            households: List[Dict[str, float]] = [
                {"consumption": float(self.rng.rand())}
                for _ in range(self.num_households)
            ]
            solar = float(self.rng.rand())
            wind = float(self.rng.rand())

            # --- step market model ---------------------------------------
            # ASSUMPTION: returns dict with keys grid_balance,
            # market_balance, household_consumption, solar_production,
            # wind_production.
            market_data: Dict[str, Any] = self.market_model.step(
                households=households,
                solar=solar,
                wind=wind,
            )

            # --- rule-based decision -------------------------------------
            state = self._market_data_to_state(market_data)
            decision = self.make_decision(state)
            # TODO: apply decision to market_model once its API for
            # accepting external price adjustments is defined.
            # Currently logged but not acted upon.

            # --- single timestamp for this logical step ------------------
            # NOTE: wall-clock timestamp is non-deterministic.  For
            # fully deterministic logs, consider replacing with a
            # simulation-clock value (e.g. step index).
            timestamp = datetime.now(tz=timezone.utc).isoformat()

            # --- authoritative log (direct call) -------------------------
            log_simulation_data(
                log_dir=self.log_dir,
                timestamp=timestamp,
                grid_balance=market_data.get("grid_balance", 0.0),
                market_balance=market_data.get("market_balance", 0.0),
                household_consumption=market_data.get(
                    "household_consumption", 0.0
                ),
                solar_production=market_data.get("solar_production", 0.0),
                wind_production=market_data.get("wind_production", 0.0),
            )

            # --- forward to communication layer (if present) -------------
            # NOTE: if the comm layer's internal message handler also
            # calls log_simulation_data for "grid" messages, this will
            # produce duplicate log entries.  The direct call above is
            # the authoritative record.
            if self._comm_layer is not None:
                self._comm_layer.send_message({
                    "component_type": "grid",
                    "timestamp": timestamp,
                    "grid_balance": market_data.get("grid_balance", 0.0),
                    "market_balance": market_data.get("market_balance", 0.0),
                    "household_consumption": market_data.get(
                        "household_consumption", 0.0
                    ),
                    "solar_production": market_data.get(
                        "solar_production", 0.0
                    ),
                    "wind_production": market_data.get(
                        "wind_production", 0.0
                    ),
                    "decision": decision,
                })

            logger.info(
                "Step %d/%d complete | balance=%.4f | decision=%s",
                step + 1,
                num_steps,
                market_data.get("grid_balance", 0.0),
                decision.get("action", "unknown"),
            )

    # ==================================================================
    # Decision logic
    # ==================================================================

    def make_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based price-adjustment heuristic.

        Args:
            state: Must contain ``"grid_balance"`` (float).

        Returns:
            Dict with ``"action"`` (str) and ``"value"`` (float).
        """
        grid_balance = state.get("grid_balance", 0.0)

        if grid_balance > 0:
            return {"action": "increase_price", "value": 0.05}
        elif grid_balance < 0:
            return {"action": "decrease_price", "value": -0.05}
        return {"action": "maintain_price", "value": 0.0}

    # ==================================================================
    # State query
    # ==================================================================

    def get_grid_state(self) -> Dict[str, Any]:
        """
        Snapshot of current grid state from the market model.

        ASSUMPTION: ``MarketModel`` exposes these instance attributes.
        ``getattr`` with defaults prevents ``AttributeError`` if the
        contract is wrong.
        """
        return {
            "current_time": self.current_time,
            "grid_balance": getattr(
                self.market_model, "grid_balance", 0.0
            ),
            "market_balance": getattr(
                self.market_model, "market_balance", 0.0
            ),
            "household_consumption": getattr(
                self.market_model, "household_consumption", 0.0
            ),
            "solar_production": getattr(
                self.market_model, "solar_production", 0.0
            ),
            "wind_production": getattr(
                self.market_model, "wind_production", 0.0
            ),
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def stop(self) -> None:
        """Signal the simulation to stop and release resources."""
        self.running = False
        if self._comm_layer is not None:
            try:
                self._comm_layer.stop()
            except Exception:
                logger.exception("Error stopping communication layer")
        logger.info("CoordinatorAgent stopped.")

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _market_data_to_state(
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalise market-model step output into a state dict
        suitable for ``make_decision``.

        Uses ``.get()`` with defaults so a missing key produces a
        safe zero rather than a ``KeyError``.
        """
        return {
            "grid_balance": market_data.get("grid_balance", 0.0),
            "market_balance": market_data.get("market_balance", 0.0),
            "household_consumption": market_data.get(
                "household_consumption", 0.0
            ),
            "solar_production": market_data.get("solar_production", 0.0),
            "wind_production": market_data.get("wind_production", 0.0),
        }