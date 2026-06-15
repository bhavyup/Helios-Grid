"""
CoordinatorAgent -- Grid simulation orchestrator.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.project_config import config
from app.domain.models.gnn_coordinator import GNNCoordinator
from app.domain.models.market_model import MarketModel
from app.infrastructure.graph_utils import create_grid_graph
from app.infrastructure.logging_utils import log_simulation_data

try:
    from app.infrastructure.communication_layer import CommunicationLayer
except ImportError:
    CommunicationLayer = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """Grid simulation orchestrator."""

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

        self.rng = np.random.RandomState(seed)

        topology_graph = create_grid_graph(
            num_households=num_households,
            num_solar_panels=num_solar_panels,
            num_wind_turbines=num_wind_turbines,
        )

        gnn_kwargs: Dict[str, Any] = {
            "graph": topology_graph,
            "log_dir": log_dir,
        }
        if seed is not None:
            gnn_kwargs["seed"] = seed
        self.gnn_coordinator = GNNCoordinator(**gnn_kwargs)

        market_kwargs: Dict[str, Any] = {
            "num_households": num_households,
            "num_solar_panels": num_solar_panels,
            "num_wind_turbines": num_wind_turbines,
            "log_dir": log_dir,
        }
        try:
            self.market_model = MarketModel(**market_kwargs)
        except TypeError:
            self.market_model = MarketModel()

        self._comm_layer = comm_layer

    def run(
        self,
        num_epochs: int = 100,
        num_steps: int = 100,
    ) -> None:
        self.running = True

        try:
            if self._comm_layer is not None:
                self._comm_layer.start()

            self.gnn_coordinator.run(num_epochs=num_epochs)
            self._simulate_grid(num_steps=num_steps)
        finally:
            self.stop()

    def _simulate_grid(self, num_steps: int = 100) -> None:
        for step in range(num_steps):
            if not self.running:
                logger.info("Simulation interrupted at step %d.", step)
                break

            self.current_time = step

            households: List[Dict[str, float]] = [
                {"consumption": float(self.rng.rand())}
                for _ in range(self.num_households)
            ]
            solar = float(self.rng.rand())
            wind = float(self.rng.rand())

            try:
                market_data: Dict[str, Any] = self.market_model.step(
                    households=households,
                    solar=solar,
                    wind=wind,
                )
            except NotImplementedError:
                market_data = self._fallback_market_step(
                    households=households,
                    solar=solar,
                    wind=wind,
                )

            state = self._market_data_to_state(market_data)
            decision = self.make_decision(state)

            timestamp = datetime.now(tz=timezone.utc).isoformat()

            if self._comm_layer is None:
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
            else:
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

    def make_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        grid_balance = state.get("grid_balance", 0.0)

        if grid_balance > 0:
            return {"action": "increase_price", "value": 0.05}
        if grid_balance < 0:
            return {"action": "decrease_price", "value": -0.05}
        return {"action": "maintain_price", "value": 0.0}

    def get_grid_state(self) -> Dict[str, Any]:
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

    def stop(self) -> None:
        self.running = False
        if self._comm_layer is not None:
            try:
                self._comm_layer.stop()
            except Exception:
                logger.exception("Error stopping communication layer")
        logger.info("CoordinatorAgent stopped.")

    @staticmethod
    def _fallback_market_step(
        households: List[Dict[str, float]],
        solar: float,
        wind: float,
    ) -> Dict[str, float]:
        household_consumption = float(
            sum(h.get("consumption", 0.0) for h in households)
        )
        generation = float(solar + wind)
        return {
            "grid_balance": generation - household_consumption,
            "market_balance": generation - household_consumption,
            "household_consumption": household_consumption,
            "solar_production": solar,
            "wind_production": wind,
        }

    @staticmethod
    def _market_data_to_state(
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "grid_balance": market_data.get("grid_balance", 0.0),
            "market_balance": market_data.get("market_balance", 0.0),
            "household_consumption": market_data.get(
                "household_consumption", 0.0
            ),
            "solar_production": market_data.get("solar_production", 0.0),
            "wind_production": market_data.get("wind_production", 0.0),
        }
