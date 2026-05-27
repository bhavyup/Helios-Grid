"""MarketEngine coordinates the MarketModel and market snapshots."""

from typing import Dict

import numpy as np

from app.core.project_config import config
from app.domain.models.market_model import MarketModel


class MarketEngine:
    """Initialize and step the market model."""

    def __init__(self) -> None:
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
        self.last_snapshot = self.market_model.reset()

    def reset(self) -> Dict[str, float]:
        self.last_snapshot = self.market_model.reset()
        return dict(self.last_snapshot)

    def step(
        self,
        supply: float,
        demand: float,
        market_action: int,
        solar: float,
        wind: float,
    ) -> Dict[str, float]:
        self.last_snapshot = self.market_model.step(
            supply=supply,
            demand=demand,
            market_action=int(market_action),
            solar=solar,
            wind=wind,
        )
        return dict(self.last_snapshot)

    def get_market_state(self, num_households: int) -> np.ndarray:
        price = float(self.last_snapshot.get("clearing_price", 0.0))
        imbalance = float(self.last_snapshot.get("imbalance", 0.0))
        market_vector = np.array([price, imbalance], dtype=np.float32)
        return np.tile(market_vector, (num_households, 1))
