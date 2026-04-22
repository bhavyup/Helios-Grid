"""Deterministic market model used by simulation and orchestration layers."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class MarketModel:
    """Lightweight deterministic market model for Phase 1 simulation flows.

    The model intentionally stays simple but production-safe:
    - deterministic under fixed inputs,
    - backward-compatible with existing orchestrator usage,
    - explicit state fields for observability.
    """

    def __init__(
        self,
        default_price: float = 0.3,
        price_min: float = 0.1,
        price_max: float = 1.0,
        max_price_step: float = 0.05,
        **_: Any,
    ) -> None:
        self.default_price = float(default_price)
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.max_price_step = abs(float(max_price_step))

        self.clearing_price = self.default_price
        self.grid_balance = 0.0
        self.market_balance = 0.0
        self.household_consumption = 0.0
        self.solar_production = 0.0
        self.wind_production = 0.0
        self.last_snapshot: Dict[str, float] = {}

        self.reset()

    def __repr__(self) -> str:
        return (
            "MarketModel("
            f"price={self.clearing_price:.4f}, "
            f"balance={self.grid_balance:.4f}"
            ")"
        )

    def get_price(self, *args: Any, **kwargs: Any) -> float:
        """Return the latest clearing price."""
        return float(self.clearing_price)

    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Reset market state and return the initial snapshot."""
        self.clearing_price = float(self.default_price)
        self.grid_balance = 0.0
        self.market_balance = 0.0
        self.household_consumption = 0.0
        self.solar_production = 0.0
        self.wind_production = 0.0

        self.last_snapshot = {
            "supply": 0.0,
            "demand": 0.0,
            "effective_supply": 0.0,
            "effective_demand": 0.0,
            "imbalance": 0.0,
            "traded_volume": 0.0,
            "unmatched_supply": 0.0,
            "unmatched_demand": 0.0,
            "clearing_price": float(self.clearing_price),
            "price_delta": 0.0,
            "market_action": 0.0,
            "grid_balance": 0.0,
            "market_balance": 0.0,
            "household_consumption": 0.0,
            "solar_production": 0.0,
            "wind_production": 0.0,
        }
        return dict(self.last_snapshot)

    def step(
        self,
        *args: Any,
        supply: float | None = None,
        demand: float | None = None,
        market_action: int = 1,
        households: Iterable[Dict[str, Any]] | None = None,
        solar: float | None = None,
        wind: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Advance the market by one timestep.

        Supports both call styles used across the repository:
        1) ``step(supply=..., demand=..., market_action=...)``
        2) ``step(households=..., solar=..., wind=...)``
        """
        derived_supply, derived_demand, solar_production, wind_production = (
            self._derive_flows(
                supply=supply,
                demand=demand,
                households=households,
                solar=solar,
                wind=wind,
            )
        )

        resolved_market_action = 1 if int(market_action) >= 1 else 0

        if resolved_market_action == 1:
            effective_supply = derived_supply
            effective_demand = derived_demand
            traded_volume = min(derived_supply, derived_demand)
        else:
            # Holding in market mode is represented as lower local matching.
            effective_supply = derived_supply * 0.95
            effective_demand = derived_demand * 1.05
            traded_volume = 0.0

        imbalance = effective_supply - effective_demand
        base_flow = max(effective_supply, effective_demand, 1e-6)
        pressure = (effective_demand - effective_supply) / base_flow
        price_delta = max(
            -self.max_price_step,
            min(self.max_price_step, pressure * 0.1),
        )

        self.clearing_price = self._clamp(
            self.clearing_price + price_delta,
            self.price_min,
            self.price_max,
        )

        unmatched_supply = max(effective_supply - effective_demand, 0.0)
        unmatched_demand = max(effective_demand - effective_supply, 0.0)

        self.grid_balance = imbalance
        self.market_balance = imbalance
        self.household_consumption = derived_demand
        self.solar_production = solar_production
        self.wind_production = wind_production

        self.last_snapshot = {
            "supply": float(derived_supply),
            "demand": float(derived_demand),
            "effective_supply": float(effective_supply),
            "effective_demand": float(effective_demand),
            "imbalance": float(imbalance),
            "traded_volume": float(traded_volume),
            "unmatched_supply": float(unmatched_supply),
            "unmatched_demand": float(unmatched_demand),
            "clearing_price": float(self.clearing_price),
            "price_delta": float(price_delta),
            "market_action": float(resolved_market_action),
            "grid_balance": float(self.grid_balance),
            "market_balance": float(self.market_balance),
            "household_consumption": float(self.household_consumption),
            "solar_production": float(self.solar_production),
            "wind_production": float(self.wind_production),
        }
        return dict(self.last_snapshot)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _derive_flows(
        supply: float | None,
        demand: float | None,
        households: Iterable[Dict[str, Any]] | None,
        solar: float | None,
        wind: float | None,
    ) -> tuple[float, float, float, float]:
        if supply is not None and demand is not None:
            solar_production = max(float(solar) if solar is not None else 0.0, 0.0)
            wind_production = max(float(wind) if wind is not None else 0.0, 0.0)
            return (
                max(float(supply), 0.0),
                max(float(demand), 0.0),
                solar_production,
                wind_production,
            )

        resolved_households = list(households or [])
        household_demand = 0.0
        for household in resolved_households:
            consumption = household.get("consumption", 0.0)
            household_demand += max(float(consumption), 0.0)

        solar_production = max(float(solar) if solar is not None else 0.0, 0.0)
        wind_production = max(float(wind) if wind is not None else 0.0, 0.0)
        aggregate_supply = solar_production + wind_production

        return (
            float(aggregate_supply),
            float(household_demand),
            float(solar_production),
            float(wind_production),
        )