"""
models/market_model.py
======================

Placeholder for the Helios-Grid market model.

This file previously contained a mislabeled duplicate of ``CommunicationLayer``
(which belongs in ``communication_layer.py``).  That duplicate has been removed.

``MarketModel`` is expected by other modules (e.g. ``house_agent.py``) via::

    from market_model import MarketModel

This stub satisfies that import without inventing unspecified market logic.

TODO
----
- Define the market-clearing mechanism (e.g. uniform-price auction, LMP, P2P).
- Define the pricing interface consumed by household agents.
- Decide whether this is a per-timestep pricer, a stateful auction, or a
  policy-driven component.
- Ensure deterministic behavior under fixed seed.
- Move out of ``models/`` if it becomes a service rather than a data model.
"""


class MarketModel:
    """Placeholder for the Helios-Grid energy market model.

    This class accepts constructor arguments and stores them as instance
    attributes. The step() method now returns a dict with placeholder values.

    When the market mechanism is designed, this stub should be replaced
    with the real implementation.
    """

    def __init__(
        self,
        num_households: int = 10,
        num_solar_panels: int = 5,
        num_wind_turbines: int = 3,
        log_dir: str = "logs",
        **kwargs
    ) -> None:
        """Initialize market model with configuration parameters."""
        self.num_households = num_households
        self.num_solar_panels = num_solar_panels
        self.num_wind_turbines = num_wind_turbines
        self.log_dir = log_dir

        # Placeholder state attributes
        self.grid_balance = 0.0
        self.market_balance = 0.0
        self.household_consumption = 0.0
        self.solar_production = 0.0
        self.wind_production = 0.0

    def __repr__(self) -> str:
        return "MarketModel(stub=True)"

    def get_price(self, *args, **kwargs):
        """Return the current energy price.  Not yet implemented."""
        raise NotImplementedError(
            "MarketModel.get_price() is a stub. "
            "No market mechanism has been implemented yet."
        )

    def step(self, households=None, solar=None, wind=None, **kwargs):
        """Advance the market by one timestep.

        Args:
            households: List of household data dicts.
            solar: Solar production value.
            wind: Wind production value.

        Returns:
            Dict with keys: grid_balance, market_balance, household_consumption,
            solar_production, wind_production.
        """
        # Placeholder implementation
        if households:
            self.household_consumption = sum(
                h.get("consumption", 0.0) for h in households
            )
        else:
            self.household_consumption = 0.0

        self.solar_production = float(solar) if solar is not None else 0.0
        self.wind_production = float(wind) if wind is not None else 0.0

        # Simple balance calculation
        total_production = self.solar_production + self.wind_production
        self.grid_balance = total_production - self.household_consumption
        self.market_balance = self.grid_balance  # Simplified

        return {
            "grid_balance": self.grid_balance,
            "market_balance": self.market_balance,
            "household_consumption": self.household_consumption,
            "solar_production": self.solar_production,
            "wind_production": self.wind_production,
        }

    def reset(self, *args, **kwargs):
        """Reset the market state.  Not yet implemented."""
        raise NotImplementedError(
            "MarketModel.reset() is a stub. "
            "No market mechanism has been implemented yet."
        )