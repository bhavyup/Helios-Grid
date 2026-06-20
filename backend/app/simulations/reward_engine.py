"""RewardEngine computes grid-level rewards from market snapshots."""

from typing import Any

from app.domain.rewards.reward_utils import compute_grid_reward


class RewardEngine:
    """Compute step rewards for the grid environment."""

    def compute(self, market_snapshot: dict[str, Any]) -> float:
        supply = float(
            market_snapshot.get("effective_supply", market_snapshot.get("supply", 0.0))
        )
        demand = float(
            market_snapshot.get("effective_demand", market_snapshot.get("demand", 0.0))
        )
        price = float(market_snapshot.get("clearing_price", 0.0))
        return float(compute_grid_reward(supply=supply, demand=demand, price=price))
