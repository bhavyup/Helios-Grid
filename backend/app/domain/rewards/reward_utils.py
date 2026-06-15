"""
Reward computation utilities for Helios-Grid.

REWARD COMPONENTS
=================
Helios-Grid uses a multi-level reward structure:

1. **Household reward** -- incentivizes local production, penalizes
   consumption, encourages balanced battery usage.
2. **Market reward** -- incentivizes supply/demand balance and
   price stability.
3. **Grid reward** -- (placeholder) would incentivize grid-level
   stability and efficiency.

TOTAL REWARD
============
The overall scalar reward is a weighted combination of the above.

CONSTANTS
=========
All scaling factors and weights are documented inline.  These are
tuning parameters that should be adjusted during calibration.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# Household-level reward
# ===================================================================

def compute_house_reward(
    consumption: float,
    production: float,
    price: float,
    battery_level: float,
    max_battery: float,
) -> float:
    """
    Compute per-household reward.

    Incentivizes:
    * Production above consumption (sell surplus).
    * Battery usage near equilibrium (neither empty nor full).
    * Low consumption (especially when price is high).

    Args:
        consumption: Energy consumed this step (kWh or abstract units).
        production: Energy produced this step.
        price: Current market price (units unspecified but typically $/kWh).
        battery_level: Current battery charge (0 to max_battery).
        max_battery: Battery capacity.

    Returns:
        Scalar reward (unbounded; typically in [-10, +10] range).
    """
    surplus = production - consumption
    surplus_reward = float(surplus * price)

    if max_battery > 0:
        battery_deviation = abs(battery_level - max_battery / 2.0)
        battery_reward = float((max_battery / 2.0 - battery_deviation) * 0.1)
    else:
        battery_reward = 0.0

    consumption_penalty = float(consumption * price * 0.5)

    reward = surplus_reward + battery_reward - consumption_penalty

    return reward


# ===================================================================
# Market-level reward
# ===================================================================

def compute_market_reward(
    supply: float,
    demand: float,
    price: float,
) -> float:
    """
    Compute market-level reward based on balance and price stability.

    Incentivizes supply/demand equilibrium and price in reasonable range.

    Args:
        supply: Total grid supply (aggregate production).
        demand: Total grid demand (aggregate consumption).
        price: Current market price (assumed in range [0, 10]).

    Returns:
        Scalar in range [0, 1] where 1.0 = perfect balance and low price.
    """
    if supply < 0 or demand < 0 or price < 0:
        logger.warning(
            "compute_market_reward received negative values: "
            "supply=%.2f, demand=%.2f, price=%.2f",
            supply, demand, price,
        )

    max_flow = max(supply, demand)
    if max_flow == 0:
        balance_reward = 1.0
    else:
        imbalance = abs(supply - demand)
        balance_reward = 1.0 - (imbalance / max_flow)

    price_reward = 1.0 - (price / 10.0)
    price_reward = max(0.0, price_reward)

    reward = float(balance_reward * price_reward)
    return reward


# ===================================================================
# Grid-level reward (placeholder)
# ===================================================================

def compute_grid_reward(
    supply: float,
    demand: float,
    price: float,
) -> float:
    """
    Compute grid-level reward.

    NOTE: This is a PLACEHOLDER that delegates to market reward.
    """
    return compute_market_reward(supply, demand, price)


# ===================================================================
# Total reward aggregation
# ===================================================================

def compute_total_reward(
    house_reward: float,
    market_reward: float,
    grid_reward: float,
) -> float:
    """
    Aggregate component rewards into a scalar objective.

    Weights are equally distributed across the three components
    (40% household, 40% market, 20% grid).  These weights are
    tuning parameters and can be adjusted based on experimental
    results.
    """
    weight_house = 0.4
    weight_market = 0.4
    weight_grid = 0.2

    total = (
        house_reward * weight_house
        + market_reward * weight_market
        + grid_reward * weight_grid
    )

    return float(total)


# ===================================================================
# Utility validation (optional for callers)
# ===================================================================

def validate_reward(reward: float, name: str = "reward") -> bool:
    """
    Check if a reward value is finite and reasonable.

    Args:
        reward: The reward value.
        name: Name for logging.

    Returns:
        ``True`` if finite, ``False`` if NaN/Inf.
    """
    if not np.isfinite(reward):
        logger.warning("%s is not finite: %s", name, reward)
        return False
    return True
