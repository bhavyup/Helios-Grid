"""
Reward computation utilities for Helios-Grid.

REWARD COMPONENTS
=================
Helios-Grid uses a multi-level reward structure:

1. **Household reward** — incentivizes local production, penalizes
   consumption, encourages balanced battery usage.
2. **Market reward** — incentivizes supply/demand balance and
   price stability.
3. **Grid reward** — (placeholder) would incentivize grid-level
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

    NOTES
    -----
    * Surplus reward is proportional to price — high-price periods
      incentivize selling.
    * Consumption penalty is also proportional to price — creates
      a price-sensitive demand response.
    * Battery reward encourages staying near 50% charge (minimizes
      stress on battery chemistry in a real system).
    """
    # Surplus production (production > consumption is positive)
    surplus = production - consumption
    # Reward for selling surplus scales with price
    surplus_reward = float(surplus * price)

    # Battery-level penalty: reward for being near max_battery / 2
    # (minimize extremes)
    if max_battery > 0:
        battery_deviation = abs(battery_level - max_battery / 2.0)
        # Scale: 0 deviation → 0.1 * max_battery reward
        # Full deviation (at empty or full) → 0.1 * max_battery / 2 reward
        battery_reward = float((max_battery / 2.0 - battery_deviation) * 0.1)
    else:
        battery_reward = 0.0

    # Consumption penalty: scales with consumption and price
    # Factor 0.5 ensures penalty is less harsh than surplus reward
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

    NOTES
    -----
    * Balance reward: 1.0 when supply == demand, decreases as imbalance
      grows. Formula: ``1 - (|supply - demand| / max(supply, demand))``.
    * Price reward: assumes price in [0, 10]. Formula: ``1 - (price / 10)``.
    * Both components are multiplied together, so either being poor
      degrades the total.
    * If both supply and demand are 0, returns 1.0 (degenerate case).
    """
    if supply < 0 or demand < 0 or price < 0:
        logger.warning(
            "compute_market_reward received negative values: "
            "supply=%.2f, demand=%.2f, price=%.2f",
            supply, demand, price,
        )

    max_flow = max(supply, demand)
    if max_flow == 0:
        # Degenerate case: zero supply and demand
        balance_reward = 1.0
    else:
        imbalance = abs(supply - demand)
        # Normalize imbalance to [0, 1]; reward decreases with imbalance
        balance_reward = 1.0 - (imbalance / max_flow)

    # Price reward: assumes price in [0, 10]
    # HARDCODED ASSUMPTION: price range is [0, 10].
    # If market prices are typically [0.1, 1.0] (from config),
    # this always returns > 0.9, making price signal weak.
    # TODO: calibrate this based on actual price distribution.
    price_reward = 1.0 - (price / 10.0)
    price_reward = max(0.0, price_reward)  # Clamp to [0, 1]

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
    A true grid reward would incorporate voltage stability,
    frequency stability, and other grid metrics that are not yet
    modeled in Helios-Grid.

    Args:
        supply: Total grid supply.
        demand: Total grid demand.
        price: Current market price.

    Returns:
        Grid reward (currently same as market reward).

    HISTORY
    -------
    The original signature was::

        def compute_grid_reward(grid_balance: float,
                               grid_stability: float) -> float

    But no module in the codebase computes ``grid_stability``,
    making that version a dead-end. This version aligns with how
    ``coordinator_agent.py`` calls it.
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

    Args:
        house_reward: Per-household reward (from ``compute_house_reward``).
        market_reward: Market-level reward (from ``compute_market_reward``).
        grid_reward: Grid-level reward (from ``compute_grid_reward``).

    Returns:
        Scalar total reward.

    WEIGHTS
    -------
    * House: 0.4 (40%) — household-level objectives
    * Market: 0.4 (40%) — market-level objectives
    * Grid: 0.2 (20%) — grid-level objectives (underweighted as placeholder)
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