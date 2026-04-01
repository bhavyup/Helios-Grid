"""
Data loading and preprocessing utilities for Helios-Grid.

RETURN-TYPE CONTRACT
====================
All ``load_*`` functions return objects that support:

* ``len(data)``           → number of timesteps / rows
* ``data[int_index]``     → single row as a dict-like object

Consumers (``grid_env.py``, ``market_env.py``) rely on integer
row-indexing.  Internally the loaders use ``pd.DataFrame`` but
expose rows via a thin wrapper so that ``data[i]`` returns
``DataFrame.iloc[i]`` (a ``pd.Series``), which supports
dict-style key access: ``row["column_name"]``.

MISSING-FILE BEHAVIOR
=====================
When a data file is missing, loaders return a minimal synthetic
dataset and emit a ``logging.warning``.  This keeps the simulation
runnable in test/demo mode without silently producing empty data
that crashes downstream.
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


# ===================================================================
# Row-indexable DataFrame wrapper
# ===================================================================

class RowIndexableDataFrame:
    """
    Thin wrapper around ``pd.DataFrame`` that makes integer bracket
    indexing return **rows** (via ``.iloc``) instead of columns.

    This exists because multiple consumers do::

        weather_datum = self.weather_data[self.current_time]

    which on a bare ``DataFrame`` performs column selection, not row
    selection.

    All other ``DataFrame`` functionality is available via the
    ``.df`` attribute.
    """

    __slots__ = ("df",)

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, index: int) -> pd.Series:
        """Return row at positional index as a pd.Series (dict-like)."""
        return self.df.iloc[index]

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"RowIndexableDataFrame({len(self.df)} rows)"


# ===================================================================
# Grid topology
# ===================================================================

def load_grid_topology(
    file_path: str = "data/grid_topology/sample_grid.json",
) -> Dict[str, Any]:
    """
    Load grid topology from a JSON file.

    Args:
        file_path: Path to JSON topology file.

    Returns:
        Parsed dict.  Structure depends on the JSON schema.
        Returns empty dict if file is missing or invalid.

    ASSUMPTION: the JSON structure is compatible with
    ``pd.read_json(orient="index")``.  If the topology is a
    node/edge list, this parser will need to be replaced.
    """
    if not os.path.exists(file_path):
        logger.warning(
            "Grid topology file not found: %s — returning empty dict.",
            file_path,
        )
        return {}

    try:
        # Read as DataFrame first to validate structure
        df = pd.read_json(file_path, orient="index")
        result = df.to_dict()

        # Validate that result is a dict mapping node IDs to properties
        if not isinstance(result, dict):
            logger.error(
                "Grid topology file %s did not produce a dict structure; got %s",
                file_path, type(result)
            )
            raise ValueError(
                f"Grid topology file {file_path} has unexpected structure: "
                f"expected dict, got {type(result)}"
            )

        # Validate that values are dict-like (have node properties)
        for node_id, props in result.items():
            if not isinstance(props, dict):
                logger.error(
                    "Grid topology file %s has invalid node %s with non-dict properties: %s",
                    file_path, node_id, type(props)
                )
                raise ValueError(
                    f"Grid topology file {file_path} has invalid structure: "
                    f"node {node_id} properties are {type(props)}, expected dict"
                )

        return result

    except Exception as e:
        logger.error(
            "Failed to load grid topology from %s: %s",
            file_path, e
        )
        # Return empty dict to allow simulation to continue with defaults
        return {}


# ===================================================================
# Weather data
# ===================================================================

_DEFAULT_WEATHER_COLUMNS = [
    "temperature",
    "solar_irradiance",
    "wind_speed",
    "humidity",
]


def load_weather_data(
    file_path: str = "data/weather_data/sample_weather.csv",
) -> RowIndexableDataFrame:
    """
    Load weather time-series data.

    Args:
        file_path: Path to CSV weather file.

    Returns:
        ``RowIndexableDataFrame`` supporting ``data[int]`` → row.
        If file is missing, returns 24-row synthetic data so the
        simulation can run in demo/test mode.

    Consumers:
        ``grid_env.py``: ``self.weather_data[self.current_time]``
    """
    if not os.path.exists(file_path):
        logger.warning(
            "Weather data file not found: %s — using synthetic defaults.",
            file_path,
        )
        n_rows = 24
        df = pd.DataFrame({
            col: np.zeros(n_rows, dtype=np.float32)
            for col in _DEFAULT_WEATHER_COLUMNS
        })
        return RowIndexableDataFrame(df)

    df = pd.read_csv(file_path)
    return RowIndexableDataFrame(df)


# ===================================================================
# Household consumption data
# ===================================================================

def load_household_data(
    file_path: str = "data/historical_energy_data/sample_consumption.csv",
) -> RowIndexableDataFrame:
    """
    Load historical household consumption data.

    Args:
        file_path: Path to CSV consumption file.

    Returns:
        ``RowIndexableDataFrame`` with at least a ``"consumption"``
        column.  If file is missing, returns 24-row zero-consumption
        synthetic data.
    """
    if not os.path.exists(file_path):
        logger.warning(
            "Household data file not found: %s — using synthetic defaults.",
            file_path,
        )
        df = pd.DataFrame({
            "consumption": np.zeros(24, dtype=np.float32),
        })
        return RowIndexableDataFrame(df)

    df = pd.read_csv(file_path)
    return RowIndexableDataFrame(df)


# ===================================================================
# Market price data
# ===================================================================

_DEFAULT_MARKET_COLUMNS = ["supply", "demand", "price"]


def load_market_data(
    file_path: str = "data/market_prices/sample_prices.csv",
) -> RowIndexableDataFrame:
    """
    Load market price time-series data.

    Args:
        file_path: Path to CSV market file.

    Returns:
        ``RowIndexableDataFrame`` supporting ``data[int]`` → row
        with keys ``"supply"``, ``"demand"``, ``"price"``.
        If file is missing, returns 24-row synthetic data.

    Consumers:
        ``market_env.py``: ``self.market_data[idx]`` then
        ``row["supply"]``, ``row["demand"]``, ``row["price"]``.
    """
    if not os.path.exists(file_path):
        logger.warning(
            "Market data file not found: %s — using synthetic defaults.",
            file_path,
        )
        n_rows = 24
        df = pd.DataFrame({
            "supply": np.full(n_rows, 75.0, dtype=np.float32),
            "demand": np.full(n_rows, 60.0, dtype=np.float32),
            "price": np.full(n_rows, 0.3, dtype=np.float32),
        })
        return RowIndexableDataFrame(df)

    df = pd.read_csv(file_path)
    return RowIndexableDataFrame(df)


# ===================================================================
# Preprocessing stubs
# ===================================================================

def preprocess_weather_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess weather data.

    NOTE: currently a no-op pass-through.  Add normalization,
    interpolation, or feature engineering here when needed.
    """
    return weather_data


def preprocess_household_data(household_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess household consumption data.

    NOTE: currently a no-op pass-through.
    """
    return household_data


def preprocess_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess market price data.

    NOTE: currently a no-op pass-through.
    """
    return market_data


# ===================================================================
# Path registry
# ===================================================================

def get_data_paths() -> Dict[str, str]:
    """
    Return canonical data file paths.

    Reads from ``config`` if available, otherwise falls back to
    hardcoded defaults.

    ASSUMPTION: ``config`` supports dict-style access
    ``config['data_paths']['key']`` or attribute-style
    ``config.data_paths.key``.  Access pattern is guarded
    with a try/except to tolerate either.
    """
    defaults = {
        "grid_topology": "data/grid_topology/sample_grid.json",
        "weather_data": "data/weather_data/sample_weather.csv",
        "household_data": "data/historical_energy_data/sample_consumption.csv",
        "market_data": "data/market_prices/sample_prices.csv",
    }

    try:
        # Try dict-style first (consistent with grid_env.py)
        paths = config["data_paths"]
        return {
            k: paths.get(k, defaults[k])
            for k in defaults
        }
    except (TypeError, KeyError, AttributeError):
        pass

    try:
        # Try attribute-style (consistent with coordinator_agent.py)
        paths = config.data_paths
        return {
            k: getattr(paths, k, defaults[k])
            for k in defaults
        }
    except AttributeError:
        pass

    logger.warning(
        "Could not read data_paths from config — using hardcoded defaults."
    )
    return defaults