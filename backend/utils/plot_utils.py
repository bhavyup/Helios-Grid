"""
Plotting utilities for Helios-Grid.

All plot functions:
* accept a ``log_dir`` destination (defaults to ``"logs"``).
* create the directory if it doesn't exist.
* validate required columns before plotting.
* close figures even on error (no memory leaks).
* convert ISO-timestamp strings to datetime objects for
  readable x-axes.

USAGE
=====
::

    from utils.logging_utils import load_logs_from_jsonl
    from utils.plot_utils import plot_rewards, plot_simulation_data

    training_logs = load_logs_from_jsonl("logs/training_log.jsonl")
    plot_rewards(training_logs, log_dir="logs/plots")

    sim_logs = load_logs_from_jsonl("logs/simulation_log.jsonl")
    plot_simulation_data(sim_logs, log_dir="logs/plots")
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Default log directory — avoids import-time config access.
_DEFAULT_LOG_DIR = "logs"


# ===================================================================
# Internal helpers
# ===================================================================

def _resolve_log_dir(log_dir: Optional[str]) -> str:
    """Return a concrete log directory, creating it if needed."""
    if log_dir is None:
        log_dir = _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _prepare_dataframe(
    logs: List[Dict[str, Any]],
    required_columns: Sequence[str],
    sort_by: str = "timestamp",
) -> Optional[pd.DataFrame]:
    """
    Convert logs to a DataFrame and validate required columns.

    Returns ``None`` (and logs a warning) if validation fails
    rather than crashing mid-plot.
    """
    if not logs:
        logger.warning("Empty log list — nothing to plot.")
        return None

    df = pd.DataFrame(logs)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.warning(
            "Cannot plot: missing columns %s. Available: %s",
            missing,
            list(df.columns),
        )
        return None

    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    return df


def _timestamp_to_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ``"timestamp"`` column to datetime for readable x-axes.

    Falls back to string if parsing fails; falls back to integer
    index if column is missing.
    """
    if "timestamp" not in df.columns:
        df = df.copy()
        df["timestamp"] = range(len(df))
        return df

    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except (ValueError, TypeError):
        pass  # keep as-is; matplotlib will handle strings

    return df


def _save_and_close(
    fig: plt.Figure,
    log_dir: str,
    filename: str,
) -> None:
    """Save figure and close, ensuring no resource leak."""
    file_path = os.path.join(log_dir, filename)
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", file_path)


# ===================================================================
# Domain-specific plots
# ===================================================================

_REWARD_COLUMNS = [
    "total_reward",
    "avg_house_reward",
    "avg_market_reward",
    "avg_grid_reward",
]

_SIMULATION_COLUMNS = [
    "grid_balance",
    "market_balance",
    "household_consumption",
    "solar_production",
    "wind_production",
]


def plot_rewards(
    logs: List[Dict[str, Any]],
    log_dir: Optional[str] = None,
    filename: str = "rewards_plot.png",
) -> None:
    """
    Plot training reward curves.

    Expects logs from ``log_training_data`` with columns:
    ``total_reward``, ``avg_house_reward``, ``avg_market_reward``,
    ``avg_grid_reward``, and optionally ``timestamp``.
    """
    log_dir = _resolve_log_dir(log_dir)
    df = _prepare_dataframe(logs, _REWARD_COLUMNS)
    if df is None:
        return

    df = _timestamp_to_axis(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        for col in _REWARD_COLUMNS:
            label = col.replace("_", " ").title()
            sns.lineplot(x="timestamp", y=col, data=df, label=label, ax=ax)

        ax.set_title("Training Rewards Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
    except Exception:
        logger.exception("Error in plot_rewards")
    finally:
        _save_and_close(fig, log_dir, filename)


def plot_simulation_data(
    logs: List[Dict[str, Any]],
    log_dir: Optional[str] = None,
    filename: str = "simulation_data_plot.png",
) -> None:
    """
    Plot simulation time-series.

    Expects logs from ``log_simulation_data`` with columns:
    ``grid_balance``, ``market_balance``, ``household_consumption``,
    ``solar_production``, ``wind_production``, and ``timestamp``.
    """
    log_dir = _resolve_log_dir(log_dir)
    df = _prepare_dataframe(logs, _SIMULATION_COLUMNS)
    if df is None:
        return

    df = _timestamp_to_axis(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        for col in _SIMULATION_COLUMNS:
            label = col.replace("_", " ").title()
            sns.lineplot(x="timestamp", y=col, data=df, label=label, ax=ax)

        ax.set_title("Simulation Data Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
    except Exception:
        logger.exception("Error in plot_simulation_data")
    finally:
        _save_and_close(fig, log_dir, filename)


def plot_state_distribution(
    logs: List[Dict[str, Any]],
    log_dir: Optional[str] = None,
    filename_suffix: str = "state_over_time.png",
    column_filter: Optional[List[str]] = None,
) -> None:
    """
    Plot time-series for selected state variables.

    Args:
        logs: Log entries.
        log_dir: Output directory.
        filename_suffix: Appended to each variable name for the
            output file.
        column_filter: Explicit list of columns to plot.  If
            *None*, plots all numeric columns except ``timestamp``.
    """
    log_dir = _resolve_log_dir(log_dir)

    if not logs:
        logger.warning("Empty log list — nothing to plot.")
        return

    df = pd.DataFrame(logs)
    df = _timestamp_to_axis(df)

    if column_filter is not None:
        state_vars = [c for c in column_filter if c in df.columns]
    else:
        state_vars = [
            c for c in df.select_dtypes(include="number").columns
            if c != "timestamp"
        ]

    if not state_vars:
        logger.warning(
            "No plottable state variables found. Columns: %s",
            list(df.columns),
        )
        return

    for var in state_vars:
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            sns.lineplot(x="timestamp", y=var, data=df, ax=ax)
            ax.set_title(f"{var.replace('_', ' ').title()} Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.grid(True)
            fig.autofmt_xdate()
        except Exception:
            logger.exception("Error plotting %s", var)
        finally:
            safe_name = var.replace(" ", "_").replace("/", "_")
            _save_and_close(fig, log_dir, f"{safe_name}_{filename_suffix}")


def plot_all(
    training_logs: Optional[List[Dict[str, Any]]] = None,
    simulation_logs: Optional[List[Dict[str, Any]]] = None,
    state_logs: Optional[List[Dict[str, Any]]] = None,
    log_dir: Optional[str] = None,
) -> None:
    """
    Plot all available log types.

    Unlike the original version, this accepts **separate** log lists
    for each schema rather than one mixed list that crashes.

    Args:
        training_logs: From ``training_log.jsonl``.
        simulation_logs: From ``simulation_log.jsonl``.
        state_logs: From ``full_state_log.jsonl`` or similar.
        log_dir: Output directory.
    """
    if training_logs:
        plot_rewards(training_logs, log_dir)

    if simulation_logs:
        plot_simulation_data(simulation_logs, log_dir)

    if state_logs:
        plot_state_distribution(state_logs, log_dir)


# ===================================================================
# Generic plots
# ===================================================================

def plot_histogram(
    data: List[float],
    log_dir: Optional[str] = None,
    filename: str = "histogram.png",
    title: str = "Histogram",
    xlabel: str = "Value",
) -> None:
    """Plot a histogram with optional KDE overlay."""
    log_dir = _resolve_log_dir(log_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        sns.histplot(data, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
    except Exception:
        logger.exception("Error in plot_histogram")
    finally:
        _save_and_close(fig, log_dir, filename)


def plot_scatter(
    x: List[float],
    y: List[float],
    log_dir: Optional[str] = None,
    filename: str = "scatter_plot.png",
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
) -> None:
    """Plot a scatter plot."""
    log_dir = _resolve_log_dir(log_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    except Exception:
        logger.exception("Error in plot_scatter")
    finally:
        _save_and_close(fig, log_dir, filename)


def save_plot_to_file(
    fig: plt.Figure,
    file_path: str,
) -> None:
    """
    Save an externally-created figure and close it.

    Creates parent directories if needed.
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", file_path)