"""
Logging and persistence utilities for Helios-Grid.

LOG FORMAT
==========
Domain loggers (``log_training_data``, ``log_simulation_data``,
``log_env_info``) append entries to **JSONL** (JSON Lines) files —
one JSON object per line.  This avoids the overwrite-on-every-call
bug in the original implementation and supports efficient append
without loading the full file.

Files produced:

* ``training_log.jsonl``   — one line per training step/epoch
* ``simulation_log.jsonl`` — one line per simulation step
* ``env_log.jsonl``        — one line per environment step
* ``full_state_log.jsonl`` — one line per state snapshot

CSV export is available via ``export_jsonl_to_csv``.

DETERMINISM NOTE
================
``create_log_directory`` accepts an optional ``run_id`` to produce
deterministic directory names.  Wall-clock timestamps are added to
individual log entries by default but can be suppressed by callers
providing their own ``timestamp`` field.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================
# JSON serialization safety
# ===================================================================

class _SafeEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy/torch types gracefully.

    Prevents ``TypeError: Object of type ndarray is not JSON
    serializable`` when logging environment state.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # torch tensors — guard import
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
        except ImportError:
            pass
        return super().default(obj)


# ===================================================================
# Directory management
# ===================================================================

def create_log_directory(
    base_dir: str = "logs",
    run_id: Optional[str] = None,
) -> str:
    """
    Create and return a log directory path.

    Args:
        base_dir: Parent directory for logs.
        run_id: Deterministic subdirectory name.  If *None*, a
            wall-clock timestamp is used (non-deterministic).

    Returns:
        Absolute-ish path to the created directory.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_dir, run_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _ensure_dir(log_dir: str) -> None:
    """Create ``log_dir`` if it does not exist."""
    os.makedirs(log_dir, exist_ok=True)


# ===================================================================
# Core persistence
# ===================================================================

def append_log_entry(
    log_dir: str,
    entry: Dict[str, Any],
    filename: str = "log.jsonl",
) -> None:
    """
    Append a single JSON object as one line to a JSONL file.

    This replaces the original ``save_log_entry`` which used
    ``"w"`` mode and overwrote the file on every call.
    """
    _ensure_dir(log_dir)
    file_path = os.path.join(log_dir, filename)
    with open(file_path, "a") as f:
        f.write(json.dumps(entry, cls=_SafeEncoder) + "\n")


def save_log_entry(
    log_dir: str,
    entry: Dict[str, Any],
    filename: str = "log_entry.json",
) -> None:
    """
    Write a single JSON entry to a file (overwrites).

    .. deprecated::
        Use ``append_log_entry`` for sequential logging.
        This function is retained only for one-shot snapshots
        (e.g., config dumps, final summaries).
    """
    _ensure_dir(log_dir)
    file_path = os.path.join(log_dir, filename)
    with open(file_path, "w") as f:
        json.dump(entry, f, indent=4, cls=_SafeEncoder)


# ===================================================================
# CSV utilities
# ===================================================================

def save_logs_to_csv(
    log_dir: str,
    logs: List[Dict[str, Any]],
    filename: str = "logs.csv",
) -> None:
    """Save a list of log dicts to a CSV file."""
    _ensure_dir(log_dir)
    df = pd.DataFrame(logs)
    file_path = os.path.join(log_dir, filename)
    df.to_csv(file_path, index=False)


def load_logs_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """Load logs from a CSV file into a list of dicts."""
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


def export_jsonl_to_csv(
    jsonl_path: str,
    csv_path: Optional[str] = None,
) -> str:
    """
    Convert a JSONL log file to CSV.

    Args:
        jsonl_path: Path to the ``.jsonl`` file.
        csv_path: Output CSV path.  Defaults to same name with
            ``.csv`` extension.

    Returns:
        Path to the written CSV.
    """
    if csv_path is None:
        csv_path = os.path.splitext(jsonl_path)[0] + ".csv"
    entries = load_logs_from_jsonl(jsonl_path)
    if entries:
        pd.DataFrame(entries).to_csv(csv_path, index=False)
    else:
        logger.warning("No entries in %s — CSV not written.", jsonl_path)
    return csv_path


# ===================================================================
# JSONL reading
# ===================================================================

def load_logs_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load all entries from a JSONL file."""
    if not os.path.exists(file_path):
        logger.warning("JSONL file not found: %s", file_path)
        return []
    entries: List[Dict[str, Any]] = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed JSON at %s:%d — skipped.", file_path, line_num
                )
    return entries


# ===================================================================
# Domain-specific loggers
# ===================================================================

def log_training_data(
    log_dir: str,
    episode: int,
    total_reward: float,
    avg_house_reward: float,
    avg_market_reward: float,
    avg_grid_reward: float,
    step: int,
    timestamp: Optional[str] = None,
) -> None:
    """
    Log a single training step / epoch.

    Appends to ``training_log.jsonl`` in ``log_dir``.

    Args:
        log_dir: Target directory.
        episode: Episode number.
        total_reward: Aggregate reward.
        avg_house_reward: Mean household reward.
        avg_market_reward: Mean market reward.
        avg_grid_reward: Mean grid-level reward.
        step: Step within the episode.
        timestamp: Optional explicit timestamp.  If *None*,
            ``datetime.now().isoformat()`` is used (non-deterministic).
    """
    entry = {
        "episode": episode,
        "step": step,
        "total_reward": total_reward,
        "avg_house_reward": avg_house_reward,
        "avg_market_reward": avg_market_reward,
        "avg_grid_reward": avg_grid_reward,
        "timestamp": timestamp or datetime.now().isoformat(),
    }
    append_log_entry(log_dir, entry, filename="training_log.jsonl")


def log_simulation_data(
    log_dir: str,
    timestamp: str,
    grid_balance: float,
    market_balance: float,
    household_consumption: float,
    solar_production: float,
    wind_production: float,
) -> None:
    """
    Log a single simulation step.

    Appends to ``simulation_log.jsonl`` in ``log_dir``.
    """
    entry = {
        "timestamp": timestamp,
        "grid_balance": grid_balance,
        "market_balance": market_balance,
        "household_consumption": household_consumption,
        "solar_production": solar_production,
        "wind_production": wind_production,
    }
    append_log_entry(log_dir, entry, filename="simulation_log.jsonl")


def log_env_info(
    episode: int,
    step: int,
    current_time: int,
    reward: float,
    log_dir: Optional[str] = None,
) -> None:
    """
    Log per-step environment info.

    Appends to ``env_log.jsonl`` in ``log_dir`` if provided,
    otherwise logs to the Python logger only.

    This function exists because ``grid_env.py`` imports::

        from utils.logging_utils import log_env_info

    Args:
        episode: Episode number.
        step: Step within the episode.
        current_time: Environment's internal clock.
        reward: Step reward.
        log_dir: Optional target directory.
    """
    entry = {
        "episode": episode,
        "step": step,
        "current_time": current_time,
        "reward": reward,
        "timestamp": datetime.now().isoformat(),
    }

    if log_dir is not None:
        append_log_entry(log_dir, entry, filename="env_log.jsonl")
    else:
        logger.info(
            "env step — episode=%d step=%d time=%d reward=%.4f",
            episode, step, current_time, reward,
        )


def log_full_state(
    log_dir: str,
    state: Dict[str, Any],
    timestamp: str,
) -> None:
    """
    Log a full state snapshot.

    Appends to ``full_state_log.jsonl`` in ``log_dir``.

    NOTE: ``_SafeEncoder`` handles numpy arrays and torch tensors
    so arbitrary state dicts can be serialized safely.
    """
    entry = {
        "timestamp": timestamp,
        "state": state,
    }
    append_log_entry(log_dir, entry, filename="full_state_log.jsonl")


# ===================================================================
# Log retrieval
# ===================================================================

def get_log_file_path(log_dir: str, filename: str) -> str:
    """Return the full path for a log file in the given directory."""
    return os.path.join(log_dir, filename)


def get_all_logs(log_dir: str) -> List[Dict[str, Any]]:
    """
    Load all log entries from a directory.

    Reads both ``.jsonl`` and ``.csv`` files.
    """
    if not os.path.isdir(log_dir):
        logger.warning("Log directory not found: %s", log_dir)
        return []

    logs: List[Dict[str, Any]] = []

    for filename in sorted(os.listdir(log_dir)):
        file_path = os.path.join(log_dir, filename)
        if filename.endswith(".jsonl"):
            logs.extend(load_logs_from_jsonl(file_path))
        elif filename.endswith(".csv"):
            logs.extend(load_logs_from_csv(file_path))

    return logs