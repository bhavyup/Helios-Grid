from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dict with recursive attribute-style access."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _as_confignode(value: Any) -> Any:
    if isinstance(value, dict):
        node = ConfigNode()
        for key, nested in value.items():
            node[key] = _as_confignode(nested)
        return node
    if isinstance(value, list):
        return [_as_confignode(item) for item in value]
    return value


def _apply_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    env = raw.setdefault("env", {})
    raw.setdefault("training", {})

    # Canonical: env.max_episode_steps (per-episode cap).
    # Deprecated alias: top-level training_steps.
    if "max_episode_steps" not in env and "training_steps" in raw:
        env["max_episode_steps"] = raw["training_steps"]
    if "max_episode_steps" in env:
        raw["training_steps"] = env["max_episode_steps"]
        raw["max_episode_steps"] = env["max_episode_steps"]

    # Backward-compatible top-level aliases used by legacy modules.
    if "num_households" in env:
        raw["num_households"] = env["num_households"]
    if "max_battery" in env:
        raw["max_battery"] = env["max_battery"]
    if "logging" in raw and isinstance(raw["logging"], dict):
        if "log_dir" in raw["logging"]:
            raw["LOG_DIR"] = raw["logging"]["log_dir"]
    if "agents" in raw and isinstance(raw["agents"], dict):
        coordinator = raw["agents"].get("coordinator_agent")
        if isinstance(coordinator, dict) and "learning_rate" in coordinator:
            raw["GNN_LR"] = coordinator["learning_rate"]

    return raw


def load_config() -> ConfigNode:
    config_dir = Path(__file__).resolve().parent
    global_cfg = _load_yaml(config_dir / "config.yml")
    agent_cfg = _load_yaml(config_dir / "agent_config.yml")
    merged = _deep_merge(global_cfg, agent_cfg)
    normalized = _apply_aliases(merged)
    return _as_confignode(normalized)


config = load_config()
