"""Project-wide YAML config loader with dict and attribute compatibility access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_CONFIG_FILES = ("config.yml", "agent_config.yml", "market_config.yml")


class ConfigProxy(Mapping[str, Any]):
    """Read-only config view supporting both dict and attribute access."""

    def __init__(self, data: Mapping[str, Any]):
        self._data: Dict[str, Any] = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._wrap(self._data[name])
        raise AttributeError(f"Config key not found: {name}")

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self._wrap(self._data[key])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return ConfigProxy(value)
        return value


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
            continue
        base[key] = value


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at root: {path}")
    return data


def load_project_config(config_dir: Path | None = None) -> Dict[str, Any]:
    root = config_dir or _CONFIG_DIR
    merged: Dict[str, Any] = {}

    for filename in _CONFIG_FILES:
        _deep_merge(merged, _load_yaml_file(root / filename))

    return merged


def refresh_config() -> ConfigProxy:
    return ConfigProxy(load_project_config())


config = refresh_config()
