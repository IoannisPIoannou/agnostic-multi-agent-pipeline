"""
utils/config_loader.py — Singleton YAML config loader.

All modules call get_config() to retrieve the loaded configuration dict.
The YAML is parsed exactly once; subsequent calls return the cached object.

Usage:
    from utils.config_loader import get_config
    cfg = get_config()
    model_name = cfg["models"]["orchestrator"]["model"]
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_config: dict | None = None
_DEFAULT_PATH = Path(__file__).parent.parent / "config.yaml"


def get_config(path: str | Path | None = None) -> dict:
    """Return the loaded config dict, parsing YAML only on first call."""
    global _config
    if _config is None:
        config_path = Path(path) if path else _DEFAULT_PATH
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Make sure config.yaml exists in the project root."
            )
        with open(config_path, encoding="utf-8") as fh:
            _config = yaml.safe_load(fh)
    return _config


def reload_config(path: str | Path | None = None) -> dict:
    """Force a re-parse of the config file (useful in tests)."""
    global _config
    _config = None
    return get_config(path)
