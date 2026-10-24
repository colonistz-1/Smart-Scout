"""Small shared helpers."""

from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    """Read a YAML config file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'")
    with open(config_path) as f:
        return yaml.safe_load(f)
