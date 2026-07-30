"""Load JSON or YAML configurations for the composable optimizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_configuration(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file.

    :param path: Configuration path ending in JSON, YAML, or YML.
    :type path: str | pathlib.Path
    :return: Parsed configuration mapping.
    :rtype: dict[str, Any]
    :raises ValueError: If the extension or document root is invalid.
    """
    resolved = Path(path)
    with resolved.open(encoding="utf-8") as stream:
        if resolved.suffix.lower() == ".json":
            value = json.load(stream)
        elif resolved.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(stream)
        else:
            raise ValueError("Configuration must use JSON, YAML, or YML format.")
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping.")
    return value
