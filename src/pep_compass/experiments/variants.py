"""Deterministic optimization-configuration grid materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class ExperimentVariant:
    """One configuration produced by an optimization parameter grid."""

    index: int
    values: Mapping[str, Any]
    config: Mapping[str, Any]

    @property
    def variant_id(self) -> str:
        """Return a stable filesystem-safe variant identifier."""
        return f"variant_{self.index:05d}"


def materialize_variants(config: Mapping[str, Any]) -> list[ExperimentVariant]:
    """Create the Cartesian product configured in ``experiment.grid``.

    Grid keys are dotted paths rooted at ``optimization``. Restricting the
    grid to optimization values allows every variant to reuse one immutable
    encoder-decoder instance safely.

    :param config: Complete experiment configuration.
    :type config: Mapping[str, Any]
    :return: Variants in deterministic mapping/product order.
    :rtype: list[ExperimentVariant]
    :raises ValueError: If a grid path or value list is invalid.
    """
    experiment = config.get("experiment", {})
    grid = experiment.get("grid")
    if grid is None:
        return [ExperimentVariant(0, {}, deepcopy(config))]
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("experiment.grid must be a non-empty mapping.")
    paths = list(grid)
    choices: list[list[Any]] = []
    for path in paths:
        if not isinstance(path, str) or not path.startswith("optimization."):
            raise ValueError("Grid paths must start with optimization.")
        values = grid[path]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"Grid values for {path} must be a non-empty sequence.")
        value_list = list(values)
        if not value_list:
            raise ValueError(f"Grid values for {path} must be a non-empty sequence.")
        choices.append(value_list)
    variants = []
    for index, combination in enumerate(product(*choices)):
        variant_config = deepcopy(config)
        selected = dict(zip(paths, combination, strict=True))
        for path, value in selected.items():
            _set_path(variant_config, path, value)
        variants.append(ExperimentVariant(index, selected, variant_config))
    return variants


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    """Set an existing dotted configuration path."""
    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        if isinstance(target, dict) and part in target:
            target = target[part]
        elif isinstance(target, list) and part.isdigit() and int(part) < len(target):
            target = target[int(part)]
        else:
            raise ValueError(f"Grid path does not exist: {path}")
    leaf = parts[-1]
    if isinstance(target, dict) and leaf in target:
        target[leaf] = value
    elif isinstance(target, list) and leaf.isdigit() and int(leaf) < len(target):
        target[int(leaf)] = value
    else:
        raise ValueError(f"Grid path does not exist: {path}")
