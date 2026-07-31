"""Configuration validation without constructing computational models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pep_compass.experiments.variants import materialize_variants


def validate_configuration(config: Mapping[str, Any]) -> None:
    """Validate every materialized optimization tree before model loading.

    :param config: Complete composable experiment configuration.
    :type config: Mapping[str, Any]
    :raises ValueError: If encoder, grid, operation, method, or structure is invalid.
    """
    encoder = config.get("encoder_decoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("encoder_decoder configuration is required.")
    if encoder.get("method") != "hydramp":
        raise ValueError(f"Unknown encoder-decoder method: {encoder.get('method')}")
    experiment = config.get("experiment", {})
    execution = experiment.get("execution", {}) if isinstance(experiment, Mapping) else {}
    if not isinstance(execution, Mapping):
        raise ValueError("experiment.execution must be a mapping.")
    if execution.get("backend", "local") not in {"local", "subprocess", "slurm"}:
        raise ValueError("experiment.execution.backend must be local, subprocess, or slurm.")
    workers = execution.get("max_workers", 1)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("experiment.execution.max_workers must be positive.")
    _load_registries()
    for variant in materialize_variants(config):
        optimization = variant.config.get("optimization")
        if not isinstance(optimization, Mapping):
            raise ValueError("optimization configuration is required.")
        _validate_steps(optimization.get("steps"), "optimization.steps")
        limits = optimization.get("limits", {})
        if not isinstance(limits, Mapping):
            raise ValueError("optimization.limits must be a mapping.")
        for name in ("oracle_calls", "generated_candidates"):
            value = limits.get(name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"optimization.limits.{name} must be null or non-negative.")


def _load_registries() -> None:
    import pep_compass.filters.strategies  # noqa: F401
    import pep_compass.mutation_generators.strategies  # noqa: F401
    import pep_compass.oracles.strategies  # noqa: F401
    import pep_compass.walkers.strategies  # noqa: F401


def _validate_steps(value: Any, path: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence.")
    for index, configuration in enumerate(value):
        step_path = f"{path}.{index}"
        if not isinstance(configuration, Mapping) or len(configuration) != 1:
            raise ValueError(f"{step_path} must contain exactly one operation key.")
        operation, settings = next(iter(configuration.items()))
        if not isinstance(settings, Mapping):
            raise ValueError(f"{step_path}.{operation} must be a mapping.")
        if operation == "loop":
            iterations = settings.get("iterations")
            if not isinstance(iterations, int) or iterations < 0:
                raise ValueError(f"{step_path}.loop.iterations must be non-negative.")
            _validate_steps(settings.get("steps"), f"{step_path}.loop.steps")
        elif operation == "parallel":
            _validate_parallel(settings, f"{step_path}.parallel")
        elif operation in {"walker", "mutation_generator", "filter", "oracle"}:
            _validate_strategy(operation, settings, f"{step_path}.{operation}")
        else:
            raise ValueError(f"Unknown optimization operation at {step_path}: {operation}")


def _validate_parallel(settings: Mapping[str, Any], path: str) -> None:
    execution = settings.get("execution", "sequential")
    if execution not in {"sequential", "concurrent"}:
        raise ValueError(f"{path}.execution must be sequential or concurrent.")
    if settings.get("merge", "concatenate") != "concatenate":
        raise ValueError(f"{path}.merge is not implemented.")
    branches = settings.get("branches")
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
        raise ValueError(f"{path}.branches must be a sequence.")
    names: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            raise ValueError(f"{path}.branches.{index} must be a mapping.")
        name = str(branch.get("name", f"branch_{index}"))
        if name in names:
            raise ValueError(f"Duplicate parallel branch name: {name}")
        names.add(name)
        _validate_steps(branch.get("steps"), f"{path}.branches.{index}.steps")


def _validate_strategy(operation: str, settings: Mapping[str, Any], path: str) -> None:
    from pep_compass.filters.manager import FilterManager
    from pep_compass.mutation_generators.manager import MutationGeneratorManager
    from pep_compass.oracles.manager import OracleManager
    from pep_compass.walkers.manager import WalkerManager

    method = settings.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"{path}.method must be a non-empty string.")
    if not isinstance(settings.get("parameters", {}), Mapping):
        raise ValueError(f"{path}.parameters must be a mapping.")
    methods = {
        "walker": WalkerManager.methods(),
        "mutation_generator": MutationGeneratorManager.methods(),
        "filter": FilterManager.methods(),
        "oracle": OracleManager.methods(),
    }[operation]
    if method not in methods:
        available = ", ".join(methods)
        raise ValueError(f"Unknown {operation} method at {path}: {method}. Available: {available}")
