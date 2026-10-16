"""Canonical resolution of flat aliases and nested MUTANG stages."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from pep_compass.utils.validation import (
    validate_declaration,
    validate_integer,
    validate_keys,
)


@dataclass(frozen=True, slots=True)
class MutangConfig:
    """Canonical mutation-generation parameters and nested stage selections."""

    maximum_candidates: int | None = field(
        default=None,
        metadata={
            "description": "Per-parent combination cap; None leaves the local product unbounded."
        },
    )
    strategies: dict = field(default_factory=dict)
    max_len: int = 25
    direction_significance_threshold: float = 1e-3
    min_number_of_directions: int = 5
    token_threshold: float = 0.1
    alphabet: list[str] | None = None


def canonical_parameters(parameters):
    """Resolve nested choices and publish the effective combination cap."""
    values = dict(parameters)
    if values.get("strategies") is None:
        values["strategies"] = {}
    registries = stage_registries()
    values["strategies"] = {
        name: {
            "method": declaration["method"],
            "parameters": registries[name].configure(
                declaration["method"], declaration.get("parameters", {})
            ),
        }
        for name, declaration in resolve_stages(values).items()
    }
    if values["strategies"]["combination"]["method"] == "baseline":
        values["maximum_candidates"] = values["strategies"]["combination"][
            "parameters"
        ]["maximum_candidates"]
    return values


def resolve_stages(parameters: Mapping) -> dict:
    """Resolve stage defaults once for validation, estimation and construction.

    :param parameters: Public MUTANG parameters.
    :return: Complete nested stage declarations.
    :raises ValueError: If stages are unknown or generation caps conflict.
    """
    alphabet = parameters.get("alphabet") or list(" ACDEFGHIKLMNPQRSTVWY")
    maximum = parameters.get("maximum_candidates")
    validate_integer(maximum, "mutang.maximum_candidates", 1, optional=True)
    stages = parameters.get("strategies", {})
    if stages is None:
        stages = {}
    if not isinstance(stages, Mapping):
        raise ValueError("mutang.strategies must be a mapping.")
    defaults = {
        "geometry": {"method": "shared", "parameters": {}},
        "direction_selection": {
            "method": "baseline",
            "parameters": {
                "threshold": parameters.get("direction_significance_threshold", 1e-3),
                "minimum": parameters.get("min_number_of_directions", 5),
            },
        },
        "scoring": {
            "method": "max_absolute_loading",
            "parameters": {
                "max_len": parameters.get("max_len", 25),
                "alphabet_size": len(alphabet),
            },
        },
        "mutation_selection": {
            "method": "threshold",
            "parameters": {"threshold": parameters.get("token_threshold", 0.1)},
        },
        "combination": {
            "method": "baseline",
            "parameters": {"alphabet": alphabet, "maximum_candidates": maximum},
        },
    }
    validate_keys(stages, set(defaults), "mutang.strategies")
    resolved = {**defaults, **stages}
    for name, declaration in resolved.items():
        validate_declaration(declaration, f"mutang.strategies.{name}")
    combination = resolved["combination"]
    if combination["method"] == "baseline":
        nested = dict(combination.get("parameters", {}))
        nested_maximum = nested.get("maximum_candidates", maximum)
        validate_integer(
            nested_maximum, "mutang.combination.maximum_candidates", 1, optional=True
        )
        if maximum is not None and nested_maximum != maximum:
            raise ValueError(
                "MUTANG maximum_candidates conflicts with the combination cap."
            )
        nested.setdefault("maximum_candidates", maximum)
        nested.setdefault("alphabet", alphabet)
        resolved["combination"] = {"method": "baseline", "parameters": nested}
    return resolved


def candidate_limit(parameters: Mapping) -> int | None:
    """Return the effective bound of the builtin combination strategy.

    :param parameters: Public MUTANG parameters.
    :return: Validated cap, or null for an unknown/unbounded strategy.
    """
    combination = resolve_stages(parameters)["combination"]
    return (
        combination["parameters"].get("maximum_candidates")
        if combination["method"] == "baseline"
        else None
    )


def validate_parameters(parameters):
    """Validate nested stages without constructing the algorithm."""
    from pep_compass.registry.strategies import mutang as registry

    parameters = dict(parameters)
    if parameters.get("strategies") is None:
        parameters["strategies"] = {}
    registries = {
        "geometry": "geometry_registry",
        "direction_selection": "direction_selection_registry",
        "scoring": "scoring_registry",
        "mutation_selection": "mutation_selection_registry",
        "combination": "combination_registry",
    }
    for stage, declaration in resolve_stages(parameters).items():
        method, values = validate_declaration(declaration, stage)
        getattr(registry, registries[stage]).validate(method, values)


def stage_registries():
    """Describe every independently registered MUTANG stage without execution."""
    from pep_compass.registry.strategies import mutang as registry

    return {
        name: getattr(registry, name + "_registry")
        for name in (
            "geometry",
            "direction_selection",
            "scoring",
            "mutation_selection",
            "combination",
        )
    }
