"""Canonical SORBES declarations shared by validation and construction."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from pep_compass.utils.validation import validate_declaration, validate_positive


@dataclass(frozen=True, slots=True)
class SorbesConfig:
    """Public aliases and independently selectable stages of one SORBES walker."""

    geometry: dict | None = field(
        default=None,
        metadata={
            "description": "Decoder geometry stage; default kappa is horizontal_threshold squared."
        },
    )
    directions: dict | None = None
    scaling: dict | None = None
    position_update: dict | None = None
    boundary: dict | None = None
    horizontal_threshold: float | None = field(
        default=None,
        metadata={
            "description": "Singular-value threshold; defaults to 0.1 when geometry is omitted."
        },
    )
    time_step: float | None = field(
        default=None,
        metadata={
            "description": "Diffusion step; defaults to 0.01 and maps to epsilon=sqrt(time_step)."
        },
    )
    max_horizontal_update_norm: float | None = None
    vertical_movement: bool = True


def canonical_parameters(parameters):
    """Store complete nested defaults rather than unresolved alias sentinels."""
    registries = stage_registries()
    return {
        name: {
            "method": declaration["method"],
            "parameters": registries[name].configure(
                declaration["method"], declaration.get("parameters", {})
            ),
        }
        for name, declaration in resolve_stages(parameters).items()
    }


def resolve_stages(parameters: Mapping) -> dict:
    """Resolve legacy aliases and complete nested SORBES defaults.

    :param parameters: Public walker parameters.
    :return: Complete declarations of the five fixed stages.
    :raises ValueError: If aliases conflict with nested stage parameters.
    """
    horizontal = parameters.get("horizontal_threshold")
    time_step = parameters.get("time_step")
    maximum_norm = parameters.get("max_horizontal_update_norm")
    vertical = parameters.get("vertical_movement", True)
    if horizontal is not None and (
        isinstance(horizontal, bool)
        or not isinstance(horizontal, (int, float))
        or not math.isfinite(horizontal)
        or horizontal < 0
    ):
        raise ValueError("SORBES horizontal_threshold must be finite and non-negative.")
    for name, value in (
        ("time_step", time_step),
        ("max_horizontal_update_norm", maximum_norm),
    ):
        if value is not None:
            validate_positive(value, f"sorbes.{name}")
    if not isinstance(vertical, bool):
        raise ValueError("SORBES vertical_movement must be a boolean.")
    defaults = {
        "geometry": {
            "method": "kappa_stable",
            "parameters": {"kappa": (0.1 if horizontal is None else horizontal) ** 2},
        },
        "directions": {
            "method": "active_inactive",
            "parameters": {"vertical_movement": vertical},
        },
        "scaling": {"method": "stable_dimension", "parameters": {}},
        "position_update": {
            "method": "main",
            "parameters": {
                "epsilon": math.sqrt(0.01 if time_step is None else time_step),
                "delta_max": 0.5 if maximum_norm is None else maximum_norm,
            },
        },
        "boundary": {"method": "main", "parameters": {}},
    }
    resolved = {
        name: parameters.get(name) if parameters.get(name) is not None else default
        for name, default in defaults.items()
    }
    for name, declaration in resolved.items():
        validate_declaration(declaration, f"sorbes.{name}")
    for value, stage, key, expected in (
        (horizontal, "geometry", "kappa", defaults["geometry"]["parameters"]["kappa"]),
        (
            time_step,
            "position_update",
            "epsilon",
            defaults["position_update"]["parameters"]["epsilon"],
        ),
        (
            maximum_norm,
            "position_update",
            "delta_max",
            defaults["position_update"]["parameters"]["delta_max"],
        ),
    ):
        nested = resolved[stage].get("parameters", {})
        if (
            value is not None
            and parameters.get(stage) is not None
            and (
                not isinstance(nested.get(key), (int, float))
                or not math.isclose(nested[key], expected, rel_tol=1e-12, abs_tol=0.0)
            )
        ):
            raise ValueError(f"SORBES flat alias conflicts with {stage}.{key}.")
    return resolved


def validate_parameters(parameters):
    """Validate nested stages without constructing the algorithm."""
    from pep_compass.registry.strategies import sorbes as registry

    registries = {
        "geometry": "geometry_registry",
        "directions": "directions_registry",
        "scaling": "scaling_registry",
        "position_update": "position_update_registry",
        "boundary": "boundary_registry",
    }
    for stage, declaration in resolve_stages(parameters).items():
        method, values = validate_declaration(declaration, stage)
        getattr(registry, registries[stage]).validate(method, values)


def stage_registries():
    """Expose selectable internal stages through public component descriptions."""
    from pep_compass.registry.strategies import sorbes as registry

    return {
        name: getattr(registry, name + "_registry")
        for name in ("geometry", "directions", "scaling", "position_update", "boundary")
    }
