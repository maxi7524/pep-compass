"""Assemble the fixed computational stages of a SORBES walker."""

from pep_compass.registry.strategies.sorbes import (
    boundary_registry,
    directions_registry,
    geometry_registry,
    position_update_registry,
    scaling_registry,
)


def build_stages(autoencoder, parameters):
    """Construct all stages from explicit nested declarations."""
    geometry = parameters["geometry"]
    directions = parameters["directions"]
    scaling = parameters["scaling"]
    position_update = parameters["position_update"]
    boundary = parameters["boundary"]
    return (
        geometry_registry.build(
            geometry["method"],
            geometry.get("parameters", {}),
            services={"autoencoder": autoencoder},
        ),
        directions_registry.build(
            directions["method"],
            directions.get("parameters", {}),
        ),
        scaling_registry.build(
            scaling["method"],
            scaling.get("parameters", {}),
        ),
        position_update_registry.build(
            position_update["method"],
            position_update.get("parameters", {}),
        ),
        boundary_registry.build(
            boundary["method"],
            boundary.get("parameters", {}),
        ),
    )
