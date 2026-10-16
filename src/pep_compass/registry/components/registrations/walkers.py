"""Built-in walker strategies."""

from pep_compass.optimization.components.walkers.strategies.sorbes.composition import (
    build_stages,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.configuration import (
    SorbesConfig,
    canonical_parameters,
    stage_registries,
    validate_parameters,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.sorbes import Sorbes
from pep_compass.optimization.components.walkers.strategies.sorbes.walker import (
    SorbesWalker,
)
from pep_compass.registry.components.families.walkers import WalkerManager


@WalkerManager.register(
    "sorbes",
    config_type=SorbesConfig,
    nested=stage_registries,
    validator=validate_parameters,
    normalizer=canonical_parameters,
    display_parameters=("geometry", "position_update", "directions"),
    output_fields=(
        "tangent_geometry",
        "point_id",
        "walker.tangent_space",
        "walker.adjusted_time_step",
    ),
    invalidated_prefixes=("oracle.",),
)
def build_sorbes(autoencoder, **parameters):
    """Build fixed SORBES stages from one canonical parameter declaration."""
    stages = build_stages(autoencoder, canonical_parameters(parameters))
    return SorbesWalker(Sorbes(autoencoder, *stages))
