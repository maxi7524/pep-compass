"""Built-in mutation generator strategies."""

from pep_compass.optimization.components.mutation_generators.strategies.mutang.combination.baseline import (
    BaselineCombination,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.direction_selection.baseline import (
    BaselineDirectionSelection,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.generator import (
    MutangGenerator,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.kappa_stable import (
    KappaStableGeometry,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.shared import (
    SharedGeometry,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.mutang import (
    Mutang,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.mutation_selection.threshold import (
    ThresholdSelection,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.scoring.max_absolute_loading import (
    MaxAbsoluteLoading,
)
from pep_compass.registry.components.families.mutation_generators import (
    MutationGeneratorManager,
)
from pep_compass.registry.strategies.mutang import (
    combination_registry,
    direction_selection_registry,
    geometry_registry,
    mutation_selection_registry,
    scoring_registry,
)


@geometry_registry.register("shared")
def build_shared_geometry() -> SharedGeometry:
    """Build geometry shared with the current SORBES point."""
    return SharedGeometry()


@geometry_registry.register("kappa_stable")
def build_kappa_stable_geometry(*, kappa: float = 1e-6) -> KappaStableGeometry:
    """Build independently computed stable geometry."""
    return KappaStableGeometry(kappa=kappa)


@direction_selection_registry.register("baseline")
def build_baseline_direction_selection(
    *, threshold: float = 1e-3, minimum: int = 5
) -> BaselineDirectionSelection:
    """Build baseline MUTANG singular-direction selection."""
    return BaselineDirectionSelection(threshold, minimum)


@scoring_registry.register("max_absolute_loading")
def build_max_absolute_loading(
    *, max_len: int = 25, alphabet_size: int = 21
) -> MaxAbsoluteLoading:
    """Build baseline per-token directional loading scorer."""
    return MaxAbsoluteLoading(max_len, alphabet_size)


@mutation_selection_registry.register("threshold")
def build_threshold_mutation_selection(*, threshold: float = 0.1) -> ThresholdSelection:
    """Build threshold-based MUTANG option selection."""
    return ThresholdSelection(threshold)


@combination_registry.register("baseline")
def build_baseline_combination(
    *, alphabet: list[str], maximum_candidates: int | None = None
) -> BaselineCombination:
    """Build baseline mutation combination enumeration."""
    return BaselineCombination(alphabet, maximum_candidates)


from pep_compass.optimization.components.mutation_generators.strategies.mutang.configuration import (
    MutangConfig,
    candidate_limit,
    canonical_parameters,
    stage_registries,
    validate_parameters,
)


@MutationGeneratorManager.register(
    "mutang",
    config_type=MutangConfig,
    nested=stage_registries,
    validator=validate_parameters,
    normalizer=canonical_parameters,
    display_parameters=("maximum_candidates", "strategies"),
    input_fields=lambda values: (
        ("tangent_geometry", "point_id")
        if values["strategies"]["geometry"]["method"] == "shared"
        else ()
    ),
    output_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "mutation.scored_options",
    ),
    invalidated_prefixes=("oracle.",),
    output_bound=lambda parameters, count: (
        None
        if candidate_limit(parameters) is None
        else candidate_limit(parameters) * count
    ),
)
def build_mutang(autoencoder, **parameters):
    """Build MUTANG using the registered canonical configuration."""
    from pep_compass.optimization.components.mutation_generators.strategies.mutang.configuration import (
        resolve_stages,
    )

    configuration = resolve_stages(parameters)
    geometry_config = configuration["geometry"]
    directions_config = configuration["direction_selection"]
    scoring_config = configuration["scoring"]
    selection_config = configuration["mutation_selection"]
    combination_config = configuration["combination"]
    mutang = Mutang(
        geometry_registry.build(
            geometry_config["method"], geometry_config.get("parameters", {})
        ),
        direction_selection_registry.build(
            directions_config["method"], directions_config.get("parameters", {})
        ),
        scoring_registry.build(
            scoring_config["method"], scoring_config.get("parameters", {})
        ),
        mutation_selection_registry.build(
            selection_config["method"], selection_config.get("parameters", {})
        ),
        combination_registry.build(
            combination_config["method"], combination_config.get("parameters", {})
        ),
        autoencoder,
    )
    return MutangGenerator(mutang)
