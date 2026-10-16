"""Registration of built-in ranked and direct filters."""

# Load classes whose own decorators register the direct-filter implementations.
from importlib import import_module

from pep_compass.optimization.components.filters.configuration import (
    LamsConfig,
    MoveConfig,
    NucleusPoolConfig,
    RandomPoolConfig,
    RankedConfig,
    TandemConfig,
)
from pep_compass.optimization.components.filters.direct.controls.random_lebo import (
    RandomLeBoFilter,
)
from pep_compass.optimization.components.filters.ranked.base import RankedFilter
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.lams import (
    LamsScore,
)
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.move import (
    MoveScore,
)
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.tandem import (
    TandemScore,
)
from pep_compass.optimization.components.filters.ranked.scoring.model_scores.decoder_likelihood import (
    DecoderLikelihoodScore,
)
from pep_compass.optimization.components.filters.ranked.scoring.model_scores.esm import (
    ESM2PPLScorer,
)
from pep_compass.optimization.components.filters.ranked.selection.nucleus import (
    NucleusSelection,
)
from pep_compass.optimization.components.filters.ranked.selection.threshold import (
    ThresholdSelection,
)
from pep_compass.optimization.components.filters.ranked.selection.top_k import (
    TopKSelection,
)
from pep_compass.registry.components.families.filters import FilterManager
from pep_compass.registry.strategies.ranked_filters import (
    scoring_registry,
    selection_registry,
)

for module in (
    "constraints.levenshtein",
    "constraints.sequence_length",
    "optimization.robot",
    "optimization.trust_region",
    "structural.candidate_subset",
    "structural.deduplicate",
):
    import_module("pep_compass.optimization.components.filters.direct." + module)


@scoring_registry.register(
    "tandem",
    services={"autoencoder"},
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
)
def build_tandem_score(
    autoencoder, *, horizontal_threshold=0.1, maximum_candidates=30000, alphabet=None
) -> TandemScore:
    """Build TANDEM latent-geometry scoring."""
    return TandemScore(autoencoder, horizontal_threshold, maximum_candidates, alphabet)


@scoring_registry.register(
    "lams",
    services={"autoencoder"},
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
)
def build_lams_score(
    autoencoder, *, horizontal_threshold=0.1, maximum_candidates=30000, alphabet=None
) -> LamsScore:
    """Build LAMS latent-geometry scoring."""
    return LamsScore(autoencoder, horizontal_threshold, maximum_candidates, alphabet)


@scoring_registry.register(
    "move",
    services={"autoencoder"},
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
)
def build_move_score(
    autoencoder, *, maximum_candidates=30000, alphabet=None
) -> MoveScore:
    """Build MOVE latent-geometry scoring."""
    return MoveScore(autoencoder, maximum_candidates, alphabet)


@scoring_registry.register(
    "decoder_likelihood",
    services={"autoencoder"},
    input_fields=("mutation.parent_sequence", "mutation.options"),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
)
def build_decoder_likelihood_score(
    autoencoder, *, maximum_candidates=30000, alphabet=None
) -> DecoderLikelihoodScore:
    """Build decoder-likelihood scoring."""
    return DecoderLikelihoodScore(autoencoder, maximum_candidates, alphabet)


@scoring_registry.register("esm")
def build_esm_score(*, model_name="esm2_t6_8M_UR50D", device="cpu") -> ESM2PPLScorer:
    """Build ESM pseudo-log-likelihood scoring."""
    return ESM2PPLScorer(model_name=model_name, device=device)


@selection_registry.register("threshold")
def build_threshold_selection(*, threshold: float) -> ThresholdSelection:
    """Build threshold score selection."""
    return ThresholdSelection(threshold)


@selection_registry.register("nucleus")
def build_nucleus_selection(*, top_p=0.9, temperature=1.0) -> NucleusSelection:
    """Build nucleus score selection."""
    return NucleusSelection(top_p, temperature)


@selection_registry.register("top_k")
def build_top_k_selection(*, k: int) -> TopKSelection:
    """Build top-k score selection."""
    return TopKSelection(k)


def ranked_field_contract(parameters, attribute):
    """Delegate field requirements to the selected score implementation."""
    declaration = parameters["scoring"]
    definition = scoring_registry.definition(declaration["method"])
    contract = getattr(definition, attribute)
    values = scoring_registry.configure(
        declaration["method"], declaration.get("parameters", {})
    )
    return contract(values) if callable(contract) else contract


def validate_ranked(parameters):
    """Validate the registered score and selection declarations."""
    from pep_compass.utils.validation import validate_declaration

    for name, registry in (
        ("scoring", scoring_registry),
        ("selection", selection_registry),
    ):
        method, values = validate_declaration(parameters[name], f"ranked.{name}")
        registry.validate(method, values)


def canonical_ranked(parameters):
    """Resolve score and selection defaults into the persisted declaration."""
    result = {}
    for name, registry in (
        ("scoring", scoring_registry),
        ("selection", selection_registry),
    ):
        declaration = parameters[name]
        result[name] = {
            "method": declaration["method"],
            "parameters": registry.configure(
                declaration["method"], declaration.get("parameters", {})
            ),
        }
    return result


@FilterManager.register(
    "ranked",
    config_type=RankedConfig,
    normalizer=canonical_ranked,
    input_fields=lambda values: ranked_field_contract(values, "input_fields"),
    optional_inputs=lambda values: ranked_field_contract(values, "optional_inputs"),
    nested=lambda: {"scoring": scoring_registry, "selection": selection_registry},
    validator=validate_ranked,
    display_parameters=("scoring", "selection"),
)
def build_ranked(autoencoder, scoring, selection):
    """Build an explicit score-and-selection filter composition."""
    scorer = scoring_registry.build(
        scoring["method"],
        scoring.get("parameters", {}),
        services={"autoencoder": autoencoder},
    )
    rule = selection_registry.build(
        selection["method"],
        selection.get("parameters", {}),
    )
    return RankedFilter(scorer, rule)


@FilterManager.register("esm_plausibility")
def build_esm_plausibility(
    *, threshold=None, top_k=None, model_name="esm2_t6_8M_UR50D", device="cpu"
):
    """Build ESM scoring with exactly one threshold or top-k rule."""
    if (threshold is None) == (top_k is None):
        raise ValueError("ESM filter requires exactly one of threshold or top_k.")
    selection = (
        ThresholdSelection(threshold) if threshold is not None else TopKSelection(top_k)
    )
    return RankedFilter(ESM2PPLScorer(model_name=model_name, device=device), selection)


@FilterManager.register(
    "lpbebo",
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
    config_type=NucleusPoolConfig,
    input_fields=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={"top_p", "temperature", "maximum_candidates", "alphabet"},
)
def build_lpbebo(autoencoder, **parameters):
    """Build the LPBEBO mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.9)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        DecoderLikelihoodScore(autoencoder, **parameters),
        NucleusSelection(top_p, temperature),
    )


@FilterManager.register(
    "lams",
    config_type=LamsConfig,
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={
        "similarity_threshold",
        "horizontal_threshold",
        "maximum_candidates",
        "alphabet",
    },
)
def build_lams(autoencoder, **parameters):
    """Build the LAMS mutation-choice filter."""
    threshold = parameters.pop("similarity_threshold", 0.15)
    return RankedFilter(
        LamsScore(autoencoder, **parameters), ThresholdSelection(threshold)
    )


@FilterManager.register(
    "tandem",
    config_type=TandemConfig,
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={
        "top_p",
        "temperature",
        "horizontal_threshold",
        "maximum_candidates",
        "alphabet",
    },
)
def build_tandem(autoencoder, **parameters):
    """Build the TANDEM mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.9)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        TandemScore(autoencoder, **parameters), NucleusSelection(top_p, temperature)
    )


@FilterManager.register(
    "move",
    config_type=MoveConfig,
    input_fields=(
        "mutation.parent_sequence",
        "mutation.options",
        "walker.tangent_space",
    ),
    optional_inputs=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={"top_p", "temperature", "maximum_candidates", "alphabet"},
)
def build_move(autoencoder, **parameters):
    """Build the MOVE mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.6)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        MoveScore(autoencoder, **parameters), NucleusSelection(top_p, temperature)
    )


@FilterManager.register(
    "random_walker",
    config_type=RandomPoolConfig,
    input_fields=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={
        "selection_fraction",
        "temperature",
        "maximum_positions",
        "residues_per_position",
        "maximum_candidates",
        "alphabet",
    },
)
def build_random_walker(**parameters):
    """Build random walker-mode mutation selection."""
    return RandomLeBoFilter(mode="walker", **parameters)


@FilterManager.register(
    "random_mutang",
    config_type=RandomPoolConfig,
    input_fields=("mutation.parent_sequence", "mutation.options"),
    accepted_parameters={
        "selection_fraction",
        "temperature",
        "maximum_positions",
        "residues_per_position",
        "maximum_candidates",
        "alphabet",
    },
)
def build_random_mutang(**parameters):
    """Build random MUTANG-mode mutation selection."""
    return RandomLeBoFilter(mode="mutang_random", **parameters)
