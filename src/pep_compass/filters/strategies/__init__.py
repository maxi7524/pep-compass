"""Built-in filters grouped by their decision semantics."""

from pep_compass.filters.strategies.selectors.deduplicate import DeduplicateFilter
from pep_compass.filters.strategies.selectors.candidate_subset import (
    CandidateSubsetSelector,
)
from pep_compass.filters.strategies.selectors.robot import RobotSelector
from pep_compass.filters.strategies.decision_models.esm_filter import ESMDecisionFilter
from pep_compass.filters.strategies.selectors.trust_region import (
    TrustRegionSelector,
    TrustRegionUpdater,
)
from pep_compass.filters.manager import FilterManager
from pep_compass.filters.strategies.mutation_choice import MutationChoiceFilter
from pep_compass.filters.strategies.mutation_filters import (
    LamsFilter,
    LpbeboFilter,
    MoveFilter,
    RandomLeBoFilter,
    TandemFilter,
)
from pep_compass.utils.strategy_factory import parameter_contract


@FilterManager.register("lpbebo")
@parameter_contract(source=LpbeboFilter)
def build_lpbebo(encoder_decoder, **parameters):
    """Build the LPBEBO mutation-choice filter."""
    return MutationChoiceFilter(LpbeboFilter(encoder_decoder, **parameters))


@FilterManager.register("lams")
@parameter_contract(
    accepted={"horizontal_threshold", "maximum_candidates", "alphabet", "similarity_threshold"}
)
def build_lams(encoder_decoder, **parameters):
    """Build the LAMS mutation-choice filter."""
    return MutationChoiceFilter(LamsFilter(encoder_decoder, **parameters))


@FilterManager.register("tandem")
@parameter_contract(source=TandemFilter)
def build_tandem(encoder_decoder, **parameters):
    """Build the TANDEM mutation-choice filter."""
    return MutationChoiceFilter(TandemFilter(encoder_decoder, **parameters))


@FilterManager.register("move")
@parameter_contract(source=MoveFilter)
def build_move(encoder_decoder, **parameters):
    """Build the MOVE mutation-choice filter."""
    return MutationChoiceFilter(MoveFilter(encoder_decoder, **parameters))


@FilterManager.register("random_walker")
@parameter_contract(source=RandomLeBoFilter)
def build_random_walker(**parameters):
    """Build random walker-mode mutation selection."""
    return MutationChoiceFilter(RandomLeBoFilter(mode="walker", **parameters))


@FilterManager.register("random_mutang")
@parameter_contract(source=RandomLeBoFilter)
def build_random_mutang(**parameters):
    """Build random MUTANG-mode mutation selection."""
    return MutationChoiceFilter(RandomLeBoFilter(mode="mutang_random", **parameters))

__all__ = [
    "CandidateSubsetSelector",
    "DeduplicateFilter",
    "ESMDecisionFilter",
    "RobotSelector",
    "TrustRegionSelector",
    "TrustRegionUpdater",
]
