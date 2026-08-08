"""Built-in mutation generator strategies."""

from pep_compass.mutation_generators.strategies.mutang import MutangGenerator
from pep_compass.mutation_generators.strategies.tangent_space import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.mutation_generators.manager import MutationGeneratorManager
from pep_compass.utils.strategy_factory import parameter_contract


@MutationGeneratorManager.register("mutang")
@parameter_contract(
    accepted={
        "max_len",
        "direction_significance_threshold",
        "min_number_of_directions",
        "token_threshold",
        "alphabet",
        "maximum_candidates",
    }
)
def build_mutang(encoder_decoder, maximum_candidates=None, **parameters):
    """Build MUTANG and its tangent-space mutation enumerator."""
    return MutangGenerator(
        MutationEnumerationInTangentSpace(**parameters),
        maximum_candidates=maximum_candidates,
        encoder_decoder=encoder_decoder,
    )

__all__ = ["MutangGenerator", "MutationEnumerationInTangentSpace"]
