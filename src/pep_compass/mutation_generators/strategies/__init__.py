"""Built-in mutation generator strategies."""

from pep_compass.mutation_generators.strategies.mutang import MutangGenerator
from pep_compass.mutation_generators.strategies.tangent_space import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.mutation_generators.manager import MutationGeneratorManager
from pep_compass.utils.strategy_factory import parameter_contract


@MutationGeneratorManager.register("mutang")
@parameter_contract(source=MutationEnumerationInTangentSpace)
def build_mutang(**parameters):
    """Build MUTANG and its tangent-space mutation enumerator."""
    return MutangGenerator(MutationEnumerationInTangentSpace(**parameters))

__all__ = ["MutangGenerator", "MutationEnumerationInTangentSpace"]
