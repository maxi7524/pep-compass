"""Built-in mutation generator strategies."""

from pep_compass.mutation_generators.strategies.mutang import MutangGenerator
from pep_compass.mutation_generators.strategies.tangent_space import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.mutation_generators.manager import MutationGeneratorManager


@MutationGeneratorManager.register("mutang")
def build_mutang(**parameters):
    """Build MUTANG and its tangent-space mutation enumerator."""
    return MutangGenerator(MutationEnumerationInTangentSpace(**parameters))

__all__ = ["MutangGenerator", "MutationEnumerationInTangentSpace"]
