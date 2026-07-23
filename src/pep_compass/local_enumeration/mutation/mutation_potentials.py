"""Compatibility imports for mutation potentials.

New code should import these objects from
``pep_compass.local_enumeration.mutation.strategies``. This module preserves the
original API while the strategy-oriented package is reviewed independently.
"""

from pep_compass.local_enumeration.mutation.strategies import (
    AmbientMetricPairwiseSimilarityPotential,
    DEFAULT_ALPHABET,
    DEFAULT_MAX_LEN,
    DecoderLogProbabilityPotential,
    LamsAnchorSimilarityPotential,
    MutantDistribution,
    MutationPotential,
    ProjectedDirectionPairwiseSimilarityPotential,
    compose_mutant_distribution,
)

__all__ = [
    "AmbientMetricPairwiseSimilarityPotential",
    "DEFAULT_ALPHABET",
    "DEFAULT_MAX_LEN",
    "DecoderLogProbabilityPotential",
    "LamsAnchorSimilarityPotential",
    "MutantDistribution",
    "MutationPotential",
    "ProjectedDirectionPairwiseSimilarityPotential",
    "compose_mutant_distribution",
]
