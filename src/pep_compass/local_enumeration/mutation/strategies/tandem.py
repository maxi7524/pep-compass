"""TANDEM nucleus-selection strategy."""

from __future__ import annotations

from pep_compass.local_enumeration.mutation.strategies.composition import (
    bounded_mutations,
    compose_mutant_distribution,
    nucleus_indices,
)
from pep_compass.local_enumeration.mutation.strategies.geometry import GeometryFilter


class TandemFilter(GeometryFilter):
    """Reweight the MUTANG product with TANDEM variant A and top-p selection."""

    def __init__(
        self,
        *args,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ):
        """Initialize TANDEM nucleus filtering.

        :param args: Positional arguments forwarded to :class:`GeometryFilter`.
        :param top_p: Cumulative probability mass retained.
        :param temperature: Positive score-scaling temperature.
        :param kwargs: Keyword arguments forwarded to :class:`GeometryFilter`.
        """
        super().__init__(*args, **kwargs)
        self.top_p = top_p
        self.temperature = temperature

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> list[str]:
        """Score the bounded product and retain its top-p nucleus.

        :param parent_peptide: Sequence defining identity choices.
        :param mutations: MUTANG residue choices.
        :return: TANDEM-selected candidate strings.
        """
        bounded = bounded_mutations(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        distribution = compose_mutant_distribution(
            parent_peptide,
            bounded,
            self.pairwise_potential(parent_peptide),
            alphabet=self.alphabet,
            include_parent_residue=True,
            maximum_candidates=self.maximum_candidates,
        )
        self.last_generated_count = len(distribution.sequences)
        selected = nucleus_indices(
            distribution.log_potentials, self.top_p, self.temperature
        )
        return [distribution.sequences[index] for index in selected]
