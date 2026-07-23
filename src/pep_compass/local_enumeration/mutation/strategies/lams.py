"""LAMS product viability strategy based on whitened latent similarity."""

from __future__ import annotations

import torch

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_MAX_LEN,
    MutationPotential,
)
from pep_compass.local_enumeration.mutation.strategies.composition import (
    bounded_mutations,
    compose_mutant_distribution,
)
from pep_compass.local_enumeration.mutation.strategies.geometry import (
    GeometryFilter,
    ProjectedDirectionPairwiseSimilarityPotential,
)


class LamsAnchorSimilarityPotential(MutationPotential):
    """Score a combination by its worst mutated-mutated cosine.

    This is the product variant used by ``rl_trials/scripts/lebo_plus.py``.
    Single mutations receive positive infinity and therefore pass every finite
    viability threshold. The argmax LAMS ablation is intentionally not included.
    """

    def __init__(self, base_potential: ProjectedDirectionPairwiseSimilarityPotential):
        """Initialize LAMS from variant-A mutation directions.

        :param base_potential: Potential providing normalized position vectors.
        """
        self.base_potential = base_potential
        self.alphabet = base_potential.alphabet

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        """Return the minimum pairwise cosine for each non-parent combination.

        :param parent_peptide: Sequence defining identity choices.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Complete residue tuples mapped to viability scores.
        """
        positions = sorted(mutations)
        if not positions:
            return {}
        device = self.base_potential.tangent_space.device
        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        parents = torch.tensor(
            [self.alphabet.index(padded_parent[position]) for position in positions],
            device=device,
            dtype=torch.long,
        )
        choices = [
            torch.tensor(mutations[position], device=device, dtype=torch.long)
            for position in positions
        ]
        vectors = [
            self.base_potential.position_vectors(position, choice, int(parents[i]))
            for i, (position, choice) in enumerate(zip(positions, choices))
        ]
        ranges = [torch.arange(len(choice), device=device) for choice in choices]
        indices = (
            ranges[0].unsqueeze(1)
            if len(ranges) == 1
            else torch.cartesian_prod(*ranges)
        )
        combinations = torch.stack(
            [choices[i][indices[:, i]] for i in range(len(positions))], dim=1
        )
        mutation_mask = combinations != parents
        valid = mutation_mask.any(dim=1)
        if len(positions) == 1:
            scores = torch.full((len(combinations),), float("inf"), device=device)
        else:
            selected = torch.stack(
                [vectors[i][indices[:, i]] for i in range(len(positions))], dim=1
            )
            similarities = selected @ selected.transpose(1, 2)
            left, right = torch.triu_indices(
                len(positions), len(positions), offset=1, device=device
            )
            pair_similarities = similarities[:, left, right]
            both_mutated = mutation_mask[:, left] & mutation_mask[:, right]
            masked_similarities = torch.where(
                both_mutated,
                pair_similarities,
                torch.full_like(pair_similarities, float("inf")),
            )
            has_pair = both_mutated.any(dim=1)
            scores = masked_similarities.min(dim=1).values
            scores = torch.where(
                has_pair, scores, torch.full_like(scores, float("inf"))
            )
        return {
            tuple(amino_acids): float(score)
            for amino_acids, score in zip(
                combinations[valid].cpu().tolist(), scores[valid].cpu().tolist()
            )
        }


class LamsFilter(GeometryFilter):
    """Apply the LAMS product viability threshold to MUTANG candidates."""

    def __init__(self, *args, similarity_threshold: float = 0.15, **kwargs):
        """Initialize LAMS hard-threshold filtering.

        :param args: Positional arguments forwarded to :class:`GeometryFilter`.
        :param similarity_threshold: Minimum global pairwise cosine.
        :param kwargs: Keyword arguments forwarded to :class:`GeometryFilter`.
        """
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> list[str]:
        """Return combinations whose worst mutated pair passes the threshold.

        :param parent_peptide: Sequence defining identity choices.
        :param mutations: MUTANG residue choices.
        :return: LAMS-compatible candidate strings.
        """
        bounded = bounded_mutations(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        potential = LamsAnchorSimilarityPotential(
            self.pairwise_potential(parent_peptide)
        )
        distribution = compose_mutant_distribution(
            parent_peptide,
            bounded,
            potential,
            alphabet=self.alphabet,
            include_parent_residue=True,
            maximum_candidates=self.maximum_candidates,
        )
        self.last_generated_count = len(distribution.sequences)
        return [
            sequence
            for sequence, score in zip(
                distribution.sequences, distribution.log_potentials
            )
            if score >= self.similarity_threshold
        ]
