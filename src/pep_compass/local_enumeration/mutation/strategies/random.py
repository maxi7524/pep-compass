"""Random controls for mutation proposal and selection."""

from __future__ import annotations

import random

import numpy as np

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_ALPHABET,
    DEFAULT_MAXIMUM_CANDIDATES,
    MutationCandidateFilter,
)
from pep_compass.local_enumeration.mutation.strategies.composition import (
    enumerate_sequences,
)


class RandomLeBoFilter(MutationCandidateFilter):
    """Provide random-walker and random-MUTANG experimental controls."""

    def __init__(
        self,
        mode: str = "walker",
        selection_fraction: float = 0.6,
        maximum_positions: int = 5,
        residues_per_position: int = 4,
        maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
        alphabet: list[str] | None = None,
    ):
        """Initialize a random control.

        :param mode: ``walker`` samples proposals; ``mutang_random`` randomly
            selects from real MUTANG proposals.
        :param selection_fraction: Probability mass retained by random selection.
        :param maximum_positions: Maximum positions sampled in walker mode.
        :param residues_per_position: Choices sampled at each walker position.
        :param maximum_candidates: Maximum mutation-product size.
        :param alphabet: Optional index-to-token mapping.
        :raises ValueError: If ``mode`` is unsupported.
        """
        if mode not in {"walker", "mutang_random"}:
            raise ValueError("mode must be 'walker' or 'mutang_random'")
        self.mode = mode
        self.selection_fraction = selection_fraction
        self.maximum_positions = maximum_positions
        self.residues_per_position = residues_per_position
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> list[str]:
        """Return candidates produced by the configured random control.

        :param parent_peptide: Sequence used to construct mutations.
        :param mutations: Real MUTANG choices, used by ``mutang_random``.
        :return: Randomly proposed or retained candidate strings.
        """
        if self.mode == "walker":
            positions = random.sample(
                range(len(parent_peptide)),
                min(len(parent_peptide), self.maximum_positions),
            )
            mutations = {
                position: random.sample(
                    [
                        index
                        for index, amino_acid in enumerate(
                            self.alphabet[1:], start=1
                        )
                        if amino_acid != parent_peptide[position]
                    ],
                    self.residues_per_position,
                )
                for position in positions
            }
        sequences = enumerate_sequences(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        self.last_generated_count = len(sequences)
        if self.mode == "mutang_random" and sequences:
            random_mass = np.random.random(len(sequences))
            random_mass /= random_mass.sum()
            order = np.argsort(random_mass)[::-1]
            cumulative = np.cumsum(random_mass[order])
            keep = np.empty(len(sequences), dtype=bool)
            keep[0] = True
            keep[1:] = cumulative[:-1] < self.selection_fraction
            sequences = [sequences[index] for index in order[keep]]
        return sequences
