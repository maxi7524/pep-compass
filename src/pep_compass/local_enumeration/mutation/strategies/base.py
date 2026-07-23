"""Shared interfaces and values for mutation candidate strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25
DEFAULT_MAXIMUM_CANDIDATES = 30_000


class MutantDistribution(NamedTuple):
    """Scored peptide candidates ordered from the highest potential.

    :param sequences: Materialized peptide candidates.
    :param log_potentials: One-dimensional scores aligned with ``sequences``.
    """

    sequences: list[str]
    log_potentials: np.ndarray


class MutationPotential(ABC):
    """Score residue choices proposed by MUTANG."""

    @abstractmethod
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]] | dict[tuple[int, ...], float]:
        """Return scores for independent choices or complete combinations.

        :param parent_peptide: Parent peptide sequence.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Per-position scores or complete tuple scores. Tuple elements
            follow ``sorted(mutations)``.
        """


class MutationCandidateFilter(ABC):
    """Filter a MUTANG mutation pool into peptide candidates."""

    last_generated_count: int = 0

    @abstractmethod
    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> list[str]:
        """Return candidates accepted by the filtering strategy.

        :param parent_peptide: Sequence from which candidates are generated.
        :param mutations: MUTANG mapping from positions to amino-acid indices.
        :return: Accepted candidate peptide strings.
        """
