"""Typed stage interfaces for the fixed MUTANG algorithm."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry
from pep_compass.configuration.objects import Configurable
from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.steps.mutations.strategies.mutang.schema import (
    MutationOption,
    SelectedDirections,
)


class GeometryStrategy(Configurable, ABC):
    """Resolve geometry aligned with the incoming candidates."""

    local_enumeration_only = False
    requirement = None

    @abstractmethod
    def resolve(
        self, batch: CandidateBatch, autoencoder
    ) -> tuple[StableTangentGeometry, ...]:
        """Return one geometry row per candidate."""


class DirectionStrategy(Configurable, ABC):
    """Select singular directions for mutation scoring."""

    @abstractmethod
    def select(self, singular_values: torch.Tensor) -> SelectedDirections:
        """Return indices into the singular-vector axis."""


class ScoringStrategy(Configurable, ABC):
    """Score residue substitutions from selected tangent directions."""

    @abstractmethod
    def score(
        self, left_vectors: torch.Tensor, selected: SelectedDirections
    ) -> torch.Tensor:
        """Return substitution scores with shape (L, A)."""


class MutationSelectionStrategy(Configurable, ABC):
    """Select scored substitutions for a sequence."""

    @abstractmethod
    def select(
        self, scores: torch.Tensor, sequence_length: int
    ) -> tuple[MutationOption, ...]:
        """Return admissible residue substitutions."""


class CombinationStrategy(Configurable, ABC):
    """Enumerate candidate sequences from selected substitutions."""

    @abstractmethod
    def generate(
        self,
        sequence: str,
        options: tuple[MutationOption, ...],
        rng: np.random.Generator,
    ) -> Iterator[str]:
        """Yield candidates using the run-owned random generator."""
