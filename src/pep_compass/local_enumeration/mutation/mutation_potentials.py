"""Potentials used to score combinations of MUTANG mutations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import itertools

import numpy as np
import torch

# NOTE: Nie przenosiłem `sampling/sorbes.py`; całość jest zaimplementowana w `pep_compass.local_enumeration.sampling_walker`.
from pep_compass.local_enumeration.sampling_walker import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25


class MutantDistribution(NamedTuple):
    """Scored peptide candidates ordered from the highest potential."""

    sequences: list[str]
    log_potentials: np.ndarray


class MutationPotential(ABC):
    """Base class for mutation potential functions.

    Subclasses must implement ``compute``, which maps a parent peptide and a
    set of candidate single-position mutations to scalar potentials.
    """

    @abstractmethod
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]] | dict[tuple[int, ...], float]:
        """Return per-position, per-amino-acid log-potentials.

        Args:
            parent_peptide: The parent peptide sequence.
            mutations: Mapping from position index to candidate amino acid
                indices (same format as ``get_mutations_from_s_u_standard``).

        Returns:
            Nested dict ``{position: {aa_index: potential_value}}``.
        """```


class DecoderLogProbabilityPotential(MutationPotential):
    """Score residues with decoder log-probability at the parent latent point."""

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        alphabet: list[str] | None = None,
    ):
        self.encoder_decoder = encoder_decoder
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]]:
        latent = self.encoder_decoder.encode_peptides([parent_peptide])
        log_probabilities = self.encoder_decoder.decoder_forward(
            latent,
            softmax=False,
            log_softmax=True,
            flatten=False,
        )[0]
        return {
            position: {
                amino_acid: log_probabilities[position, amino_acid].item()
                for amino_acid in amino_acids
            }
            for position, amino_acids in mutations.items()
        }


class ProjectedDirectionPairwiseSimilarityPotential(MutationPotential):
    """TANDEM potential based on pairwise projected mutation directions."""

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        taken_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        if direction_mode not in {"onehot", "diff"}:
            raise ValueError("direction_mode must be 'onehot' or 'diff'")
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.direction_mode = direction_mode
        self.taken_taken_transform = taken_taken_transform or self._log_taken_taken
        self.taken_not_taken_transform = (
            taken_not_taken_transform or self._log_taken_not_taken
        )

    @staticmethod
    def _log_taken_taken(values: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp((1.0 + values) / 2.0, min=1e-12, max=1.0))

    @staticmethod
    def _log_taken_not_taken(values: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp((1.0 - values) / 2.0, min=1e-12, max=1.0))

    def _projection_matrix(self) -> torch.Tensor:
        if self.tangent_space.projection_matrix is None:
            ambient_dimension = DEFAULT_MAX_LEN * len(self.alphabet)
            self.tangent_space.project_ambient_vector_to_horizontal_space(
                torch.zeros(
                    ambient_dimension,
                    device=self.tangent_space.device,
                    dtype=self.tangent_space.U.dtype,
                )
            )
        return self.tangent_space.projection_matrix

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        return self._projection_matrix()[:, flat_indices].T

    def position_vectors(
        self,
        position: int,
        amino_acids: torch.Tensor,
        parent_amino_acid: int,
    ) -> torch.Tensor:
        flat_indices = position * len(self.alphabet) + amino_acids
        vectors = self._raw_directions(flat_indices)
        if self.direction_mode == "diff":
            parent_index = torch.tensor(
                [position * len(self.alphabet) + parent_amino_acid],
                device=amino_acids.device,
            )
            vectors = vectors - self._raw_directions(parent_index)
        return vectors / (torch.linalg.norm(vectors, dim=1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        positions = sorted(mutations)
        if not positions:
            return {}

        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        device = self.tangent_space.device
        parent_amino_acids = torch.tensor(
            [self.alphabet.index(padded_parent[position]) for position in positions],
            device=device,
            dtype=torch.long,
        )
        choices = [
            torch.tensor(mutations[position], device=device, dtype=torch.long)
            for position in positions
        ]
        vectors = [
            self.position_vectors(position, amino_acids, int(parent_amino_acids[i]))
            for i, (position, amino_acids) in enumerate(zip(positions, choices))
        ]
        ranges = [torch.arange(len(choice), device=device) for choice in choices]
        combination_indices = (
            ranges[0].unsqueeze(1)
            if len(ranges) == 1
            else torch.cartesian_prod(*ranges)
        )
        combinations = torch.stack(
            [choices[i][combination_indices[:, i]] for i in range(len(positions))],
            dim=1,
        )
        mutation_mask = combinations != parent_amino_acids
        valid = mutation_mask.any(dim=1)

        if len(positions) == 1:
            scores = torch.zeros(len(combinations), device=device)
        else:
            selected_vectors = torch.stack(
                [vectors[i][combination_indices[:, i]] for i in range(len(positions))],
                dim=1,
            )
            similarities = selected_vectors @ selected_vectors.transpose(1, 2)
            left, right = torch.triu_indices(
                len(positions), len(positions), offset=1, device=device
            )
            pair_similarities = similarities[:, left, right]
            left_mutated = mutation_mask[:, left]
            right_mutated = mutation_mask[:, right]
            both_mutated = left_mutated & right_mutated
            one_mutated = left_mutated ^ right_mutated
            included = both_mutated | one_mutated
            scores = (
                self.taken_taken_transform(pair_similarities) * both_mutated
                + self.taken_not_taken_transform(pair_similarities) * one_mutated
            ).sum(dim=1) / included.sum(dim=1).clamp(min=1)

        return {
            tuple(amino_acids): float(score)
            for amino_acids, score in zip(
                combinations[valid].cpu().tolist(), scores[valid].cpu().tolist()
            )
        }


class LamsAnchorSimilarityPotential(MutationPotential):
    """LAMS product score for the best viable anchor in each candidate."""

    def __init__(self, base_potential: ProjectedDirectionPairwiseSimilarityPotential):
        self.base_potential = base_potential
        self.alphabet = base_potential.alphabet

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
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
            partner_mask = mutation_mask.unsqueeze(1).expand_as(similarities)
            diagonal = torch.eye(
                len(positions), device=device, dtype=torch.bool
            ).unsqueeze(0)
            partner_mask = partner_mask & ~diagonal
            anchor_minimum = torch.where(
                partner_mask,
                similarities,
                torch.full_like(similarities, float("inf")),
            )
            anchor_minimum = anchor_minimum.min(dim=2).values
            anchor_minimum = torch.where(
                mutation_mask,
                anchor_minimum,
                torch.full_like(anchor_minimum, float("-inf")),
            )
            scores = anchor_minimum.max(dim=1).values
        return {
            tuple(amino_acids): float(score)
            for amino_acids, score in zip(
                combinations[valid].cpu().tolist(), scores[valid].cpu().tolist()
            )
        }


def compose_mutant_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    potential: MutationPotential,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    include_parent_residue: bool = False,
    maximum_candidates: int | None = None,
) -> MutantDistribution:
    """Compose, score and sort the Cartesian product of mutation choices."""
    alphabet = alphabet or DEFAULT_ALPHABET
    padded_parent = parent_peptide.ljust(max_len)
    augmented = {
        position: sorted(
            set(amino_acids)
            | (
                {alphabet.index(padded_parent[position])}
                if include_parent_residue
                else set()
            )
        )
        for position, amino_acids in mutations.items()
    }
    potentials = potential.compute(parent_peptide, augmented)
    if not potentials:
        return MutantDistribution([], np.array([], dtype=np.float64))

    positions = sorted(augmented)
    sequences: list[str] = []
    scores: list[float] = []
    if isinstance(next(iter(potentials)), tuple):
        for amino_acids, score in potentials.items():
            sequence = list(padded_parent)
            for position, amino_acid in zip(positions, amino_acids):
                sequence[position] = alphabet[amino_acid]
            sequences.append("".join(sequence[: len(parent_peptide)]))
            scores.append(score)
    else:
        choices = [list(potentials[position]) for position in positions]
        for amino_acids in itertools.product(*choices):
            sequence = list(padded_parent)
            score = 0.0
            for position, amino_acid in zip(positions, amino_acids):
                sequence[position] = alphabet[amino_acid]
                score += potentials[position][amino_acid]
            candidate = "".join(sequence[: len(parent_peptide)])
            if candidate != parent_peptide:
                sequences.append(candidate)
                scores.append(score)

    order = np.argsort(scores)[::-1]
    if maximum_candidates is not None:
        order = order[:maximum_candidates]
    return MutantDistribution(
        [sequences[index] for index in order],
        np.asarray([scores[index] for index in order], dtype=np.float64),
    )
