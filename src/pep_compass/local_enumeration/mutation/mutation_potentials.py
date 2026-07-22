"""Potentials used to score combinations of MUTANG mutations.

This module consolidates the reusable parts of the historical
``mutation/mutation_potentials.py`` modules and the experiment-local potential
classes from ``rl_trials``. Experiment orchestration is intentionally kept in
the runner and filters; this module only represents and scores mutation choices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import itertools

import numpy as np
import torch

# NOTE: Nie kopiujemy historycznego `sampling/sorbes.py`; kanoniczny SORBES
# pozostaje w `pep_compass.local_enumeration.sampling_walker`.
from pep_compass.local_enumeration.sampling_walker import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25


class MutantDistribution(NamedTuple):
    """Scored peptide candidates ordered from the highest potential."""

    sequences: list[str]
    log_potentials: np.ndarray  # 1-D float64, sorted descending


class MutationPotential(ABC):
    """Base class for mutation potential functions.

    Subclasses implement ``compute``, which maps a parent peptide and a set of
    candidate single-position mutations either to independent residue scores or
    to scores for complete Cartesian-product combinations.
    """

    @abstractmethod
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]] | dict[tuple[int, ...], float]:
        """Return scores for candidate mutation choices.

        Args:
            parent_peptide: Parent peptide sequence.
            mutations: Position indices mapped to candidate amino-acid indices,
                in the format returned by ``get_mutations_from_s_u``.

        Returns:
            Either per-position scores ``{position: {aa_index: score}}`` or
            combination scores ``{aa_index_tuple: score}``. Tuple elements
            follow ``sorted(mutations)``.
        """


class DecoderLogProbabilityPotential(MutationPotential):
    """Log-probability of each mutant residue under the parent distribution."""

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
        )[0]  # (max_len, alphabet_size)
        return {
            position: {
                amino_acid: log_probabilities[position, amino_acid].item()
                for amino_acid in amino_acids
            }
            for position, amino_acids in mutations.items()
        }


class ProjectedDirectionPairwiseSimilarityPotential(MutationPotential):
    r"""TANDEM potential based on pairwise projected mutation directions.

    For every complete combination, mutation directions are compared pairwise.
    Mutated-mutated pairs contribute ``log((1 + cos) / 2)`` and pairs in which
    exactly one residue is mutated contribute ``log((1 - cos) / 2)``. The
    contributions are averaged over all pairs involving at least one mutation.
    A single-position mutation has score zero because it has no pair.

    The returned key is a tuple of amino-acid indices ordered according to
    ``sorted(mutations)``. The all-parent combination is excluded.

    ``onehot`` represents a change at position :math:`l` to residue :math:`a`
    by :math:`e_{l,a}`. ``diff`` represents the actual displacement from the
    parent, :math:`e_{l,a} - e_{l,p_l}`.
    """

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
        # "onehot": the direction is the ambient one-hot e_(position, target).
        # "diff": it is e_(position, target) - e_(position, parent), which
        # represents movement from the current residue to the target residue.
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
        """Return the cached horizontal ambient-to-latent projection matrix."""
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
        r"""Represent ambient one-hot directions by latent pull-backs.

        This is Variant A from ``rl_trials``: ``J_h^+ e`` is read from columns
        of the horizontal projection matrix. Euclidean cosine between these
        vectors is therefore a whitened, inverse-singular-value-weighted
        similarity.
        """
        return self._projection_matrix()[:, flat_indices].T

    def position_vectors(
        self,
        position: int,
        amino_acids: torch.Tensor,
        parent_amino_acid: int,
    ) -> torch.Tensor:
        """Return normalized candidate directions for one sequence position.

        In ``onehot`` mode this returns representations of ``e_(position, aa)``.
        In ``diff`` mode the parent representation is subtracted before
        normalization; the identity choice consequently becomes a zero vector.
        """
        flat_indices = position * len(self.alphabet) + amino_acids
        vectors = self._raw_directions(flat_indices)
        if self.direction_mode == "diff":
            parent_index = torch.tensor(
                [position * len(self.alphabet) + parent_amino_acid],
                device=amino_acids.device,
            )
            vectors = vectors - self._raw_directions(parent_index)  # Broadcast (1, d).
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

        # Parent indices identify identity choices in the Cartesian product.
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
    """LAMS hard-viability score for a complete mutation combination.

    This is the product variant from ``rl_trials/scripts/lebo_plus.py``. For
    every pair of mutated residues, the whitened cosine is computed. The
    candidate score is the minimum across those pairs, so every mutation in a
    multi-mutant must remain compatible with every other mutation.
    A candidate with one mutation has score ``+inf`` and therefore always passes
    a finite downstream threshold. The all-parent candidate is excluded.
    """

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


def compose_mutant_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    potential: MutationPotential,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    include_parent_residue: bool = False,
    maximum_candidates: int | None = None,
) -> MutantDistribution:
    """Compose, score, and sort the Cartesian product of mutation choices.

    Per-position potentials are added across positions. Tuple-keyed potentials
    already describe complete combinations and are materialized directly. When
    requested, the parent's residue is added at each position so the product
    includes candidates mutating only a subset of the available positions.
    ``maximum_candidates`` limits the returned highest-scoring rows.
    """
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
