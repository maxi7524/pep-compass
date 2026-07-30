"""Candidate filters for LPBEBO, LAMS, TANDEM, MOVE, and random controls.

The historical ``rl_trials`` scripts embedded a potential builder, Cartesian
product limiting, selection logic, and a complete SORBES loop in every file.
The migration maps their script-local objects as follows:

* ``scripts/run_lpbebo_optimization_apex.py`` and ``scripts/lpbebo_plus.py``
  become ``LpbeboFilter``;
* ``scripts/lebo_plus.py`` becomes ``LamsFilter``;
* ``scripts/lpbebo_plus.py`` also provides the TANDEM experiment now represented
  explicitly by ``TandemFilter``;
* ``scripts/move.py`` becomes ``MoveFilter``;
* ``scripts/random_lebo.py`` becomes ``RandomLeBoFilter``.

Repeated ``_cap_mutations`` and ``_top_p_filter`` functions become the shared
``_bounded_mutations`` and ``_nucleus_indices`` helpers. Potentials only score
choices, filters turn a MUTANG mutation map into selected peptide strings, and
local enumerators own the optional SORBES trajectory.
"""

from __future__ import annotations

import itertools
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from pep_compass.filters.strategies.decision_models.mutation_potentials import (
    DEFAULT_ALPHABET,
    DecoderLogProbabilityPotential,
    LamsAnchorSimilarityPotential,
    ProjectedDirectionPairwiseSimilarityPotential,
    compose_mutant_distribution,
)
from pep_compass.walkers.strategies.subriemannian import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


def _nucleus_indices(
    scores: np.ndarray,
    top_p: float,
    temperature: float,
) -> np.ndarray:
    """Select the smallest descending-score prefix with probability mass ``top_p``."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if len(scores) == 0 or top_p == 1.0:
        return np.arange(len(scores))
    order = np.argsort(scores)[::-1]
    scaled = scores[order] / temperature
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    keep = np.empty(len(scores), dtype=bool)
    keep[0] = True
    keep[1:] = cumulative[:-1] < top_p
    return order[keep]


@dataclass(slots=True)
class CandidateFilterTrace:
    """Store scores aligned with ``last_bounded_candidates``.

    :param scores: Raw method-specific scores before temperature scaling.
    :param probabilities: Softmax probabilities used by nucleus selection.
    :param cumulative_probabilities: Inclusive cumulative mass in score order.
    :param ranks: One-based descending-score ranks.
    """

    scores: np.ndarray
    probabilities: np.ndarray
    cumulative_probabilities: np.ndarray
    ranks: np.ndarray


def _score_trace(scores: np.ndarray, temperature: float) -> CandidateFilterTrace:
    """Build aligned probability and rank diagnostics for candidate scores."""
    if len(scores) == 0:
        empty = np.array([], dtype=float)
        return CandidateFilterTrace(empty, empty, empty, empty.astype(int))
    order = np.argsort(scores)[::-1]
    scaled = scores[order] / temperature
    ordered_probabilities = np.exp(scaled - scaled.max())
    ordered_probabilities /= ordered_probabilities.sum()
    probabilities = np.empty(len(scores), dtype=float)
    cumulative = np.empty(len(scores), dtype=float)
    ranks = np.empty(len(scores), dtype=int)
    probabilities[order] = ordered_probabilities
    cumulative[order] = np.cumsum(ordered_probabilities)
    ranks[order] = np.arange(1, len(scores) + 1)
    return CandidateFilterTrace(scores, probabilities, cumulative, ranks)


def _bounded_mutations(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    alphabet: list[str],
    maximum_candidates: int,
) -> dict[int, list[int]]:
    """Randomly reduce a mutation product until it fits the candidate budget.

    Parent residues are retained so combinations with a subset of positions
    mutated remain possible. This replaces the per-script ``_cap_mutations``
    implementations from ``lebo_plus.py``, ``lpbebo_plus.py``, and ``move.py``.
    """
    mutations = _valid_mutations(parent_peptide, mutations)
    padded_parent = parent_peptide.ljust(25)
    choices = {
        position: sorted(set(amino_acids) | {alphabet.index(padded_parent[position])})
        for position, amino_acids in mutations.items()
    }
    if not choices:
        return {}

    while math.prod(map(len, choices.values())) > maximum_candidates:
        position = max(choices, key=lambda key: len(choices[key]))
        parent_amino_acid = alphabet.index(padded_parent[position])
        alternatives = [
            amino_acid
            for amino_acid in choices[position]
            if amino_acid != parent_amino_acid
        ]
        if alternatives:
            choices[position].remove(random.choice(alternatives))
        elif len(choices) > 1:
            del choices[position]
        else:
            break
    return choices


def _proposal_count(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    alphabet: list[str],
) -> int:
    """Return Cartesian-product size before the candidate limit is applied."""
    mutations = _valid_mutations(parent_peptide, mutations)
    padded_parent = parent_peptide.ljust(25)
    choice_counts = [
        len(set(amino_acids) | {alphabet.index(padded_parent[position])})
        for position, amino_acids in mutations.items()
    ]
    return max(math.prod(choice_counts) - 1, 0) if choice_counts else 0


def _valid_mutations(
    parent_peptide: str,
    mutations: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Return unique MUTANG residue options within the peptide length.

    :param parent_peptide: Sequence to which mutations will be applied.
    :param mutations: Proposed residue indices grouped by model position.
    :return: Valid non-empty options for positions in the peptide sequence.
    """
    return {
        position: sorted(set(amino_acids))
        for position, amino_acids in mutations.items()
        if 0 <= position < len(parent_peptide) and amino_acids
    }


def _enumerate_sequences(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    alphabet: list[str],
    maximum_candidates: int,
) -> list[str]:
    """Materialize the bounded Cartesian product as peptide strings.

    :param parent_peptide: Sequence whose residues are replaced.
    :param mutations: Candidate amino-acid indices grouped by position.
    :param alphabet: Index-to-token mapping, including the padding token.
    :param maximum_candidates: Maximum size of the bounded product.
    :return: Candidate strings excluding the unchanged parent.
    """
    choices = _bounded_mutations(
        parent_peptide, mutations, alphabet, maximum_candidates
    )
    positions = sorted(choices)
    sequences = []
    for amino_acids in itertools.product(
        *(choices[position] for position in positions)
    ):
        sequence = list(parent_peptide)
        for position, amino_acid in zip(positions, amino_acids):
            if position < len(sequence):
                sequence[position] = alphabet[amino_acid]
        candidate = "".join(sequence)
        if candidate != parent_peptide:
            sequences.append(candidate)
    return list(dict.fromkeys(sequences))


class MutationCandidateFilter(ABC):
    """Filter a MUTANG mutation pool into peptide candidates."""

    last_generated_count: int = 0
    last_proposed_count: int = 0
    last_bounded_candidates: ClassVar[list[str]] = []
    last_filter_trace: CandidateFilterTrace | None = None

    def _record_scores(self, scores: np.ndarray, temperature: float = 1.0) -> None:
        """Record score diagnostics aligned with the bounded candidate list."""
        self.last_filter_trace = _score_trace(np.asarray(scores), temperature)

    @abstractmethod
    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Return selected peptide candidates.

        :param parent_peptide: Sequence from which candidates are generated.
        :param mutations: MUTANG mapping from positions to amino-acid indices.
        :return: Candidate strings accepted by the filtering policy.
        """


class LpbeboFilter(MutationCandidateFilter):
    """LPBEBO: decoder log-probability followed by nucleus selection.

    MUTANG supplies allowed residues. Their decoder log-probabilities at the
    parent latent point are added for each Cartesian-product candidate, then
    temperature-scaled top-p selection retains the most probable mass.
    """

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        top_p: float = 0.9,
        temperature: float = 1.0,
        maximum_candidates: int = 30_000,
        alphabet: list[str] | None = None,
    ):
        """Initialize decoder-probability filtering.

        :param encoder_decoder: Model used to evaluate the parent-conditioned
            decoder distribution.
        :param top_p: Cumulative probability mass retained by nucleus selection.
        :param temperature: Positive score-scaling temperature.
        :param maximum_candidates: Maximum product size before scoring.
        :param alphabet: Optional index-to-token mapping.
        """
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.potential = DecoderLogProbabilityPotential(encoder_decoder, self.alphabet)
        self.top_p = top_p
        self.temperature = temperature
        self.maximum_candidates = maximum_candidates

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Score the bounded MUTANG product and retain its top-p nucleus.

        :param parent_peptide: Sequence defining the decoder distribution.
        :param mutations: MUTANG residue choices.
        :return: Candidates selected by decoder log-probability.
        """
        self.last_proposed_count = _proposal_count(
            parent_peptide, mutations, self.alphabet
        )
        bounded = _bounded_mutations(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        distribution = compose_mutant_distribution(
            parent_peptide,
            bounded,
            self.potential,
            alphabet=self.alphabet,
            maximum_candidates=self.maximum_candidates,
        )
        self.last_generated_count = len(distribution.sequences)
        self.last_bounded_candidates = distribution.sequences
        self._record_scores(distribution.log_potentials, self.temperature)
        selected = _nucleus_indices(
            distribution.log_potentials, self.top_p, self.temperature
        )
        return [distribution.sequences[index] for index in selected]


class _GeometryFilter(MutationCandidateFilter):
    """Build a fresh tangent-space potential at the current parent peptide.

    This replaces ``DynamicSORBESPairwiseSimilarityPotential`` and
    ``DynamicSORBESMutangPlusPotential`` from the historical scripts. The
    decoder Jacobian is decomposed once per filter call; its SVD initializes the
    canonical ``SubRiemannianTangentSpace`` used by both LAMS and TANDEM.
    """

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        horizontal_threshold: float = 0.1,
        maximum_candidates: int = 30_000,
        alphabet: list[str] | None = None,
    ):
        """Initialize the shared tangent-geometry filter state.

        :param encoder_decoder: Model providing the decoder Jacobian.
        :param horizontal_threshold: Singular-value threshold defining the
            horizontal tangent subspace.
        :param maximum_candidates: Maximum product size before scoring.
        :param alphabet: Optional index-to-token mapping.
        """
        self.encoder_decoder = encoder_decoder
        self.horizontal_threshold = horizontal_threshold
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def _pairwise_potential(
        self,
        parent_peptide: str,
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> ProjectedDirectionPairwiseSimilarityPotential:
        """Build a projected-direction potential at the current parent.

        :param parent_peptide: Sequence at which the Jacobian is evaluated.
        :return: Pairwise potential backed by the current tangent space.
        """
        if tangent_space is not None:
            return ProjectedDirectionPairwiseSimilarityPotential(
                tangent_space, self.alphabet
            )
        latent_batch = self.encoder_decoder.encode_peptides([parent_peptide])
        latent = latent_batch[0]
        jacobian = self.encoder_decoder.decoder_jacobian(latent_batch)[0]
        left, singular_values, right = torch.linalg.svd(jacobian, full_matrices=False)
        tangent_space = SubRiemannianTangentSpace(
            U=left,
            S=singular_values,
            V=right,
            horizontal_threshold=self.horizontal_threshold,
            device=str(latent.device),
        )
        return ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space, self.alphabet
        )


class LamsFilter(_GeometryFilter):
    """LAMS hard filter over the product viability score.

    The parent residue is included at each mutable position, every non-parent
    combination is scored by ``LamsAnchorSimilarityPotential``, and only scores
    greater than or equal to ``similarity_threshold`` survive. No decoder
    probability or nucleus selection is applied.
    """

    def __init__(self, *args, similarity_threshold: float = 0.15, **kwargs):
        """Initialize LAMS hard-threshold filtering.

        :param args: Positional arguments forwarded to :class:`_GeometryFilter`.
        :param similarity_threshold: Minimum global pairwise cosine accepted.
        :param kwargs: Keyword arguments forwarded to :class:`_GeometryFilter`.
        """
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Return combinations whose worst mutated pair passes the threshold.

        :param parent_peptide: Sequence defining identity residue choices.
        :param mutations: MUTANG residue choices.
        :return: LAMS-compatible candidate strings.
        """
        self.last_proposed_count = _proposal_count(
            parent_peptide, mutations, self.alphabet
        )
        bounded = _bounded_mutations(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        potential = LamsAnchorSimilarityPotential(
            self._pairwise_potential(parent_peptide, tangent_space)
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
        self.last_bounded_candidates = distribution.sequences
        self._record_scores(distribution.log_potentials)
        return [
            sequence
            for sequence, score in zip(
                distribution.sequences, distribution.log_potentials
            )
            if score >= self.similarity_threshold
        ]


class TandemFilter(_GeometryFilter):
    """TANDEM pairwise potential followed by nucleus selection.

    This is the reusable replacement for ``SamplingWithMutangPlusPlusLocalEnumerator``.
    It includes parent residues, scores the complete product with projected
    pairwise direction similarities, and retains a temperature-scaled top-p
    probability mass.
    """

    def __init__(
        self,
        *args,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ):
        """Initialize TANDEM nucleus filtering.

        :param args: Positional arguments forwarded to :class:`_GeometryFilter`.
        :param top_p: Cumulative probability mass retained after scoring.
        :param temperature: Positive score-scaling temperature.
        :param kwargs: Keyword arguments forwarded to :class:`_GeometryFilter`.
        """
        super().__init__(*args, **kwargs)
        self.top_p = top_p
        self.temperature = temperature

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Score the bounded product with TANDEM and apply top-p selection.

        :param parent_peptide: Sequence defining identity residue choices.
        :param mutations: MUTANG residue choices.
        :return: TANDEM-selected candidate strings.
        """
        self.last_proposed_count = _proposal_count(
            parent_peptide, mutations, self.alphabet
        )
        bounded = _bounded_mutations(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        distribution = compose_mutant_distribution(
            parent_peptide,
            bounded,
            self._pairwise_potential(parent_peptide, tangent_space),
            alphabet=self.alphabet,
            include_parent_residue=True,
            maximum_candidates=self.maximum_candidates,
        )
        self.last_generated_count = len(distribution.sequences)
        self.last_bounded_candidates = distribution.sequences
        self._record_scores(distribution.log_potentials, self.temperature)
        selected = _nucleus_indices(
            distribution.log_potentials, self.top_p, self.temperature
        )
        return [distribution.sequences[index] for index in selected]


class MoveFilter(MutationCandidateFilter):
    """MOVE filter based on net latent displacement of a mutation combination.

    Every distinct single substitution is encoded once. Its displacement from
    the parent embedding is cached, and displacements are added for each
    multi-mutant as a first-order superposition. The score is the negative norm
    of that sum, so nucleus selection favours candidates predicted to remain
    locally close to the parent. This replaces ``SamplingWithMoveLocalEnumerator``.
    """

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        top_p: float = 0.6,
        temperature: float = 1.0,
        maximum_candidates: int = 30_000,
        alphabet: list[str] | None = None,
    ):
        """Initialize MOVE displacement filtering.

        :param encoder_decoder: Model used to encode the parent and each unique
            single substitution.
        :param top_p: Cumulative probability mass retained after scoring.
        :param temperature: Positive score-scaling temperature.
        :param maximum_candidates: Maximum product size before scoring.
        :param alphabet: Optional index-to-token mapping.
        """
        self.encoder_decoder = encoder_decoder
        self.top_p = top_p
        self.temperature = temperature
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Rank candidates by approximate net latent displacement.

        :param parent_peptide: Sequence used as the latent displacement origin.
        :param mutations: MUTANG residue choices.
        :return: MOVE-selected candidate strings.
        """
        self.last_proposed_count = _proposal_count(
            parent_peptide, mutations, self.alphabet
        )
        sequences = _enumerate_sequences(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        self.last_generated_count = len(sequences)
        self.last_bounded_candidates = sequences
        if not sequences:
            self._record_scores(np.array([]), self.temperature)
            return []
        parent_latent = self.encoder_decoder.encode_peptides([parent_peptide])[0]
        single_mutations = {
            (position, sequence[position])
            for sequence in sequences
            for position in range(len(parent_peptide))
            if sequence[position] != parent_peptide[position]
        }
        single_keys = sorted(single_mutations)
        single_sequences = []
        for position, amino_acid in single_keys:
            sequence = list(parent_peptide)
            sequence[position] = amino_acid
            single_sequences.append("".join(sequence))
        single_latents = self.encoder_decoder.encode_peptides(single_sequences)
        displacements = single_latents - parent_latent
        displacement_index = {key: index for index, key in enumerate(single_keys)}
        candidate_displacements = torch.zeros(
            (len(sequences), parent_latent.shape[0]),
            device=parent_latent.device,
            dtype=parent_latent.dtype,
        )
        for candidate_index, sequence in enumerate(sequences):
            indices = [
                displacement_index[(position, sequence[position])]
                for position in range(len(parent_peptide))
                if sequence[position] != parent_peptide[position]
            ]
            candidate_displacements[candidate_index] = displacements[indices].sum(dim=0)
        scores = (
            (-torch.linalg.vector_norm(candidate_displacements, dim=1)).cpu().numpy()
        )
        self._record_scores(scores, self.temperature)
        selected = _nucleus_indices(scores, self.top_p, self.temperature)
        return [sequences[index] for index in selected]


class RandomLeBoFilter(MutationCandidateFilter):
    """Random proposal and random-selection controls for geometry-aware filters.

    ``walker`` discards MUTANG choices and samples positions and target residues
    uniformly. ``mutang_random`` preserves the MUTANG candidate product, assigns
    random probability mass, and keeps a random top-p nucleus, isolating the
    value of the LAMS/TANDEM/MOVE scoring rule. This replaces
    ``RandomLocalEnumerator`` from ``rl_trials/scripts/random_lebo.py``.
    """

    def __init__(
        self,
        mode: str = "walker",
        selection_fraction: float = 0.6,
        temperature: float = 1.0,
        maximum_positions: int = 5,
        residues_per_position: int = 4,
        maximum_candidates: int = 30_000,
        alphabet: list[str] | None = None,
    ):
        """Initialize a random proposal or random-selection control.

        :param mode: ``walker`` discards MUTANG choices; ``mutang_random``
            randomizes selection over the real MUTANG product.
        :param selection_fraction: Random probability mass retained in
            ``mutang_random`` mode.
        :param temperature: Positive scaling applied to random baseline scores.
        :param maximum_positions: Maximum positions sampled in ``walker`` mode.
        :param residues_per_position: Residue choices sampled per position.
        :param maximum_candidates: Maximum product size.
        :param alphabet: Optional index-to-token mapping.
        """
        if mode not in {"walker", "mutang_random"}:
            raise ValueError("mode must be 'walker' or 'mutang_random'")
        self.mode = mode
        self.selection_fraction = selection_fraction
        self.temperature = temperature
        self.maximum_positions = maximum_positions
        self.residues_per_position = residues_per_position
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    def filter_candidates(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        tangent_space: SubRiemannianTangentSpace | None = None,
    ) -> list[str]:
        """Return candidates produced by the configured random control.

        :param parent_peptide: Sequence used to construct mutations.
        :param mutations: Real MUTANG choices, used only by ``mutang_random``.
        :return: Randomly proposed or randomly retained candidate strings.
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
                        for index, amino_acid in enumerate(self.alphabet[1:], start=1)
                        if amino_acid != parent_peptide[position]
                    ],
                    self.residues_per_position,
                )
                for position in positions
            }
        self.last_proposed_count = _proposal_count(
            parent_peptide, mutations, self.alphabet
        )
        sequences = _enumerate_sequences(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        self.last_generated_count = len(sequences)
        self.last_bounded_candidates = sequences
        self.last_filter_trace = None
        if self.mode == "mutang_random" and sequences:
            random_scores = np.random.random(len(sequences))
            self._record_scores(random_scores, self.temperature)
            selected = _nucleus_indices(
                random_scores, self.selection_fraction, self.temperature
            )
            sequences = [sequences[index] for index in selected]
        return sequences
