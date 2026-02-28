from __future__ import annotations

# New potential class: ProjectedDirectionPairwiseSimilarityPotential
import math
from typing import Callable, NamedTuple


"""Mutation potential functions and scored mutant composition."""


import itertools
from abc import ABC, abstractmethod

import numpy as np
import torch

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

# Import SubRiemannianTangentSpace for projection
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace


DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25


class MutantDistribution(NamedTuple):
    sequences: list[str]
    log_potentials: np.ndarray  # 1-D float64, sorted descending


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
    ) -> dict[int, dict[int, float]]:
        """Return per-position, per-amino-acid log-potentials.

        Args:
            parent_peptide: The parent peptide sequence.
            mutations: Mapping from position index to candidate amino acid
                indices (same format as ``get_mutations_from_s_u_standard``).

        Returns:
            Nested dict ``{position: {aa_index: potential_value}}``.
        """
        ...


class DecoderLogProbPotential(MutationPotential):
    """Log-probability of each mutant residue under the parent's decoder distribution."""

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
        z_parent = self.encoder_decoder.encode_peptides([parent_peptide])
        log_probs = self.encoder_decoder.decoder_forward(
            z_parent,
            softmax=False,
            log_softmax=True,
            flatten=False,
        )
        log_probs = log_probs[0]  # (max_len, alphabet_size)

        potentials: dict[int, dict[int, float]] = {}
        for pos, aa_indices in mutations.items():
            potentials[pos] = {
                aa_idx: log_probs[pos, aa_idx].item() for aa_idx in aa_indices
            }
        return potentials


class ProjectedDirectionPairwiseSimilarityPotential(MutationPotential):
    """
    Full Cartesian-product pairwise similarity potential.

    For each full mutant combination:

        E = sum_{i<j} log((1 + cos(v_i, v_j)) / 2)

    Returns:
        dict[tuple[int, ...], float]

    The tuple is ordered according to sorted(mutations.keys()).
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        similarity_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET

        # T(x) = log((1+x)/2)
        self.similarity_transform = similarity_transform or (
            lambda x: torch.log(0.99 * (1.0 + x) / 2.0 + 1e-12)
        )

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:

        positions = sorted(mutations.keys())
        n_pos = len(positions)

        # Ambient space: (max_len, alphabet_size) flattened
        max_len = DEFAULT_MAX_LEN
        alphabet_size = len(self.alphabet)
        ambient_dim = max_len * alphabet_size
        device = self.tangent_space.device

        # ---- 1. Precompute projected + normalized vectors per position ----

        vectors_per_position = []

        for pos in positions:
            dirs = []
            for aa_idx in mutations[pos]:
                # Create one-hot vector in ambient (decoder output) space
                direction = torch.zeros(ambient_dim, device=device)
                flat_idx = pos * alphabet_size + aa_idx
                direction[flat_idx] = 1.0
                dirs.append(direction)

            dirs = torch.stack(dirs)  # (k_pos, ambient_dim)

            # Project each direction vector
            projected = torch.stack(
                [
                    self.tangent_space.project_ambient_vector_to_horizontal_space(d)
                    for d in dirs
                ]
            )

            norms = torch.norm(projected, dim=1, keepdim=True)
            projected = projected / (norms + 1e-12)

            vectors_per_position.append(projected)  # list of (k_pos, latent_dim)

        # ---- 2. Enumerate true Cartesian product ----

        result: dict[tuple[int, ...], float] = {}

        for combo_indices in itertools.product(
            *[range(len(mutations[p])) for p in positions]
        ):
            # wybrane wektory: (n_pos, dim)
            selected = torch.stack(
                [vectors_per_position[i][combo_indices[i]] for i in range(n_pos)]
            )

            # macierz cosinusów
            cos_matrix = selected @ selected.T  # (n_pos, n_pos)

            # tylko i<j
            i_idx, j_idx = torch.triu_indices(n_pos, n_pos, offset=1)
            pairwise_cos = cos_matrix[i_idx, j_idx]

            # transformacja
            energy = self.similarity_transform(pairwise_cos).sum().item()

            # tuple aa_idx (nie indeks lokalny!)
            aa_choice = tuple(
                mutations[positions[i]][combo_indices[i]] for i in range(n_pos)
            )

            result[aa_choice] = energy

        return result


def _build_sequences(
    padded: str,
    sorted_positions: list[int],
    pos_aa_indices: list[np.ndarray],
    alphabet: list[str],
    flat_indices: np.ndarray,
) -> list[str]:
    """Materialize peptide strings only for selected cartesian-product indices."""
    shapes = tuple(len(a) for a in pos_aa_indices)
    padded_arr = np.array(list(padded))
    seq_len = len(padded.rstrip())

    multi_indices = np.unravel_index(flat_indices, shapes)

    base = np.tile(padded_arr, (len(flat_indices), 1))
    for dim, pos in enumerate(sorted_positions):
        aa_idxs = pos_aa_indices[dim][multi_indices[dim]]
        base[:, pos] = np.array(alphabet)[aa_idxs]

    return ["".join(row[:seq_len]) for row in base]


def compose_mutant_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    potential: MutationPotential,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    include_parent_residue: bool = False,
    top_k: int | None = None,
) -> MutantDistribution:
    """Build a scored table of mutants from the cartesian product of per-position candidates.

    Scoring is vectorized with numpy; sequences are only materialized for the
    returned rows.

    Args:
        parent_peptide: The parent peptide sequence.
        mutations: Mapping from position index to candidate amino acid indices.
        potential: A ``MutationPotential`` instance used to score each residue choice.
        alphabet: Token list. Defaults to the standard 21-token alphabet.
        max_len: Maximum peptide length (for padding).
        include_parent_residue: If True, the parent's own residue at each
            mutable position is included as a candidate.
        top_k: If set, only the *top_k* highest-scoring mutants are returned.
            Uses ``argpartition`` for O(N) selection instead of full sort.

    Returns:
        ``MutantDistribution(sequences, log_potentials)`` sorted by
        ``log_potentials`` descending.
    """
    alphabet = alphabet or DEFAULT_ALPHABET
    padded = parent_peptide.ljust(max_len)

    if include_parent_residue:
        augmented: dict[int, list[int]] = {}
        for pos, aa_indices in mutations.items():
            parent_aa_idx = alphabet.index(padded[pos])
            combined = set(aa_indices)
            combined.add(parent_aa_idx)
            augmented[pos] = list(combined)
    else:
        augmented = mutations

    potentials = potential.compute(parent_peptide, augmented)

    # Check if potentials is Cartesian product format or per-position format
    if potentials and isinstance(next(iter(potentials.keys())), tuple):
        # Cartesian product format: dict[tuple[int, ...], float]
        sorted_positions = sorted(augmented.keys())
        sequences = []
        scores = []

        for aa_tuple, score in potentials.items():
            # Build sequence from aa_tuple
            seq_arr = list(padded)
            for i, pos in enumerate(sorted_positions):
                aa_idx = aa_tuple[i]
                seq_arr[pos] = alphabet[aa_idx]
            seq = "".join(seq_arr[: len(parent_peptide)])
            sequences.append(seq)
            scores.append(score)

        # Sort by score descending
        sorted_idxs = np.argsort(scores)[::-1]
        if top_k is not None:
            sorted_idxs = sorted_idxs[:top_k]

        return MutantDistribution(
            sequences=[sequences[i] for i in sorted_idxs],
            log_potentials=np.array([scores[i] for i in sorted_idxs], dtype=np.float64),
        )

    else:
        # Per-position format: dict[int, dict[int, float]]
        sorted_positions = sorted(potentials.keys())
        pos_aa_indices: list[np.ndarray] = []
        pos_log_pots: list[np.ndarray] = []
        for pos in sorted_positions:
            items = list(potentials[pos].items())
            pos_aa_indices.append(np.array([aa for aa, _ in items], dtype=np.intp))
            pos_log_pots.append(np.array([lp for _, lp in items], dtype=np.float64))

        total_log_pots = sum(np.meshgrid(*pos_log_pots, indexing="ij"))
        flat_pots = total_log_pots.ravel()

        if top_k is not None and top_k < len(flat_pots):
            kth = len(flat_pots) - top_k
            part_idx = np.argpartition(flat_pots, kth)[kth:]
            order = np.argsort(flat_pots[part_idx])[::-1]
            selected = part_idx[order]
        else:
            selected = np.argsort(flat_pots)[::-1]

        sequences = _build_sequences(
            padded,
            sorted_positions,
            pos_aa_indices,
            alphabet,
            selected,
        )

        return MutantDistribution(
            sequences=sequences,
            log_potentials=flat_pots[selected],
        )
