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
        # When cos(v_i, v_j) = -1 (anti-aligned), this gives -inf
        # which correctly represents that such combinations should be avoided
        def default_similarity_transform(x):
            arg = 0.99 * (1.0 + x) / 2.0
            result = torch.where(
                arg > 0,
                torch.log(arg),
                torch.tensor(float("-inf"), dtype=x.dtype, device=x.device),
            )
            return result

        self.similarity_transform = similarity_transform or default_similarity_transform

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:

        positions = sorted(mutations.keys())
        n_pos = len(positions)

        # Get parent amino acid indices for identity detection
        padded = parent_peptide.ljust(DEFAULT_MAX_LEN)
        parent_aa_indices = [self.alphabet.index(padded[pos]) for pos in positions]

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
            # tuple aa_idx (nie indeks lokalny!)
            aa_choice = tuple(
                mutations[positions[i]][combo_indices[i]] for i in range(n_pos)
            )

            # ---- Identify non-identity positions (actual mutations) ----
            mutated_positions_mask = [
                aa_choice[i] != parent_aa_indices[i] for i in range(n_pos)
            ]
            mutated_indices = [i for i in range(n_pos) if mutated_positions_mask[i]]

            # Exclude parent peptide (all identities)
            if len(mutated_indices) == 0:
                continue

            # If only 1 mutation, no pairwise similarity to compute
            if len(mutated_indices) == 1:
                result[aa_choice] = 0.0
                continue

            # Select only vectors for mutated positions
            selected = torch.stack(
                [vectors_per_position[i][combo_indices[i]] for i in mutated_indices]
            )

            # macierz cosinusów
            cos_matrix = selected @ selected.T  # (n_mutated, n_mutated)

            # tylko i<j
            n_mutated = len(mutated_indices)
            i_idx, j_idx = torch.triu_indices(n_mutated, n_mutated, offset=1)
            pairwise_cos = cos_matrix[i_idx, j_idx]

            # transformacja
            energy = self.similarity_transform(pairwise_cos).sum().item()

            result[aa_choice] = energy

        return result

    def compute_with_identities(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        """
        Compute potentials with identity (parent amino acid) included as an option.

        This method automatically adds the parent amino acid at each position as a
        candidate, allowing the cartesian product to include combinations where some
        positions remain unchanged (identity).

        The parent peptide (all identities) is automatically excluded from results.

        Args:
            parent_peptide: The parent peptide sequence.
            mutations: Mapping from position index to candidate amino acid indices
                (without parent amino acids).

        Returns:
            dict[tuple[int, ...], float]: Potentials for all combinations except
            the parent peptide.
        """
        # Add parent amino acid as identity option for each position
        padded = parent_peptide.ljust(DEFAULT_MAX_LEN)
        mutations_with_identity = {}
        for pos, aa_indices in mutations.items():
            parent_aa_idx = self.alphabet.index(padded[pos])
            combined = set(aa_indices)
            combined.add(parent_aa_idx)
            mutations_with_identity[pos] = sorted(combined)

        # Compute potentials (parent is automatically excluded)
        return self.compute(parent_peptide, mutations_with_identity)

    def return_list_abovethreshold(
        self,
        parent_peptide: str,
        potentials: dict[tuple[int, ...], float],
        mutations: dict[int, list[int]],
        threshold: float = -0.5,
    ) -> tuple[list[str], list[float]]:
        """
        Return sequences and potentials above a threshold.

        Args:
            parent_peptide: The parent peptide sequence.
            potentials: Dict mapping amino acid tuples to potential values
                (from compute or compute_with_identities).
            mutations: Mapping from position index to candidate amino acid indices
                (needed to map tuples back to sequences).
            threshold: Minimum potential value (default: -0.5).

        Returns:
            Tuple of (sequences, potentials) where sequences are full peptide strings
            and potentials are their corresponding values, filtered by threshold.
        """
        padded = parent_peptide.ljust(DEFAULT_MAX_LEN)
        positions = sorted(mutations.keys())

        sequences = []
        potential_values = []

        for aa_tuple, score in potentials.items():
            if score >= threshold:
                # Build full sequence from aa_tuple
                seq_arr = list(padded)
                for i, pos in enumerate(positions):
                    aa_idx = aa_tuple[i]
                    seq_arr[pos] = self.alphabet[aa_idx]
                seq = "".join(seq_arr[: len(parent_peptide)])

                sequences.append(seq)
                potential_values.append(score)

        # Sort by potential value descending
        sorted_indices = np.argsort(potential_values)[::-1]
        sequences = [sequences[i] for i in sorted_indices]
        potential_values = [potential_values[i] for i in sorted_indices]

        return sequences, potential_values


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
