from __future__ import annotations

# New potential class: ProjectedDirectionPairwiseSimilarityPotential
import math
from typing import Callable, NamedTuple


"""Mutation potential functions and scored mutant composition."""


import itertools
import random
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

        E = mean( +T(cos) for mutated-mutated pairs
                  -T(cos) for mutated-identity pairs )

    where by default T(x) = log((1 + x) / 2).

    Returns:
        dict[tuple[int, ...], float]

    The tuple is ordered according to sorted(mutations.keys()).
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        similarity_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET

        # Default transform: +log((1+x)/2).
        # Clamp keeps the argument strictly positive for numerical stability.
        def default_similarity_transform(x):
            arg = torch.clamp((1.0 + x) / 2.0, min=1e-12, max=1.0)
            return torch.log(arg)

        def default_taken_not_taken_transform(x):
            arg = torch.clamp((1.0 - x) / 2.0, min=1e-12, max=1.0)
            return torch.log(arg)

        self.similarity_transform = similarity_transform or default_similarity_transform
        self.taken_not_taken_transform = (
            taken_not_taken_transform or default_taken_not_taken_transform
        )

    def compute_similarity_matrix(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
        use_parent: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        For each position, compute the upper triangular similarity matrix (cosine similarity)
        between all candidate amino acids (including parent if use_parent=True).
        Returns a dict mapping position index to the similarity matrix (numpy array).
        """
        import numpy as np

        positions = sorted(mutations.keys())
        max_len = (
            getattr(self, "DEFAULT_MAX_LEN", 25)
            if hasattr(self, "DEFAULT_MAX_LEN")
            else 25
        )
        alphabet_size = len(self.alphabet)
        ambient_dim = max_len * alphabet_size
        device = self.tangent_space.device

        padded = parent_peptide.ljust(max_len)
        sim_matrices = {}
        for pos in positions:
            aa_indices = list(mutations[pos])
            if use_parent:
                parent_aa_idx = self.alphabet.index(padded[pos])
                if parent_aa_idx not in aa_indices:
                    aa_indices.append(parent_aa_idx)
            dirs = []
            for aa_idx in aa_indices:
                direction = torch.zeros(ambient_dim, device=device)
                flat_idx = pos * alphabet_size + aa_idx
                direction[flat_idx] = 1.0
                dirs.append(direction)
            dirs = torch.stack(dirs)
            projected = torch.stack(
                [
                    self.tangent_space.project_ambient_vector_to_horizontal_space(d)
                    for d in dirs
                ]
            )
            norms = torch.norm(projected, dim=1, keepdim=True)
            projected = projected / (norms + 1e-12)
            # Cosine similarity matrix
            cos_matrix = projected @ projected.T
            sim_matrices[pos] = cos_matrix.detach().cpu().numpy()
        return sim_matrices

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

            # Build cosine matrix over all selected residues (mutated + identity).
            # We include pairs where at least one residue is mutated:
            # mutated-mutated (taken-taken) and mutated-identity (taken-not-taken).
            selected_all = torch.stack(
                [vectors_per_position[i][combo_indices[i]] for i in range(n_pos)]
            )
            cos_matrix = selected_all @ selected_all.T  # (n_pos, n_pos)

            mut_mask = torch.tensor(
                mutated_positions_mask,
                dtype=torch.bool,
                device=device,
            )
            i_idx, j_idx = torch.triu_indices(n_pos, n_pos, offset=1, device=device)
            include_mask = mut_mask[i_idx] | mut_mask[j_idx]
            both_mutated_mask = mut_mask[i_idx] & mut_mask[j_idx]
            taken_not_taken_mask = include_mask & (~both_mutated_mask)

            if not torch.any(include_mask):
                result[aa_choice] = 0.0
                continue

            pairwise_cos_all = cos_matrix[i_idx, j_idx]

            # standard transform for taken-taken pairs
            score_taken_taken = self.similarity_transform(
                pairwise_cos_all[both_mutated_mask]
            ).sum()

            # other transform for taken-not-taken pairs
            score_taken_not_taken = self.taken_not_taken_transform(
                pairwise_cos_all[taken_not_taken_mask]
            ).sum()

            total_pairs = include_mask.sum().item()

            # Normalize by number of included pairs (taken-taken + taken-not-taken).
            energy = (score_taken_taken + score_taken_not_taken).item() / total_pairs

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
    sample_combinations: int | None = None,
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
        sample_combinations: If set and potential returns cartesian tuple keys,
            keep at most this many combinations sampled uniformly at random
            before sorting/top-k selection.
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

        if sample_combinations is not None and 0 < sample_combinations < len(scores):
            sampled_indices = random.sample(range(len(scores)), sample_combinations)
            sequences = [sequences[i] for i in sampled_indices]
            scores = [scores[i] for i in sampled_indices]

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
