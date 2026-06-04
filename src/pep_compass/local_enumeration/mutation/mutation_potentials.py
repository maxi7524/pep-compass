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


# --------------------------------------------------------------------------- #
# Pair transforms used by the pairwise-similarity potentials.
# Each pair (i, j) of single-position mutations contributes, to a candidate's energy:
#   * the "taken-taken" transform  T_tt(x)   when both mutations are applied,
#   * the "taken-not-taken" transform T_tnt(x) when exactly one is applied,
# with x = CS(v_i, v_j) in [-1, 1]. The required monotonic intuitions are:
#   T_tt  increasing in x  (reward aligned, penalise opposed mutations taken together),
#   T_tnt decreasing in x  (penalise taking only one of a parallel/cooperative pair).
# --------------------------------------------------------------------------- #
def log_taken_taken_transform(x):
    """Default: log((1 + x) / 2).  0 at x=1, -> -inf at x=-1 (clamped)."""
    return torch.log(torch.clamp((1.0 + x) / 2.0, min=1e-12, max=1.0))


def log_taken_not_taken_transform(x):
    """Default: log((1 - x) / 2).  0 at x=-1, -> -inf at x=+1 (clamped)."""
    return torch.log(torch.clamp((1.0 - x) / 2.0, min=1e-12, max=1.0))


def linear_taken_taken_transform(x):
    """Bounded linear alternative: (x - 1) / 2 in [-1, 0].  0 at x=1, -1 at x=-1.

    Satisfies the same monotonic intuition as the log form (increasing in x, maximal for
    perfectly aligned mutations) but is finite at x=-1, so strongly-opposed pairs are not
    clamped to a flat -inf and remain rankable."""
    return (x - 1.0) / 2.0


def linear_taken_not_taken_transform(x):
    """Bounded linear alternative: -(x + 1) / 2 in [-1, 0].  0 at x=-1, -1 at x=+1.

    Decreasing in x: taking only one of a parallel pair (x->+1) is maximally penalised,
    while independent/opposed mutations (x<=0) are not."""
    return -(x + 1.0) / 2.0


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
        direction_mode: str = "onehot",
    ):
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET
        # "onehot": a mutation l->a is the ambient one-hot e_{l,a}.
        # "diff":   it is the difference e_{l,a} - e_{l,p_l} (+1 at target, -1 at parent residue),
        #           a more faithful "movement from parent to target" direction.
        if direction_mode not in ("onehot", "diff"):
            raise ValueError(f"direction_mode must be 'onehot' or 'diff', got {direction_mode!r}")
        self.direction_mode = direction_mode

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

    # ------------------------------------------------------------------ #
    # Direction construction (overridden by the metric variant)
    # ------------------------------------------------------------------ #
    def _ensure_projection(self) -> torch.Tensor:
        """Return the cached ambient->latent horizontal projection matrix (latent, ambient)."""
        ambient_dim = DEFAULT_MAX_LEN * len(self.alphabet)
        if self.tangent_space.projection_matrix is None:
            _ = self.tangent_space.project_ambient_vector_to_horizontal_space(
                torch.zeros(ambient_dim, device=self.tangent_space.device)
            )
        return self.tangent_space.projection_matrix

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Representation of the ambient one-hot directions at the given flat indices.

        Variant A (this class): the latent pseudo-inverse pull-back ``J_h^+ e``, i.e.\\ the
        columns of the horizontal projection matrix. Comparing these by Euclidean cosine
        gives the whitened (inverse-singular-value weighted) similarity.
        """
        proj = self._ensure_projection()  # (latent, ambient)
        return proj[:, flat_indices].T    # (n, latent)

    def _normalized_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        vecs = self._raw_directions(flat_indices)
        norms = torch.linalg.norm(vecs, dim=1, keepdim=True)
        return vecs / (norms + 1e-12)

    def _position_vectors(
        self, pos: int, aa_indices: torch.Tensor, parent_aa_idx: int
    ) -> torch.Tensor:
        """Normalised direction vectors for the candidate residues at one position.

        ``onehot``: representation of e_{pos,a}. ``diff``: representation of
        e_{pos,a} - e_{pos,parent}, built before normalisation (linearity of ``_raw_directions``);
        the parent/identity candidate then maps to the zero direction.
        """
        alphabet_size = len(self.alphabet)
        flat = pos * alphabet_size + aa_indices
        vecs = self._raw_directions(flat)
        if self.direction_mode == "diff":
            parent_flat = torch.tensor(
                [pos * alphabet_size + parent_aa_idx],
                device=aa_indices.device, dtype=torch.long,
            )
            vecs = vecs - self._raw_directions(parent_flat)  # broadcast (1, dim)
        norms = torch.linalg.norm(vecs, dim=1, keepdim=True)
        return vecs / (norms + 1e-12)

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
        max_len = DEFAULT_MAX_LEN
        alphabet_size = len(self.alphabet)
        device = self.tangent_space.device

        padded = parent_peptide.ljust(max_len)
        sim_matrices = {}
        for pos in positions:
            parent_aa_idx = self.alphabet.index(padded[pos])
            aa_indices = list(mutations[pos])
            if use_parent and parent_aa_idx not in aa_indices:
                aa_indices.append(parent_aa_idx)
            aa_t = torch.tensor(aa_indices, device=device, dtype=torch.long)
            projected = self._position_vectors(pos, aa_t, parent_aa_idx)
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
        if not positions:
            return {}

        n_pos = len(positions)

        # Get parent amino acid indices for identity detection
        padded = parent_peptide.ljust(DEFAULT_MAX_LEN)
        parent_aa_indices = torch.tensor(
            [self.alphabet.index(padded[pos]) for pos in positions],
            device=self.tangent_space.device,
            dtype=torch.long,
        )

        # Ambient space: (max_len, alphabet_size) flattened
        max_len = DEFAULT_MAX_LEN
        alphabet_size = len(self.alphabet)
        device = self.tangent_space.device

        # ---- 1. Precompute projected + normalized vectors per position ----
        # The direction representation is provided by ``_normalized_directions`` and is the
        # only thing that differs between the projected (Variant A) and metric (Variant B)
        # potentials.
        vectors_per_position = []
        aa_indices_per_position = []

        for i_pos, pos in enumerate(positions):
            aa_indices = torch.tensor(
                mutations[pos], device=device, dtype=torch.long
            )
            aa_indices_per_position.append(aa_indices)
            parent_aa_idx = int(parent_aa_indices[i_pos].item())
            vectors_per_position.append(
                self._position_vectors(pos, aa_indices, parent_aa_idx)
            )

        # ---- 2. Enumerate Cartesian product (vectorized) ----
        index_ranges = [
            torch.arange(len(a), device=device) for a in aa_indices_per_position
        ]
        if n_pos == 1:
            combo_indices = index_ranges[0].unsqueeze(1)
        else:
            combo_indices = torch.cartesian_prod(*index_ranges)

        aa_choice = torch.stack(
            [
                aa_indices_per_position[i][combo_indices[:, i]]
                for i in range(n_pos)
            ],
            dim=1,
        )

        mut_mask = aa_choice != parent_aa_indices
        has_mutation = mut_mask.any(dim=1)

        if n_pos < 2:
            energy = torch.zeros(combo_indices.shape[0], device=device)
        else:
            selected_vectors = torch.stack(
                [
                    vectors_per_position[i][combo_indices[:, i]]
                    for i in range(n_pos)
                ],
                dim=1,
            )
            cos_matrix = selected_vectors @ selected_vectors.transpose(1, 2)

            i_idx, j_idx = torch.triu_indices(n_pos, n_pos, offset=1, device=device)
            pairwise_cos_all = cos_matrix[:, i_idx, j_idx]

            mut_i = mut_mask[:, i_idx]
            mut_j = mut_mask[:, j_idx]
            include_mask = mut_i | mut_j
            both_mutated_mask = mut_i & mut_j
            taken_not_taken_mask = include_mask & (~both_mutated_mask)

            score_taken_taken = (
                self.similarity_transform(pairwise_cos_all) * both_mutated_mask
            ).sum(dim=1)
            score_taken_not_taken = (
                self.taken_not_taken_transform(pairwise_cos_all) * taken_not_taken_mask
            ).sum(dim=1)

            total_pairs = include_mask.sum(dim=1).clamp(min=1)
            energy = (score_taken_taken + score_taken_not_taken) / total_pairs

        valid_mask = has_mutation
        aa_choice_valid = aa_choice[valid_mask].cpu().tolist()
        energy_valid = energy[valid_mask].cpu().tolist()

        return {
            tuple(aa_tuple): float(score)
            for aa_tuple, score in zip(aa_choice_valid, energy_valid)
        }

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


class AmbientMetricPairwiseSimilarityPotential(
    ProjectedDirectionPairwiseSimilarityPotential
):
    r"""Pairwise similarity potential using the decoder pullback metric (paper variant).

    This is the metric-aware variant of \S sec:similarities: the similarity of two mutation
    directions is the cosine under the pullback metric ``G = J^T J``,

        CS(v_i, v_j) = <v_i, v_j>_G / (||v_i||_G ||v_j||_G).

    For the ambient one-hot mutation directions ``e_i`` read off the tangent space, this
    reduces to the *normalised horizontal projector* ``U_h U_h^T``: writing ``U_h`` for the
    left singular vectors of the decoder Jacobian whose singular values exceed
    ``horizontal_threshold`` (the kappa-stable subspace), ``CS_ij`` is the cosine of the rows
    ``U_h[i, :]`` and ``U_h[j, :]``. The metric basis ``U_h`` is therefore formed **once** in
    ``__init__`` and every pairwise similarity is an inner product of two of its rows -- no
    per-pair recomputation of ``G``.

    The pairwise aggregation (taken-taken ``+T(cos)``, taken-not-taken ``-T(cos)``, averaged
    over involved pairs) and the ``similarity_transform`` / ``taken_not_taken_transform``
    hooks are inherited unchanged from the projected variant; only the per-direction
    representation differs (rows of ``U_h`` instead of the whitened latent pull-back). With
    the default transform ``T(x) = log((1 + x) / 2)`` this is the TANDEM potential of the
    thesis evaluated with the decoder-pullback (rather than Euclidean-latent) angle.
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        similarity_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        super().__init__(
            tangent_space=tangent_space,
            alphabet=alphabet,
            similarity_transform=similarity_transform,
            taken_not_taken_transform=taken_not_taken_transform,
            direction_mode=direction_mode,
        )
        # Metric basis U_h: horizontal left-singular vectors, computed ONCE.
        S = tangent_space.S
        U = tangent_space.U
        horizontal_mask = torch.abs(S) > tangent_space.horizontal_threshold
        # U has shape (ambient_dim, n_singular); keep the horizontal columns -> (ambient, h).
        self.U_h = U[:, horizontal_mask].contiguous()

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Metric variant: the ambient one-hot direction ``e_i`` is represented by row ``i``
        of ``U_h``; the Euclidean cosine of these rows equals the metric-aware cosine
        ``CS`` under ``G = J^T J``."""
        return self.U_h[flat_indices]  # (n, horizontal_dim)


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
    if not potentials:
        return MutantDistribution(
            sequences=[],
            log_potentials=np.array([], dtype=np.float64),
        )

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
        if not sorted_positions:
            return MutantDistribution(
                sequences=[],
                log_potentials=np.array([], dtype=np.float64),
            )
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
