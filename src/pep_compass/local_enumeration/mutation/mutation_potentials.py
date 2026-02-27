"""Mutation potential functions and scored mutant composition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25


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
    include_parent_residue: bool = True,
    top_k: int | None = None,
) -> pd.DataFrame:
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
        DataFrame with columns ``sequence`` and ``log_potential``, sorted by
        ``log_potential`` descending.
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

    return pd.DataFrame(
        {
            "sequence": sequences,
            "log_potential": flat_pots[selected],
        }
    )
