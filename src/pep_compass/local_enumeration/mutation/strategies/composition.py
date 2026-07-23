"""Cartesian-product composition shared by mutation strategies.

The historical experiment scripts repeated mutation-product limiting, nucleus
selection, and sequence materialization. These helpers keep those operations
consistent while leaving each strategy responsible only for its scoring rule.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_ALPHABET,
    DEFAULT_MAX_LEN,
    MutantDistribution,
    MutationPotential,
)


def nucleus_indices(
    scores: np.ndarray,
    top_p: float,
    temperature: float,
) -> np.ndarray:
    """Select the smallest descending-score prefix with mass ``top_p``.

    :param scores: One-dimensional candidate scores.
    :param top_p: Cumulative probability mass to retain.
    :param temperature: Positive softmax temperature.
    :return: Indices into the original score array, ordered by descending score.
    :raises ValueError: If ``top_p`` or ``temperature`` is invalid.
    """
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


def bounded_mutations(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    alphabet: list[str],
    maximum_candidates: int,
) -> dict[int, list[int]]:
    """Reduce a mutation product while retaining parent-residue choices.

    :param parent_peptide: Sequence providing identity choices.
    :param mutations: Candidate amino-acid indices grouped by position.
    :param alphabet: Index-to-token mapping including the padding token.
    :param maximum_candidates: Maximum Cartesian-product size.
    :return: Bounded choices including the parent residue at each retained
        position.
    """
    padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
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


def enumerate_sequences(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    alphabet: list[str],
    maximum_candidates: int,
) -> list[str]:
    """Materialize a bounded mutation product without the unchanged parent.

    :param parent_peptide: Sequence whose residues are replaced.
    :param mutations: Candidate amino-acid indices grouped by position.
    :param alphabet: Index-to-token mapping including the padding token.
    :param maximum_candidates: Maximum Cartesian-product size.
    :return: Candidate peptide strings.
    """
    choices = bounded_mutations(
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
    return sequences


def compose_mutant_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    potential: MutationPotential,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    include_parent_residue: bool = False,
    maximum_candidates: int | None = None,
) -> MutantDistribution:
    """Compose, score, and order a Cartesian product of mutation choices.

    :param parent_peptide: Sequence providing unchanged residues.
    :param mutations: Candidate amino-acid indices grouped by position.
    :param potential: Strategy used to score the choices.
    :param alphabet: Optional index-to-token mapping including the padding token.
    :param max_len: Length used to pad the parent for positional lookup.
    :param include_parent_residue: Whether every mutable position can remain
        unchanged.
    :param maximum_candidates: Optional maximum number of returned rows.
    :return: Candidate sequences and aligned scores in descending order.
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
