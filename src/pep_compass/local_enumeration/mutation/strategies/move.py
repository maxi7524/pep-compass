"""MOVE net latent-displacement mutation strategy."""

from __future__ import annotations

import torch

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_ALPHABET,
    DEFAULT_MAXIMUM_CANDIDATES,
    MutationCandidateFilter,
)
from pep_compass.local_enumeration.mutation.strategies.composition import (
    enumerate_sequences,
    nucleus_indices,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


class MoveFilter(MutationCandidateFilter):
    """Prefer combinations with small approximate net latent displacement.

    Every distinct single substitution is encoded once. Candidate displacement
    is the sum of its single-substitution displacements, matching the forward
    operationalization described for MOVE in the thesis.
    """

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        top_p: float = 0.6,
        temperature: float = 1.0,
        maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
        alphabet: list[str] | None = None,
    ):
        """Initialize MOVE filtering.

        :param encoder_decoder: Model used to encode parent and single mutants.
        :param top_p: Cumulative probability mass retained.
        :param temperature: Positive score-scaling temperature.
        :param maximum_candidates: Maximum mutation-product size.
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
    ) -> list[str]:
        """Rank candidates by approximate net latent displacement.

        :param parent_peptide: Sequence used as the displacement origin.
        :param mutations: MUTANG residue choices.
        :return: MOVE-selected candidate strings.
        """
        sequences = enumerate_sequences(
            parent_peptide, mutations, self.alphabet, self.maximum_candidates
        )
        self.last_generated_count = len(sequences)
        if not sequences:
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
            -torch.linalg.vector_norm(candidate_displacements, dim=1)
        ).cpu().numpy()
        selected = nucleus_indices(scores, self.top_p, self.temperature)
        return [sequences[index] for index in selected]
