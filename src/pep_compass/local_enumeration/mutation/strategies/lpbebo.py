"""LPBEBO decoder-probability mutation strategy."""

from __future__ import annotations

import torch

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_ALPHABET,
    DEFAULT_MAXIMUM_CANDIDATES,
    MutationCandidateFilter,
    MutationPotential,
)
from pep_compass.local_enumeration.mutation.strategies.composition import (
    bounded_mutations,
    compose_mutant_distribution,
    nucleus_indices,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


class DecoderLogProbabilityPotential(MutationPotential):
    """Score each residue by its parent-conditioned decoder probability."""

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        alphabet: list[str] | None = None,
    ):
        """Initialize decoder scoring.

        :param encoder_decoder: Model used to encode and decode the parent.
        :param alphabet: Optional index-to-token mapping.
        """
        self.encoder_decoder = encoder_decoder
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]]:
        """Return decoder log-probability for every proposed residue.

        :param parent_peptide: Sequence defining the latent decoder condition.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Nested position and amino-acid log-probabilities.
        """
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


class LpbeboFilter(MutationCandidateFilter):
    """Apply decoder probability and nucleus selection to MUTANG choices."""

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        top_p: float = 0.9,
        temperature: float = 1.0,
        maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
        alphabet: list[str] | None = None,
    ):
        """Initialize LPBEBO filtering.

        :param encoder_decoder: Model defining parent-conditioned probabilities.
        :param top_p: Cumulative probability mass retained.
        :param temperature: Positive score-scaling temperature.
        :param maximum_candidates: Maximum mutation-product size.
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
    ) -> list[str]:
        """Return the top-p nucleus of the bounded mutation product.

        :param parent_peptide: Sequence defining the decoder distribution.
        :param mutations: MUTANG residue choices.
        :return: Decoder-probability-selected candidates.
        """
        bounded = bounded_mutations(
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
        selected = nucleus_indices(
            distribution.log_potentials, self.top_p, self.temperature
        )
        return [distribution.sequences[index] for index in selected]
