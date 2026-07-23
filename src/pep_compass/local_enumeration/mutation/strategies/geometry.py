"""TANDEM direction similarities and shared decoder geometry.

Variant A is the recommended default: it compares inverse-singular-value
whitened latent pull-backs. Variant B uses the stable ambient pullback projector
and is retained for reproducing the comparison reported in the thesis.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from pep_compass.local_enumeration.mutation.strategies.base import (
    DEFAULT_ALPHABET,
    DEFAULT_MAX_LEN,
    DEFAULT_MAXIMUM_CANDIDATES,
    MutationCandidateFilter,
    MutationPotential,
)
from pep_compass.local_enumeration.sampling_walker import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


class ProjectedDirectionPairwiseSimilarityPotential(MutationPotential):
    r"""TANDEM variant A based on whitened latent mutation directions.

    Mutated-mutated pairs contribute ``log((1 + cos) / 2)`` and pairs with
    exactly one selected mutation contribute ``log((1 - cos) / 2)``. Pair
    contributions are averaged so the score does not scale with mutation count.
    ``onehot`` represents :math:`e_{l,a}` and ``diff`` represents
    :math:`e_{l,a} - e_{l,p_l}`.
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        taken_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        """Initialize the recommended TANDEM similarity variant.

        :param tangent_space: Local SVD geometry used to pull ambient residue
            directions into horizontal latent space.
        :param alphabet: Optional index-to-token mapping.
        :param taken_taken_transform: Optional transform for two selected
            mutations. The logarithmic TANDEM transform is the default.
        :param taken_not_taken_transform: Optional transform for exactly one
            selected mutation. The logarithmic TANDEM transform is the default.
        :param direction_mode: ``onehot`` or ``diff`` direction encoding.
        :raises ValueError: If ``direction_mode`` is unsupported.
        """
        if direction_mode not in {"onehot", "diff"}:
            raise ValueError("direction_mode must be 'onehot' or 'diff'")
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.direction_mode = direction_mode
        self.taken_taken_transform = taken_taken_transform or self._log_taken_taken
        self.taken_not_taken_transform = (
            taken_not_taken_transform or self._log_taken_not_taken
        )

    @staticmethod
    def _log_taken_taken(values: torch.Tensor) -> torch.Tensor:
        """Map aligned jointly selected directions to higher log-potentials."""
        return torch.log(torch.clamp((1.0 + values) / 2.0, min=1e-12, max=1.0))

    @staticmethod
    def _log_taken_not_taken(values: torch.Tensor) -> torch.Tensor:
        """Penalize selecting only one member of an aligned direction pair."""
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
        r"""Represent ambient directions by :math:`J_\kappa^+e_i`.

        :param flat_indices: Flattened position-residue indices.
        :return: Inverse-singular-value-whitened latent directions.
        """
        return self._projection_matrix()[:, flat_indices].T

    def position_vectors(
        self,
        position: int,
        amino_acids: torch.Tensor,
        parent_amino_acid: int,
    ) -> torch.Tensor:
        """Return normalized directions for choices at one position.

        :param position: Zero-based sequence position.
        :param amino_acids: Candidate residue indices.
        :param parent_amino_acid: Parent residue index.
        :return: Normalized candidate directions.
        """
        flat_indices = position * len(self.alphabet) + amino_acids
        vectors = self._raw_directions(flat_indices)
        if self.direction_mode == "diff":
            parent_index = torch.tensor(
                [position * len(self.alphabet) + parent_amino_acid],
                device=amino_acids.device,
            )
            vectors = vectors - self._raw_directions(parent_index)
        return vectors / (torch.linalg.norm(vectors, dim=1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        """Score every non-parent combination with the TANDEM pair rule.

        :param parent_peptide: Sequence defining identity residue choices.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Complete residue tuples mapped to mean pair potentials.
        """
        positions = sorted(mutations)
        if not positions:
            return {}

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


class AmbientMetricPairwiseSimilarityPotential(
    ProjectedDirectionPairwiseSimilarityPotential
):
    r"""TANDEM variant B using the stable ambient pullback projector.

    An ambient mutation :math:`d_i` is represented by
    :math:`U_\kappa^T d_i`, which is equivalent to the normalized form induced
    by :math:`G_\kappa=U_\kappa U_\kappa^T`. Variant A remains recommended;
    this variant supports thesis reproduction and explicit A/B comparisons.
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        taken_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        """Initialize TANDEM with stable ambient pullback geometry.

        :param tangent_space: Local decoder SVD defining the stable subspace.
        :param alphabet: Optional index-to-token mapping.
        :param taken_taken_transform: Optional jointly-selected transform.
        :param taken_not_taken_transform: Optional partially-selected transform.
        :param direction_mode: ``onehot`` or ``diff`` direction encoding.
        :raises ValueError: If ``direction_mode`` is unsupported.
        """
        super().__init__(
            tangent_space=tangent_space,
            alphabet=alphabet,
            taken_taken_transform=taken_taken_transform,
            taken_not_taken_transform=taken_not_taken_transform,
            direction_mode=direction_mode,
        )
        horizontal = torch.abs(tangent_space.S) > tangent_space.horizontal_threshold
        self.horizontal_basis = tangent_space.U[:, horizontal].contiguous()

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        r"""Return :math:`U_\kappa^T e_i` representations.

        :param flat_indices: Flattened position-residue indices.
        :return: Corresponding rows of the stable ambient basis.
        """
        return self.horizontal_basis[flat_indices]


class GeometryFilter(MutationCandidateFilter):
    """Build fresh tangent geometry at the current parent peptide."""

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        horizontal_threshold: float = 0.1,
        maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
        alphabet: list[str] | None = None,
    ):
        """Initialize geometry shared by LAMS and TANDEM.

        :param encoder_decoder: Model providing the decoder Jacobian.
        :param horizontal_threshold: Stable singular-value threshold.
        :param maximum_candidates: Maximum mutation-product size.
        :param alphabet: Optional index-to-token mapping.
        """
        self.encoder_decoder = encoder_decoder
        self.horizontal_threshold = horizontal_threshold
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def pairwise_potential(
        self, parent_peptide: str
    ) -> ProjectedDirectionPairwiseSimilarityPotential:
        """Build the recommended variant-A potential at the parent.

        Variant B can be constructed from the same tangent space, but exposing
        that choice through experiment configuration requires a runner update.

        :param parent_peptide: Sequence at which the Jacobian is evaluated.
        :return: Variant-A pairwise potential backed by current geometry.
        """
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

    def _pairwise_potential(
        self, parent_peptide: str
    ) -> ProjectedDirectionPairwiseSimilarityPotential:
        """Preserve the pre-refactor private geometry-builder API.

        :param parent_peptide: Sequence at which the Jacobian is evaluated.
        :return: Variant-A pairwise potential backed by current geometry.
        """
        return self.pairwise_potential(parent_peptide)
