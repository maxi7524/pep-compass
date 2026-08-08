"""MUTANG adapter preserving the latent origin of every generated sequence."""

from __future__ import annotations

import math

import torch

from pep_compass.core.latent_geometry import compute_tangent_space_svd
from pep_compass.mutation_generators.base import MutationGenerator
from pep_compass.mutation_generators.manager import MutationGeneratorManager
from pep_compass.optimization.batch import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.context import OptimizationContext
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@MutationGeneratorManager.register("mutang")
class MutangGenerator(MutationGenerator):
    """Generate the existing MUTANG Cartesian product for every candidate."""

    def __init__(
        self,
        mutation_enumerator,
        maximum_candidates: int | None = None,
        encoder_decoder=None,
    ) -> None:
        if maximum_candidates is not None and maximum_candidates < 1:
            raise ValueError("MUTANG maximum_candidates must be positive.")
        self.mutation_enumerator = mutation_enumerator
        self.maximum_candidates = maximum_candidates
        self.encoder_decoder = encoder_decoder

    def _ensure_tangent_geometry(self, batch: CandidateBatch) -> CandidateBatch:
        """Return a batch with the decoder-Jacobian SVD required by MUTANG.

        Cached walker fields take precedence. If either field is absent, both
        are recomputed so that they describe the same Jacobian decomposition.
        """
        singular_values = batch.fields.get("walker.singular_values")
        left_vectors = batch.fields.get("walker.left_vectors")
        if isinstance(singular_values, TensorField) and isinstance(
            left_vectors, TensorField
        ):
            return batch
        if self.encoder_decoder is None:
            raise ValueError(
                "MUTANG requires cached tangent geometry or an encoder-decoder "
                "service capable of computing it."
            )

        logger.info(
            "Computing tangent geometry for %s MUTANG candidates.",
            len(batch),
        )
        left_vector_values, singular_value_values, _ = compute_tangent_space_svd(
            self.encoder_decoder,
            batch.latent_origins,
        )
        return batch.with_field(
            "walker.singular_values",
            TensorField(singular_value_values),
        ).with_field(
            "walker.left_vectors",
            TensorField(left_vector_values),
        )

    def _bounded_mutations(
        self,
        sequence: str,
        mutations: dict[int, list[int]],
        context: OptimizationContext,
    ) -> dict[int, list[int]]:
        """Reduce mutation choices before materializing their Cartesian product.

        Parent residues remain available at every retained position. Random
        removals use the run-scoped generator to preserve reproducibility.
        """
        if self.maximum_candidates is None:
            return mutations
        choices = {
            position: sorted(
                set(residues)
                | {self.mutation_enumerator.alphabet.index(sequence[position])}
            )
            for position, residues in mutations.items()
            if position < len(sequence)
        }
        while (
            choices
            and math.prod(map(len, choices.values())) > self.maximum_candidates
        ):
            position = max(choices, key=lambda key: len(choices[key]))
            parent_residue = self.mutation_enumerator.alphabet.index(
                sequence[position]
            )
            alternatives = [
                residue
                for residue in choices[position]
                if residue != parent_residue
            ]
            if alternatives:
                removed = int(context.rng.choice(alternatives))
                choices[position].remove(removed)
            elif len(choices) > 1:
                del choices[position]
            else:
                break
        return choices

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Generate candidates and repeat every parent-aligned batch field."""
        # Tangent geometry is shared by all products of the same parent.
        batch = self._ensure_tangent_geometry(batch)
        singular_values = batch.fields.get("walker.singular_values")
        left_vectors = batch.fields.get("walker.left_vectors")
        assert isinstance(singular_values, TensorField)
        assert isinstance(left_vectors, TensorField)

        # Candidate expansion
        sequences: list[str] = []
        parent_indices: list[int] = []
        mutation_options: list[dict[int, list[int]]] = []
        parent_sequences: list[str] = []
        for index, sequence in enumerate(batch.sequences):
            mutations = self.mutation_enumerator.get_mutations_from_s_u(
                singular_values.values[index],
                left_vectors.values[index],
            )
            mutations = {
                position: residues
                for position, residues in mutations.items()
                if position < len(sequence)
            }
            mutations = self._bounded_mutations(sequence, mutations, context)
            generated = list(
                dict.fromkeys(
                    self.mutation_enumerator.mutate_peptide(sequence, mutations)
                )
            )
            sequences.extend(generated)
            parent_indices.extend([index] * len(generated))
            mutation_options.extend([mutations] * len(generated))
            parent_sequences.extend([sequence] * len(generated))

        # Parent-field propagation
        indices = torch.as_tensor(
            parent_indices,
            dtype=torch.long,
            device=batch.latent_origins.device,
        )
        expanded = batch.repeat_from_parents(indices)
        result = expanded.with_sequences(sequences)
        context.state.record_generated_candidates(len(result))
        result = result.with_field(
            "mutation.parent_sequence",
            ObjectField(parent_sequences),
        )
        return result.with_field(
            "mutation.options",
            ObjectField(mutation_options),
        )
