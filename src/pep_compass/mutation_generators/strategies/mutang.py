"""MUTANG adapter preserving the latent origin of every generated sequence."""

from __future__ import annotations

import torch

from pep_compass.mutation_generators.base import MutationGenerator
from pep_compass.mutation_generators.manager import MutationGeneratorManager
from pep_compass.optimization.batch import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.context import OptimizationContext


@MutationGeneratorManager.register("mutang")
class MutangGenerator(MutationGenerator):
    """Generate the existing MUTANG Cartesian product for every candidate."""

    def __init__(self, mutation_enumerator) -> None:
        self.mutation_enumerator = mutation_enumerator

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        singular_values = batch.fields.get("walker.singular_values")
        left_vectors = batch.fields.get("walker.left_vectors")
        if not isinstance(singular_values, TensorField) or not isinstance(
            left_vectors, TensorField
        ):
            raise ValueError(
                "MUTANG requires walker.singular_values and walker.left_vectors fields."
            )

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
            generated = list(
                dict.fromkeys(
                    self.mutation_enumerator.mutate_peptide(sequence, mutations)
                )
            )
            sequences.extend(generated)
            parent_indices.extend([index] * len(generated))
            mutation_options.extend([mutations] * len(generated))
            parent_sequences.extend([sequence] * len(generated))

        indices = torch.as_tensor(
            parent_indices,
            dtype=torch.long,
            device=batch.latent_origins.device,
        )
        expanded = batch.repeat_from_parents(indices)
        result = expanded.with_sequences(sequences)
        result = result.with_field(
            "mutation.parent_sequence",
            ObjectField(parent_sequences),
        )
        return result.with_field(
            "mutation.options",
            ObjectField(mutation_options),
        )
