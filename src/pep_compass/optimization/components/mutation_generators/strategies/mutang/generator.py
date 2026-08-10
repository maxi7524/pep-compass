"""Optimization-engine adapter for MUTANG."""

import torch

from pep_compass.data.optimization import CandidateBatch, ObjectField
from pep_compass.optimization.components.mutation_generators.base import MutationGenerator


class MutangGenerator(MutationGenerator):
    """Convert the internal MUTANG results into an aligned candidate batch."""

    def __init__(self, mutang):
        self.mutang = mutang
        self.geometry_requirement = getattr(mutang.geometry, "requirement", None)
        self.requires_local_enumeration = mutang.geometry.local_enumeration_only

    def _execute(self, batch, context):
        sequences, parents, parent_indices, options, scored_options = [], [], [], [], []
        for index, (generated, selected) in enumerate(self.mutang.generate(batch, context)):
            sequences.extend(generated)
            parents.extend([batch.sequences[index]] * len(generated))
            parent_indices.extend([index] * len(generated))
            grouped = {}
            for option in selected:
                grouped.setdefault(option.position, []).append(option.residue)
            options.extend([grouped] * len(generated))
            scored_options.extend([selected] * len(generated))
        indices = torch.as_tensor(parent_indices, device=batch.latent_origins.device)
        result = batch.repeat_from_parents(indices).with_sequences(sequences)
        context.state.record_generated_candidates(len(result))
        result = result.with_field("mutation.parent_sequence", ObjectField(parents))
        result = result.with_field("mutation.options", ObjectField(options))
        return result.with_field("mutation.scored_options", ObjectField(scored_options))
