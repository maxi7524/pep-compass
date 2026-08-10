"""Adapters for scores defined over candidates from one MUTANG parent."""

from collections import defaultdict

import torch

from pep_compass.data.optimization import ObjectField
from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction


class MutationPoolScore(ScoreFunction):
    """Map a mutation-pool scoring implementation onto an aligned batch."""

    def __call__(self, batch, context):
        del context
        parents = batch.fields.get("mutation.parent_sequence")
        options = batch.fields.get("mutation.options")
        tangent_spaces = batch.fields.get("walker.tangent_space")
        if not isinstance(parents, ObjectField) or not isinstance(options, ObjectField):
            raise ValueError("Latent-geometry scoring requires MUTANG parent metadata.")
        groups = defaultdict(list)
        for index, (parent, mutation_map) in enumerate(zip(parents.values, options.values)):
            groups[(parent, id(mutation_map))].append(index)
        # Parent identity is emitted by MUTANG but is intentionally absent from
        # scored mutant distributions. Negative infinity keeps it outside every
        # ranked selection without introducing a second structural filter.
        scores = torch.full(
            (len(batch),), float("-inf"), device=batch.latent_origins.device,
            dtype=batch.latent_origins.dtype,
        )
        for indices in groups.values():
            first = indices[0]
            tangent = tangent_spaces.values[first] if isinstance(tangent_spaces, ObjectField) else None
            score_by_sequence = self.score_group(
                parents.values[first], options.values[first], tangent
            )
            for index in indices:
                if batch.sequences[index] in score_by_sequence:
                    scores[index] = score_by_sequence[batch.sequences[index]]
        return scores

    def score_group(self, parent, mutations, tangent_space):
        """Return complete candidate sequences mapped to scalar scores."""
        raise NotImplementedError
