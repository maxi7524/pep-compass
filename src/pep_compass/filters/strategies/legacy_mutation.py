"""Adapters for existing MUTANG candidate filter implementations."""

from __future__ import annotations

from collections import defaultdict

import torch

from pep_compass.filters.base import Filter
from pep_compass.optimization.batch import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.context import OptimizationContext


class LegacyMutationFilter(Filter):
    """Apply an existing mutation filter to candidate groups from one parent."""

    def __init__(self, candidate_filter) -> None:
        self.candidate_filter = candidate_filter

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        parent_sequences = batch.fields.get("mutation.parent_sequence")
        mutation_options = batch.fields.get("mutation.options")
        tangent_spaces = batch.fields.get("walker.tangent_space")
        if not isinstance(parent_sequences, ObjectField) or not isinstance(
            mutation_options, ObjectField
        ):
            raise ValueError(
                "Mutation filters require mutation.parent_sequence and mutation.options."
            )

        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        option_ids: dict[int, int] = {}
        for index, (parent, options) in enumerate(
            zip(parent_sequences.values, mutation_options.values)
        ):
            option_id = option_ids.setdefault(id(options), len(option_ids))
            groups[(parent, option_id)].append(index)

        retained: list[int] = []
        method_scores = torch.full(
            (len(batch),),
            torch.nan,
            dtype=batch.latent_origins.dtype,
            device=batch.latent_origins.device,
        )
        for indices in groups.values():
            first = indices[0]
            tangent_space = None
            if isinstance(tangent_spaces, ObjectField):
                tangent_space = tangent_spaces.values[first]
            accepted = self.candidate_filter.filter_candidates(
                parent_sequences.values[first],
                mutation_options.values[first],
                tangent_space=tangent_space,
            )
            accepted_counts: dict[str, int] = defaultdict(int)
            for sequence in accepted:
                accepted_counts[sequence] += 1
            for index in indices:
                sequence = batch.sequences[index]
                if accepted_counts[sequence] > 0:
                    retained.append(index)
                    accepted_counts[sequence] -= 1

            trace = getattr(self.candidate_filter, "last_filter_trace", None)
            bounded = getattr(self.candidate_filter, "last_bounded_candidates", [])
            if trace is not None and trace.scores is not None:
                score_by_sequence = {
                    sequence: float(trace.scores[position])
                    for position, sequence in enumerate(bounded)
                }
                for index in indices:
                    score = score_by_sequence.get(batch.sequences[index])
                    if score is not None:
                        method_scores[index] = score

        index_tensor = torch.as_tensor(
            retained,
            dtype=torch.long,
            device=batch.latent_origins.device,
        )
        result = batch.with_field(
            f"filter.{self.candidate_filter.__class__.__name__}.score",
            TensorField(method_scores),
        )
        return result.select(index_tensor)
