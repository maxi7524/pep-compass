"""Optimization-engine adapter for MUTANG."""

import torch

from pep_compass.data_schemas.candidates import ObjectField, TensorField
from pep_compass.steps.base import Step


class Mutang(Step):
    """Convert the internal MUTANG results into an aligned candidate batch."""

    stage_parameters = ("geometry", "directions", "scoring", "selection", "combination")
    category = "mutation_generator"

    required_resources = ("autoencoder",)
    output_fields = (
        "mutation.parent_sequence",
        "mutation.options",
        "mutation.scored_options",
    )
    invalidated_prefixes = ("oracle.",)
    display_parameters = ("maximum_candidates", "geometry")

    def __init__(
        self,
        *,
        maximum_candidates=None,
        geometry=None,
        directions=None,
        scoring=None,
        selection=None,
        combination=None,
        max_len=25,
        direction_significance_threshold=0.001,
        min_number_of_directions=5,
        token_threshold=0.1,
        alphabet=None,
    ):
        """Configure independent mutation stages without model initialization.

        :param maximum_candidates: Per-parent allocation cap, or None for unbounded generation.
        :param geometry: Shared local geometry by default; pass an independent geometry explicitly.
        :param combination: Explicit combination strategy; its cap must agree with the outer cap.
        """
        from pep_compass.utils.validation import validate_integer

        from .combination.baseline import BaselineCombination
        from .direction_selection.baseline import BaselineDirectionSelection
        from .geometry.shared import SharedGeometry
        from .mutation_selection.threshold import ThresholdSelection
        from .scoring.max_absolute_loading import MaxAbsoluteLoading

        validate_integer(
            maximum_candidates, "Mutang.maximum_candidates", 1, optional=True
        )
        alphabet = (
            list(alphabet) if alphabet is not None else list(" ACDEFGHIKLMNPQRSTVWY")
        )
        self.geometry = geometry if geometry is not None else SharedGeometry()
        self.directions = (
            directions
            if directions is not None
            else BaselineDirectionSelection(
                direction_significance_threshold, min_number_of_directions
            )
        )
        self.scoring = (
            scoring
            if scoring is not None
            else MaxAbsoluteLoading(max_len, len(alphabet))
        )
        self.selection = (
            selection if selection is not None else ThresholdSelection(token_threshold)
        )
        self.combination = (
            combination
            if combination is not None
            else BaselineCombination(alphabet, maximum_candidates)
        )
        cap = getattr(self.combination, "maximum_candidates", None)
        if maximum_candidates is not None and cap != maximum_candidates:
            raise ValueError(
                "Mutang maximum_candidates conflicts with its combination cap."
            )
        self.maximum_candidates = cap
        self._configure(
            dict(
                maximum_candidates=cap,
                geometry=self.geometry,
                directions=self.directions,
                scoring=self.scoring,
                selection=self.selection,
                combination=self.combination,
                max_len=max_len,
                direction_significance_threshold=direction_significance_threshold,
                min_number_of_directions=min_number_of_directions,
                token_threshold=token_threshold,
                alphabet=alphabet,
            )
        )
        self.geometry_requirement = getattr(self.geometry, "requirement", None)
        self.requires_local_enumeration = self.geometry.local_enumeration_only

    def contract(self):
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            ("tangent_geometry", "point_id") if self.requires_local_enumeration else (),
            produced_fields=self.output_fields,
            invalidated_prefixes=self.invalidated_prefixes,
            resources=self.required_resources,
        )

    def candidate_bound(self, count):
        return (
            None if self.maximum_candidates is None else count * self.maximum_candidates
        )

    def _initialize(self, autoencoder):
        from .algorithm import MutationAlgorithm

        self.mutang = MutationAlgorithm(
            self.geometry,
            self.directions,
            self.scoring,
            self.selection,
            self.combination,
            autoencoder,
        )

    def _execute(self, batch, context):
        limit = context.state.limits.generated_candidates
        if limit is not None and context.state.generated_candidates >= limit:
            context.state.stop_requested = True
            return batch
        sequences, parents, parent_indices, options, scored_options = [], [], [], [], []
        for index, (generated, selected) in enumerate(
            self.mutang.generate(batch, context)
        ):
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
        parent_ids = batch.fields.get("lineage.candidate_id")
        if isinstance(parent_ids, TensorField):
            result = result.with_field(
                "lineage.parent_candidate_id",
                TensorField(parent_ids.values.index_select(0, indices)),
            )
        result = result.with_field(
            "lineage.candidate_id",
            TensorField(
                torch.as_tensor(
                    context.state.next_candidate_ids(len(result)),
                    dtype=torch.long,
                    device=result.latent_origins.device,
                )
            ),
        )
        context.state.record_generated_candidates(len(result))
        result = result.with_field("mutation.parent_sequence", ObjectField(parents))
        result = result.with_field("mutation.options", ObjectField(options))
        return result.with_field("mutation.scored_options", ObjectField(scored_options))
