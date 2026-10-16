"""Public executable PepCompass pipeline."""

from __future__ import annotations

import torch

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.data_schemas.candidates import (
    Candidate,
    CandidateBatch,
    ObjectField,
    OptionalField,
    SharedField,
    TensorField,
)
from pep_compass.data_schemas.execution import OptimizationLimits
from pep_compass.data_schemas.results import OptimizationResult
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.tracking import NullStepTracker, StepTracker
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

logger = get_custom_logger(__name__)


class PepCompassPipeline:
    """Execute a fully constructed PepCompass computation.

    The pipeline is independent of YAML, experiment planning, persistence and
    CLI concerns. Callers may construct it manually or obtain it from
    :class:`pep_compass.core.builder.PipelineBuilder`.

    :param autoencoder: Autoencoder shared by computational components.
    :param root: Root operation of the executable computation graph.
    :param tracker: Optional detailed step tracker.
    :param limits: Optional run-level computation limits.
    :param stability_monitor: Optional process and accelerator memory monitor.
    """

    def __init__(
        self,
        *,
        autoencoder: Autoencoder,
        root: Step,
        tracker: StepTracker | None = None,
        limits: OptimizationLimits | None = None,
        stability_monitor: StabilityMonitor | NullStabilityMonitor | None = None,
    ) -> None:
        self.autoencoder = autoencoder
        self.root = root
        self.tracker = tracker or NullStepTracker()
        self.limits = limits or OptimizationLimits()
        self.stability_monitor = stability_monitor or NullStabilityMonitor()

    def run(
        self,
        sequences: list[str],
        *,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Encode input sequences and execute the computation graph.

        :param sequences: Non-empty starting peptide sequences.
        :param seed: Optional deterministic run seed.
        :return: Final candidates and optional objective summary.
        :raises ValueError: If no starting sequence is supplied.
        """
        with random_scope(seed):
            return self._run(sequences, seed=seed)

    def _run(
        self,
        sequences: list[str],
        *,
        seed: int | None,
        context=None,
        close_tracker=True,
    ) -> OptimizationResult:
        """Execute one run inside the deterministic random-state boundary."""
        # Sanity check: Empty sequences
        if not sequences:
            raise ValueError("At least one starting sequence is required.")

        # Initial candidate representation
        with torch.no_grad():
            latent_origins = self.autoencoder.encode_peptides(sequences)  # (B, D)
        context = context or OptimizationContext(
            autoencoder=self.autoencoder,
            tracker=self.tracker,
            seed=seed,
            state=OptimizationState(limits=self.limits),
            stability_monitor=self.stability_monitor,
        )
        candidate_ids = torch.as_tensor(
            context.state.next_candidate_ids(len(sequences)),
            dtype=torch.long,
            device=latent_origins.device,
        )  # (B,)
        batch = CandidateBatch(sequences, latent_origins)
        batch = batch.with_field("lineage.candidate_id", TensorField(candidate_ids))
        batch = batch.with_field(
            "lineage.parent_candidate_id",
            TensorField(torch.full_like(candidate_ids, -1)),
        )

        # Computation lifecycle
        try:
            self.stability_monitor.sample("pipeline.input", batch, context.state)
            logger.info("Precomputing PepCompass pipeline steps.")
            self.root.precompute(context)
            logger.info(
                "Executing PepCompass pipeline for %s starting sequences.",
                len(sequences),
            )
            result = self.root(batch, context)
            self.stability_monitor.sample("pipeline.output", result, context.state)
            return self._summarize(result, context.state)
        finally:
            if close_tracker:
                context.tracker.close()

    @staticmethod
    def _summarize(
        batch: CandidateBatch,
        state: OptimizationState,
    ) -> OptimizationResult:
        """Return the complete evaluated archive for the terminal objective."""
        terminal = batch
        archives = {}
        for objective, observations in state.observations.items():
            latents = state.observation_latents.get(objective, {})
            if not observations or not all(
                sequence in latents for sequence in observations
            ):
                continue
            sequences = tuple(observations)
            fields = {
                f"oracle.{objective}.score": TensorField(
                    torch.tensor(
                        list(observations.values()),
                        dtype=torch.float64,
                    )
                ),  # (N,)
                f"oracle.{objective}.direction": SharedField(
                    state.objective_directions.get(objective, "minimize")
                ),
                f"oracle.{objective}.name": SharedField(objective),
            }
            lineage = state.observation_lineage.get(objective, {})
            lineage_names = sorted(
                {name for values in lineage.values() for name in values}
            )
            for name in lineage_names:
                if all(name in lineage.get(sequence, {}) for sequence in sequences):
                    fields[name] = TensorField(
                        torch.tensor(
                            [lineage[sequence][name] for sequence in sequences],
                            dtype=torch.long,
                        )
                    )  # (N,)
            archives[objective] = CandidateBatch(
                sequences,
                torch.stack([latents[sequence] for sequence in sequences]),
                fields,
            )  # latent origins: (N, D)

        # Objective identity is execution state, independent of field iteration order.
        objective = state.last_objective
        if objective in archives:
            batch = archives[objective]
        score_names = [
            name
            for name, value in batch.fields.items()
            if name.startswith("oracle.")
            and name.endswith(".score")
            and isinstance(value, (TensorField, OptionalField))
        ]
        if objective is None and score_names:
            objective = score_names[-1].removeprefix("oracle.").removesuffix(".score")
        prefix = f"oracle.{objective}"
        score_field = batch.fields.get(f"{prefix}.score")
        valid = None
        if isinstance(score_field, OptionalField):
            valid = score_field.valid
            score_field = score_field.values
        if (
            not isinstance(score_field, TensorField)
            or score_field.values.ndim != 1
            or not len(batch)
        ):
            return OptimizationResult(
                candidates=batch,
                terminal_candidates=terminal,
                evaluated_archives=archives,
            )
        scores = score_field.values  # (N,)
        finite = torch.isfinite(scores)  # (N,)
        if valid is not None:
            finite &= valid.to(scores.device)
        if not finite.any():
            return OptimizationResult(
                candidates=batch,
                terminal_candidates=terminal,
                evaluated_archives=archives,
            )
        direction_field = batch.fields.get(f"{prefix}.direction")
        if isinstance(direction_field, OptionalField):
            metadata_indices = direction_field.valid.nonzero(as_tuple=True)[
                0
            ]  # (N_valid,)
            direction_field = direction_field.values
            direction = (
                direction_field.values[int(metadata_indices[0])]
                if isinstance(direction_field, ObjectField) and len(metadata_indices)
                else "minimize"
            )
        else:
            direction = (
                direction_field.value
                if isinstance(direction_field, SharedField)
                else "minimize"
            )
        direction = state.objective_directions.get(objective, direction)
        indices = finite.nonzero(as_tuple=True)[0]  # (N_valid,)
        valid_scores = scores[indices]  # (N_valid,)
        offset = (
            valid_scores.argmax() if direction == "maximize" else valid_scores.argmin()
        )  # ()
        best_index = int(indices[offset])
        return OptimizationResult(
            candidates=batch,
            best_candidate=Candidate(
                batch.sequences[best_index], batch.latent_origins[best_index]
            ),
            best_score=float(scores[best_index]),
            objective_name=objective,
            objective_direction=direction,
            terminal_candidates=terminal,
            evaluated_archives=archives,
        )
