"""Execution entry point for a configured optimization tree."""

from __future__ import annotations

import numpy as np
import torch

from pep_compass.optimization.batch import (
    Candidate,
    CandidateBatch,
    SharedField,
    TensorField,
)
from pep_compass.optimization.context import OptimizationContext
from pep_compass.optimization.result import OptimizationResult
from pep_compass.optimization.state import OptimizationLimits, OptimizationState
from pep_compass.optimization.step import Step
from pep_compass.optimization.tracking import NullStepTracker, StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class OptimizationRunner:
    """Encode starting sequences once and execute the configured root step."""

    def __init__(
        self,
        encoder_decoder,
        root_step: Step,
        tracker: StepTracker | None = None,
        limits: OptimizationLimits | None = None,
    ) -> None:
        self.encoder_decoder = encoder_decoder
        self.root_step = root_step
        self.tracker = tracker or NullStepTracker()
        self.limits = limits or OptimizationLimits()

    def run(
        self,
        sequences: list[str],
        *,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run the configured optimization tree.

        :param sequences: Required starting peptide sequences.
        :type sequences: list[str]
        :param seed: Optional deterministic random seed.
        :type seed: int | None
        :return: Final candidate batch. Objective summary fields remain ``None``
            when the configured steps do not produce an oracle result.
        :rtype: OptimizationResult
        :raises ValueError: If no starting sequence is supplied.
        """
        if not sequences:
            raise ValueError("At least one starting sequence is required.")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        with torch.no_grad():
            latent_origins = self.encoder_decoder.encode_peptides(sequences)
        batch = CandidateBatch(sequences, latent_origins)
        context = OptimizationContext(
            encoder_decoder=self.encoder_decoder,
            tracker=self.tracker,
            seed=seed,
            rng=np.random.default_rng(seed),
            state=OptimizationState(limits=self.limits),
        )
        try:
            logger.info("Precomputing configured optimization steps.")
            self.root_step.precompute(context)
            logger.info(
                "Executing optimization for %s starting sequences.", len(sequences)
            )
            result = self.root_step(batch, context)
            return self._summarize(result)
        finally:
            self.tracker.close()

    @staticmethod
    def _summarize(batch: CandidateBatch) -> OptimizationResult:
        score_names = [
            name
            for name, value in batch.fields.items()
            if name.startswith("oracle.")
            and name.endswith(".score")
            and isinstance(value, TensorField)
        ]
        if not score_names or len(batch) == 0:
            return OptimizationResult(candidates=batch)
        score_name = score_names[-1]
        prefix = score_name.removesuffix(".score")
        score_field = batch.fields[score_name]
        direction_field = batch.fields.get(f"{prefix}.direction")
        name_field = batch.fields.get(f"{prefix}.name")
        direction = (
            direction_field.value
            if isinstance(direction_field, SharedField)
            else "minimize"
        )
        scores = score_field.values
        if scores.ndim != 1:
            return OptimizationResult(candidates=batch)
        best_index = int(
            scores.argmax().item()
            if direction == "maximize"
            else scores.argmin().item()
        )
        return OptimizationResult(
            candidates=batch,
            best_candidate=Candidate(
                batch.sequences[best_index],
                batch.latent_origins[best_index],
            ),
            best_score=float(scores[best_index].item()),
            objective_name=(
                str(name_field.value)
                if isinstance(name_field, SharedField)
                else prefix.removeprefix("oracle.")
            ),
            objective_direction=direction,
        )
