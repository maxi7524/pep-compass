"""Adapter exposing existing POLI black boxes as optimization steps."""

from __future__ import annotations

from dataclasses import dataclass, fields
from importlib import import_module
from typing import ClassVar

import numpy as np
import torch

from pep_compass.data_schemas.candidates import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step


@dataclass
class BlackBoxOracle(Step):
    """Evaluate a complete candidate batch and retain aligned objective scores.

    :param batch_size: Native predictor batch size; None uses that predictor's default.
    :param parallelize: Enable native black-box parallel processing, default False.
    :param num_workers: Optional native worker count.
    :param evaluation_budget: Optional native evaluation budget, reset for each run.
    :param force_isolation: Enable native black-box process isolation.
    :param evaluation_batch_size: Optional adapter microbatch size; all completed results are retained.
    :param model: Named weight variant owned by this model implementation.
    :param models_directory: Optional external directory replacing the packaged model root.
    :param device: Device used by this objective's predictor, independent of the autoencoder.
    """

    display_parameters = ("model", "device", "scale", "evaluation_batch_size")
    category = "oracle"

    batch_size: int | None = None
    parallelize: bool = False
    num_workers: int | None = None
    evaluation_budget: int | float | None = None
    force_isolation: bool = False
    evaluation_batch_size: int | None = None
    implementation: ClassVar[str]
    objective: ClassVar[str]

    def __post_init__(self):
        from pep_compass.utils.validation import validate_integer

        for name in ("batch_size", "num_workers", "evaluation_batch_size"):
            validate_integer(getattr(self, name), "oracle." + name, 1, optional=True)
        for name in ("parallelize", "force_isolation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(name + " must be boolean.")
        if self.evaluation_budget is not None and (
            isinstance(self.evaluation_budget, bool)
            or not np.isfinite(self.evaluation_budget)
            or self.evaluation_budget < 0
        ):
            raise ValueError(
                "Oracle evaluation_budget must be finite and non-negative, or None."
            )
        self._configure(
            {
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.init
            }
        )
        self.field_name = "oracle." + self.objective + ".score"

    def _initialize(self, autoencoder=None):
        module, _, name = self.implementation.rpartition(".")
        values = self.configuration
        values.pop("evaluation_batch_size", None)
        if values.get("evaluation_budget") is None:
            values.pop("evaluation_budget", None)
        self.black_box = getattr(import_module(module), name)(**values)

    def contract(self):
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            produced_fields=tuple(
                "oracle." + self.objective + suffix
                for suffix in (".score", ".direction", ".name")
            )
        )

    def precompute(self, context):
        """Reset per-run evaluation state while retaining expensive model weights."""
        for name in ("reset_evaluation_budget", "clear_cache"):
            reset = getattr(self.black_box, name, None)
            if callable(reset):
                reset()

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        remaining = context.state.remaining_oracle_calls()
        if remaining == 0:
            context.state.stop_requested = True
            return batch
        # Budget boundaries apply before a new evaluation, never inside a batch.
        # Completed observations remain available even if this batch exceeds the threshold.
        evaluated_batch = batch
        if len(evaluated_batch) == 0:
            scores = torch.empty(
                0,
                dtype=torch.get_default_dtype(),
                device=evaluated_batch.latent_origins.device,
            )
            return self._attach_result(evaluated_batch, scores)
        if self.evaluation_batch_size is None:
            raw = self.black_box(np.asarray(evaluated_batch.sequences))
        else:
            values = []
            for start in range(0, len(evaluated_batch), self.evaluation_batch_size):
                values.append(
                    self.black_box(
                        np.asarray(
                            evaluated_batch.sequences[
                                start : start + self.evaluation_batch_size
                            ]
                        )
                    )
                )
            raw = np.concatenate(values, axis=0)
        scores = torch.as_tensor(raw, device=evaluated_batch.latent_origins.device)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        context.state.record_observations(
            self.field_name.removesuffix(".score").removeprefix("oracle."),
            evaluated_batch.sequences,
            [float(score) for score in scores.detach().cpu().tolist()],
            evaluated_batch.latent_origins,
            direction="maximize"
            if getattr(self.black_box, "maximize", False)
            else "minimize",
            lineage={
                name: value.values.detach().cpu().tolist()
                for name, value in evaluated_batch.fields.items()
                if name.startswith("lineage.") and isinstance(value, TensorField)
            },
        )
        return self._attach_result(evaluated_batch, scores)

    def _attach_result(
        self,
        batch: CandidateBatch,
        scores: torch.Tensor,
    ) -> CandidateBatch:
        """Attach score and oracle metadata fields to an evaluated batch."""
        result = batch.with_field(self.field_name, TensorField(scores))
        direction = (
            "maximize" if getattr(self.black_box, "maximize", False) else "minimize"
        )
        prefix = self.field_name.removesuffix(".score")
        result = result.with_field(f"{prefix}.direction", SharedField(direction))
        return result.with_field(
            f"{prefix}.name", SharedField(prefix.removeprefix("oracle."))
        )
