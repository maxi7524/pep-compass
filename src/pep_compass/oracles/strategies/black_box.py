"""Adapter exposing existing POLI black boxes as optimization steps."""

from __future__ import annotations

import numpy as np
import torch

from pep_compass.optimization.batch import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.context import OptimizationContext
from pep_compass.oracles.base import Oracle


class BlackBoxOracle(Oracle):
    """Evaluate sequences with an existing black box and attach its scores."""

    def __init__(
        self, black_box, *, field_name: str, batch_size: int | None = None
    ) -> None:
        self.black_box = black_box
        self.field_name = field_name
        self.batch_size = batch_size

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        if self.batch_size is None:
            raw = self.black_box(np.asarray(batch.sequences))
        else:
            values = []
            for start in range(0, len(batch), self.batch_size):
                values.append(
                    self.black_box(
                        np.asarray(batch.sequences[start : start + self.batch_size])
                    )
                )
            raw = np.concatenate(values, axis=0)
        scores = torch.as_tensor(raw, device=batch.latent_origins.device)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        result = batch.with_field(self.field_name, TensorField(scores))
        direction = (
            "maximize" if getattr(self.black_box, "maximize", False) else "minimize"
        )
        prefix = self.field_name.removesuffix(".score")
        result = result.with_field(f"{prefix}.direction", SharedField(direction))
        return result.with_field(
            f"{prefix}.name", SharedField(prefix.removeprefix("oracle."))
        )
