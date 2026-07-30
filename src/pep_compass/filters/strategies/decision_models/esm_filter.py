"""ESM sequence-plausibility decision filter."""

from __future__ import annotations

import torch

from pep_compass.filters.base import Filter
from pep_compass.filters.manager import FilterManager
from pep_compass.filters.strategies.decision_models.esm import ESM2PPLScorer
from pep_compass.optimization.batch import CandidateBatch, TensorField
from pep_compass.optimization.context import OptimizationContext


@FilterManager.register("esm_plausibility")
class ESMDecisionFilter(Filter):
    """Select sequences by ESM pseudo-log-likelihood without changing latents."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        top_k: int | None = None,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cpu",
        scorer=None,
    ) -> None:
        if (threshold is None) == (top_k is None):
            raise ValueError("ESM filter requires exactly one of threshold or top_k.")
        if top_k is not None and top_k < 1:
            raise ValueError("ESM filter top_k must be positive.")
        self.threshold = threshold
        self.top_k = top_k
        self.scorer = scorer or ESM2PPLScorer(model_name=model_name, device=device)

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        scores = torch.as_tensor(
            [self.scorer.pll(sequence) for sequence in batch.sequences],
            dtype=batch.latent_origins.dtype,
            device=batch.latent_origins.device,
        )
        scored = batch.with_field("filter.esm_plausibility.score", TensorField(scores))
        if self.threshold is not None:
            indices = torch.nonzero(scores >= self.threshold, as_tuple=False).flatten()
        else:
            count = min(self.top_k or 0, len(batch))
            indices = torch.argsort(scores, descending=True, stable=True)[:count]
        return scored.select(indices)
