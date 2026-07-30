"""Explicit candidate deduplication filter."""

from __future__ import annotations

import torch

from pep_compass.filters.base import Selector
from pep_compass.filters.manager import FilterManager
from pep_compass.optimization.batch import CandidateBatch
from pep_compass.optimization.context import OptimizationContext


@FilterManager.register("deduplicate")
class DeduplicateFilter(Selector):
    """Keep the first candidate for every configured identity key."""

    def __init__(self, key: str = "sequence") -> None:
        if key not in {"sequence", "sequence_and_latent"}:
            raise ValueError(
                "Deduplication key must be sequence or sequence_and_latent."
            )
        self.key = key

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        seen: set[object] = set()
        retained: list[int] = []
        latent_rows = batch.latent_origins.detach().cpu()
        for index, sequence in enumerate(batch.sequences):
            identity: object = sequence
            if self.key == "sequence_and_latent":
                identity = (sequence, latent_rows[index].numpy().tobytes())
            if identity not in seen:
                seen.add(identity)
                retained.append(index)
        indices = torch.as_tensor(
            retained,
            dtype=torch.long,
            device=batch.latent_origins.device,
        )
        return batch.select(indices)
