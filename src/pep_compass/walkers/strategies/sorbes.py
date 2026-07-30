"""SORBES adapter for the composable optimization engine."""

from __future__ import annotations

import torch

from pep_compass.optimization.batch import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.context import OptimizationContext
from pep_compass.walkers.base import Walker
from pep_compass.walkers.manager import WalkerManager


@WalkerManager.register("sorbes")
class SorbesWalker(Walker):
    """Apply one existing SORBES step to every latent origin in a batch."""

    def __init__(self, sampling_walker) -> None:
        self.sampling_walker = sampling_walker

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        positions = []
        step_information = []
        for latent_origin in batch.latent_origins:
            position, information = self.sampling_walker.step(latent_origin)
            positions.append(position)
            step_information.append(information)

        latent_origins = torch.stack(positions)
        sequences = context.encoder_decoder.decode_peptides(latent_origins)
        result = CandidateBatch(sequences, latent_origins, batch.fields)
        result = result.with_field(
            "walker.singular_values",
            TensorField(torch.stack([item["S"] for item in step_information])),
        )
        result = result.with_field(
            "walker.left_vectors",
            TensorField(torch.stack([item["U"] for item in step_information])),
        )
        result = result.with_field(
            "walker.adjusted_time_step",
            TensorField(
                torch.as_tensor(
                    [item["adjusted_time_step"] for item in step_information],
                    device=latent_origins.device,
                )
            ),
        )
        return result.with_field(
            "walker.tangent_space",
            ObjectField([item["tangent_space"] for item in step_information]),
        )
