"""Optimization-engine adapter for SORBES."""

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry, TangentDecomposition
from pep_compass.data.optimization import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.components.walkers.base import Walker
from pep_compass.optimization.components.walkers.strategies.subriemannian import SubRiemannianTangentSpace


class SorbesWalker(Walker):
    """Run SORBES and attach output-point geometry for local enumeration."""

    def __init__(self, sorbes) -> None:
        self.sorbes = sorbes

    def _execute(self, batch, context):
        current = self._current_geometry(batch)
        step = self.sorbes.step(batch.latent_origins, current)
        sequences = context.autoencoder.decode_peptides(step.positions)
        fields = dict(batch.fields)
        fields["walker.geometry"] = ObjectField(
            [step.geometry.select(index) for index in range(len(batch))]
        )
        fields["walker.singular_values"] = TensorField(step.geometry.singular_values)
        fields["walker.left_vectors"] = TensorField(step.geometry.left_vectors)
        fields["walker.adjusted_time_step"] = TensorField(step.adjusted_time_steps)
        threshold = step.geometry.kappa**0.5
        fields["walker.tangent_space"] = ObjectField(
            [
                SubRiemannianTangentSpace(
                    step.geometry.left_vectors[index],
                    step.geometry.singular_values[index],
                    step.geometry.right_vectors[index],
                    threshold,
                )
                for index in range(len(batch))
            ]
        )
        return CandidateBatch(sequences, step.positions, fields)

    @staticmethod
    def _current_geometry(batch):
        field = batch.fields.get("walker.geometry")
        if not isinstance(field, ObjectField) or not field.values:
            return None
        if not all(isinstance(value, StableTangentGeometry) for value in field.values):
            return None
        values = field.values
        decomposition = TangentDecomposition(
            torch.cat([value.left_vectors for value in values]),
            torch.cat([value.singular_values for value in values]),
            torch.cat([value.right_vectors for value in values]),
        )
        return StableTangentGeometry(
            decomposition,
            values[0].kappa,
            torch.cat([value.active_mask for value in values]),
        )
