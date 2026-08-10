"""Tests for trajectory-isolated local enumeration."""

import torch

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.operations import Flow, LocalEnumeration
from tests.fixtures.autoencoders import MockAutoencoder
from tests.fixtures.components import SuffixStep


class RecordingWalker(SuffixStep):
    """Advance one-row trajectories and record input cardinality."""

    def __init__(self) -> None:
        super().__init__("W")
        self.input_sizes: list[int] = []

    def _execute(self, batch, context):
        self.input_sizes.append(len(batch))
        return super()._execute(batch, context)


class PairGenerator(SuffixStep):
    """Emit two local candidates for every trajectory point."""

    def __init__(self) -> None:
        super().__init__("")

    def _execute(self, batch, context):
        indices = torch.arange(len(batch), device=batch.latent_origins.device).repeat_interleave(2)
        expanded = batch.repeat_from_parents(indices)
        return expanded.with_sequences(
            [f"{sequence}{suffix}" for sequence in batch.sequences for suffix in ("A", "B")]
        )


def test_local_enumeration_does_not_feed_mutations_back_to_walker() -> None:
    """Every SORBES call must receive only its private trajectory point."""
    walker = RecordingWalker()
    operation = LocalEnumeration(
        walker=walker,
        mutation_generator=PairGenerator(),
        filters=Flow([]),
        trajectories=2,
        iterations=2,
        walk_time=None,
    )
    batch = CandidateBatch(["S"], torch.zeros((1, 2)))  # (B=1, D=2)

    result = operation(batch, OptimizationContext(autoencoder=MockAutoencoder()))

    assert walker.input_sizes == [1, 1, 1, 1]
    # Two trajectories, two SORBES points, and three emissions per point:
    # the walk point plus two locally generated candidates.
    assert len(result) == 12
    assert result.sequences.count("SWA") == 2
    assert result.sequences.count("SWWB") == 2
