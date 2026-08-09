"""Tests for shared optimization data contracts."""

import torch

from pep_compass.data.optimization import TensorField
from fixtures.candidates import candidate_batch


def test_candidate_selection_preserves_every_aligned_column() -> None:
    """Selecting a row must select sequences, latents and fields together."""
    selected = candidate_batch().select(torch.tensor([1]))

    assert selected.sequences == ("BB",)
    assert selected.latent_origins.tolist() == [[2.0, 3.0]]
    assert isinstance(selected.fields["score"], TensorField)
    assert selected.fields["score"].values.tolist() == [0.75]
