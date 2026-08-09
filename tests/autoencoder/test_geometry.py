"""Tests for geometry derived from decoder Jacobians."""

import torch

from pep_compass.autoencoder.geometry import compute_tangent_decomposition
from fixtures.autoencoders import MockAutoencoder


def test_tangent_decomposition_batches_single_latent_position() -> None:
    """A one-dimensional latent position must produce a one-row decomposition."""
    result = compute_tangent_decomposition(
        MockAutoencoder(),
        torch.tensor([1.0, 3.0]),  # (D=2,)
    )

    assert result.left_vectors.shape == (1, 2, 2)
    assert result.singular_values.shape == (1, 2)
    assert result.right_vectors.shape == (1, 2, 2)
    assert result.singular_values.tolist() == [[4.0, 2.0]]
