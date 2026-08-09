"""Geometry derived from an autoencoder decoder Jacobian."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pep_compass.autoencoder.base import Autoencoder


@dataclass(frozen=True, slots=True)
class TangentDecomposition:
    """Store a batched compact SVD of decoder Jacobians.

    :param left_vectors: Ambient-space singular vectors ``(B, A, K)``.
    :param singular_values: Singular values ``(B, K)``.
    :param right_vectors: Latent-space right singular vectors ``(B, K, D)``.
    """

    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors: torch.Tensor


def compute_tangent_decomposition(
    autoencoder: Autoencoder,
    latent_positions: torch.Tensor,
) -> TangentDecomposition:
    """Compute compact decoder-Jacobian SVDs for latent positions.

    :param autoencoder: Autoencoder providing batched decoder Jacobians.
    :param latent_positions: One latent vector ``(D,)`` or batch ``(B, D)``.
    :return: Named batched tangent decomposition.
    """
    if latent_positions.ndim == 1:
        latent_positions = latent_positions.unsqueeze(0)  # (1, D)
    decoder_jacobians = autoencoder.decoder_jacobian(latent_positions)  # (B, A, D)
    left, singular, right = torch.linalg.svd(
        decoder_jacobians,
        full_matrices=False,
    )
    return TangentDecomposition(left, singular, right)
