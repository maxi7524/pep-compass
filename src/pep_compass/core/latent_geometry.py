"""Shared latent-space geometry computations."""

from __future__ import annotations

import torch

from pep_compass.encoder_decoder.base import EncoderDecoder


def compute_tangent_space_svd(
    encoder_decoder: EncoderDecoder,
    latent_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute decoder-Jacobian SVDs for a batch of latent positions.

    :param encoder_decoder: Model providing batched decoder Jacobians.
    :param latent_positions: One latent vector ``(D,)`` or a batch ``(B, D)``.
    :return: Left singular vectors, singular values and right singular vectors.
    """
    if latent_positions.ndim == 1:
        latent_positions = latent_positions.unsqueeze(0)  # (1, D)
    decoder_jacobians = encoder_decoder.decoder_jacobian(
        latent_positions
    )  # (B, A, D)
    return torch.linalg.svd(decoder_jacobians, full_matrices=False)
