from abc import ABC, abstractmethod
import torch
from typing import Literal

from .utils import decoder_jacobian_approx, decoder_jacobian_strict


class EncoderDecoder(ABC):

    def __init__(
        self,
        *,
        jacobian_mode: Literal["strict", "approx"] = "strict",
        jacobian_eps: float,
        field_eps: float,
    ):
        self.jacobian_mode = jacobian_mode
        self.jacobian_eps = jacobian_eps
        self.field_eps = field_eps
        assert self.jacobian_mode in ["strict", "approx"], ValueError(
            f"Unknown jacobian mode: {self.jacobian_mode}"
        )

    @abstractmethod
    def decoder_forward(self, x):
        pass

    @abstractmethod
    def encoder_forward(self, x):
        pass

    @property
    @abstractmethod
    def latent_dim(self):
        pass

    @property
    @abstractmethod
    def ambient_dim(self):
        pass

    def decoder_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        r"""input shape: (batch_dim, latent_dim), output shape: (batch_dim, ambient_dim, latent_dim)"""
        onedim = x.ndim == 1
        if onedim:
            x = x.unsqueeze(0)
        # assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
        if self.jacobian_mode == "strict":
            jac =  decoder_jacobian_strict(self.decoder_forward, x)
        elif self.jacobian_mode == "approx":
            jac = decoder_jacobian_approx(self.decoder_forward, x, self.jacobian_eps)
        if onedim:
            jac = jac.squeeze(0)
        return jac

    def field_derivative(self, latent_point, direction, eps=0.1, ambient_point=None):
        if ambient_point is None:
            center = self.decoder_forward(latent_point)
        else:
            center = ambient_point
        right = self.decoder_forward(latent_point + self.field_eps * direction)
        left = self.decoder_forward(latent_point - self.field_eps * direction)
        return (right + left - 2 * center) / eps**2

    def get_ambient_covariant_derivative(self, latent_point, vector1, vector2, eps=0.1):
        ambient_covariant_derivative = (
            self.decoder_forward(latent_point + eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point - eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point + eps * vector1 - eps * vector2)
            + self.decoder_forward(latent_point - eps * vector1 - eps * vector2)
        ) / (4 * eps**2)

        return ambient_covariant_derivative
