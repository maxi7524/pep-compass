from abc import ABC, abstractmethod
import torch
from typing import Literal
from einops import rearrange, repeat


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
        assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
        if self.jacobian_mode == "strict":
            return self._decoder_jacobian_strict(x)
        elif self.jacobian_mode == "approx":
            return self._decoder_jacobian_approx(x)

    def _decoder_jacobian_strict(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            torch.autograd.functional.jacobian(self.decoder_forward, x)[0],
            "a b d -> b a d",
        )

    def _decoder_jacobian_approx(self, x):
        decoder_input_delta = torch.cat(
            [
                torch.eye(self.latent_dim, device=x.device) * self.jacobian_eps,
                torch.zeros((1, self.latent_dim), device=x.device),
            ],
            dim=0,
        )
        decoder_input = rearrange(
            rearrange(x, "b d -> b 1 d")
            + rearrange(decoder_input_delta, "d_plus_1 d -> 1 d_plus_1 d"),
            "b d_plus_1 d -> (b d_plus_1) d",
        )
        with torch.no_grad():
            decoder_output = self.decoder_forward(decoder_input)

        decoder_output = rearrange(
            decoder_output, "(b d_plus_1) d -> b d_plus_1 d", b=x.shape[0]
        )
        approx_jac = (
            decoder_output[:, :-1, :] - decoder_output[:, [-1], :]
        ) / self.jacobian_eps

        return rearrange(approx_jac, "b d a -> b a d")

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
