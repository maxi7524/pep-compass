from abc import ABC, abstractmethod

import torch

class EncoderDecoder(ABC):

    def __init__(self, jacobian_mode="strict", jacobian_eps=0.1, field_eps=0.1):
        self.jacobian_mode = jacobian_mode
        self.jacobian_eps = jacobian_eps
        self.field_eps = field_eps

    @abstractmethod
    def decoder_forward(self, x):
        pass

    @abstractmethod
    def encoder_forward(self, x):
        pass

    @abstractmethod
    def get_latent_dim(self):
        pass

    @abstractmethod
    def get_ambient_dim(self):
        pass

    def decoder_jacobian(self, x):
        if self.jacobian_mode == "strict":
            return self._decoder_jacobian_strict(x)
        elif self.jacobian_mode == "approx":
            return self._decoder_jacobian_approx(x)
        else:
            raise ValueError(f"Unknown jacobian mode: {self.jacobian_mode}")

    def _decoder_jacobian_strict(self, x):
        return torch.autograd.functional.jacobian(self.decoder_forward, x)[0]

    def _decoder_jacobian_approx(self, x):
        decoder_input = (
            torch.eye(self.get_latent_dim(), device=x.device) * self.jacobian_eps
        )
        decoder_input = torch.cat(
            [decoder_input, torch.zeros_like(x).unsqueeze(0)], dim=0
        )
        decoder_input = decoder_input + x

        with torch.no_grad():
            decoder_output = self.decoder_forward(decoder_input)

        approx_jac = (
            decoder_output[:-1, :] - decoder_output[-1, :]
        ) / self.jacobian_eps

        # TODO: remove this unsqueeze
        return approx_jac.T

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
