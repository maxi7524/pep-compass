"""Article-notation kappa-stable decoder geometry."""

import torch

from pep_compass.autoencoder.geometry import compute_stable_tangent_geometry
from pep_compass.steps.walkers.strategies.sorbes.geometry.base import GeometryStrategy


class KappaStableGeometry(GeometryStrategy):
    """Compute one reusable decoder SVD and ``S**2 > kappa`` mask."""

    def __init__(self, autoencoder=None, *, kappa: float) -> None:
        from pep_compass.utils.validation import validate_positive

        validate_positive(kappa, "KappaStableGeometry.kappa", allow_zero=True)
        self._configure({"kappa": kappa})
        self.autoencoder = autoencoder
        self.kappa = kappa

    def __call__(self, positions: torch.Tensor):
        return compute_stable_tangent_geometry(
            self.autoencoder, positions, kappa=self.kappa
        )
