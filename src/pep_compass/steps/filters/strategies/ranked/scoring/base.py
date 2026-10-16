"""Base contract for reusable candidate scoring."""

from abc import ABC, abstractmethod

import torch

from pep_compass.configuration.objects import Configurable


class ScoreFunction(Configurable, ABC):
    """Assign one scalar score to every candidate in a batch."""

    higher_is_better: bool = True

    def contract(self):
        """Declare optional resources and fields required by this scoring implementation."""
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            resources=("autoencoder",) if hasattr(self, "autoencoder") else ()
        )

    @abstractmethod
    def __call__(self, batch, context) -> torch.Tensor:
        """Return scores with shape ``(B,)``."""
