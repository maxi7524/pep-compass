"""Direction-scaling contract for SORBES."""

from abc import ABC, abstractmethod

from pep_compass.configuration.objects import Configurable


class ScalingStrategy(Configurable, ABC):
    """Scale sampled directions without resampling them."""

    @abstractmethod
    def __call__(self, directions, geometry):
        """Return scaled active and inactive directions."""
