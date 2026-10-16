"""Direction-stage contract for SORBES."""

from abc import ABC, abstractmethod

from pep_compass.autoencoder.geometry import StableTangentGeometry
from pep_compass.configuration.objects import Configurable
from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections


class DirectionStrategy(Configurable, ABC):
    """Generate active and inactive directions from fixed point geometry."""

    @abstractmethod
    def __call__(self, geometry: StableTangentGeometry) -> SorbesDirections:
        """Return directions without modifying the shared geometry."""
