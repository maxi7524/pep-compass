"""Position-update contract for SORBES."""

from abc import ABC, abstractmethod

from pep_compass.configuration.objects import Configurable


class PositionUpdateStrategy(Configurable, ABC):
    """Calculate the bounded displacement from one SORBES point."""

    @abstractmethod
    def __call__(self, positions, directions, acceleration):
        """Return ``(new_positions, effective_time_steps)``."""
