"""Walker step contract."""

from abc import ABC

from pep_compass.optimization.step import Step


class Walker(Step, ABC):
    """Move candidates through latent space while retaining trajectory state."""
