"""Oracle step contract."""

from abc import ABC

from pep_compass.optimization.step import Step


class Oracle(Step, ABC):
    """Annotate candidates with objective values and return the same batch."""
