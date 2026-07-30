"""Filter and selector step contracts."""

from abc import ABC

from pep_compass.optimization.step import Step


class Filter(Step, ABC):
    """Retain or annotate candidates according to a configured policy."""


class Selector(Filter, ABC):
    """Filter that selects a subset using batch-level information."""
