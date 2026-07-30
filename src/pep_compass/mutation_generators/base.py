"""Mutation generator step contract."""

from abc import ABC

from pep_compass.optimization.step import Step


class MutationGenerator(Step, ABC):
    """Generate sequence candidates from latent origins and cached geometry."""
