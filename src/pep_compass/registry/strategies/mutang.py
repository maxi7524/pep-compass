"""Registries for interchangeable stages of the fixed MUTANG algorithm."""

from pep_compass.optimization.components.mutation_generators.strategies.mutang.base import (
    CombinationStrategy,
    DirectionStrategy,
    GeometryStrategy,
    MutationSelectionStrategy,
    ScoringStrategy,
)
from pep_compass.registry.factories.registry import Registry

geometry_registry: Registry[GeometryStrategy] = Registry(
    "MUTANG geometry", expected_type=GeometryStrategy
)
direction_selection_registry: Registry[DirectionStrategy] = Registry(
    "MUTANG direction selection", expected_type=DirectionStrategy
)
scoring_registry: Registry[ScoringStrategy] = Registry(
    "MUTANG scoring", expected_type=ScoringStrategy
)
mutation_selection_registry: Registry[MutationSelectionStrategy] = Registry(
    "MUTANG mutation selection", expected_type=MutationSelectionStrategy
)
combination_registry: Registry[CombinationStrategy] = Registry(
    "MUTANG combination", expected_type=CombinationStrategy
)
