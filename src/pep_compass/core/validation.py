"""Semantic validation of neutral PepCompass pipeline specifications."""

from __future__ import annotations

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)


def validate_pipeline_specification(specification: PipelineSpecification) -> None:
    """Validate a complete pipeline declaration before object construction.

    :param specification: Neutral pipeline declaration.
    :raises ValueError: If limits, component declarations or graph nodes are invalid.
    """
    limits = specification.limits
    for name, value in (
        ("oracle_calls", limits.oracle_calls),
        ("generated_candidates", limits.generated_candidates),
    ):
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError(f"Pipeline limit {name} must be null or non-negative.")
    _validate_step(specification.root, "pipeline.root")


def validate_registered_components(specification: PipelineSpecification) -> None:
    """Validate registered methods and parameters without constructing models."""
    import pep_compass.optimization.components.filters.strategies  # noqa: F401
    import pep_compass.optimization.components.mutation_generators.strategies  # noqa: F401
    import pep_compass.optimization.components.oracles.strategies  # noqa: F401
    import pep_compass.optimization.components.walkers.strategies  # noqa: F401

    _validate_registered_step(specification.root)


def _validate_registered_step(specification: StepSpecification) -> None:
    """Recursively validate component registry contracts."""
    if isinstance(specification, ComponentSpecification):
        from pep_compass.optimization.components.filters import FilterManager
        from pep_compass.optimization.components.mutation_generators import MutationGeneratorManager
        from pep_compass.optimization.components.oracles import OracleManager
        from pep_compass.optimization.components.walkers import WalkerManager

        managers = {
            "filter": FilterManager,
            "mutation_generator": MutationGeneratorManager,
            "oracle": OracleManager,
            "walker": WalkerManager,
        }
        managers[specification.kind].validate(
            specification.method,
            specification.parameters,
        )
        return
    if isinstance(specification, FlowSpecification):
        for step in specification.steps:
            _validate_registered_step(step)
        return
    if isinstance(specification, LoopSpecification):
        _validate_registered_step(specification.body)
        return
    if isinstance(specification, ParallelSpecification):
        for branch in specification.branches:
            _validate_registered_step(branch.body)
        return
    if isinstance(specification, LocalEnumerationSpecification):
        _validate_registered_step(specification.walker)
        _validate_registered_step(specification.generator)
        _validate_registered_step(specification.filters)
        return
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _validate_step(specification: StepSpecification, path: str) -> None:
    """Validate one node and recursively validate its children."""
    if isinstance(specification, ComponentSpecification):
        if not specification.method:
            raise ValueError(f"{path}.method cannot be empty.")
        return
    if isinstance(specification, FlowSpecification):
        for index, step in enumerate(specification.steps):
            _validate_step(step, f"{path}.steps.{index}")
        return
    if isinstance(specification, LoopSpecification):
        if specification.iterations < 0:
            raise ValueError(f"{path}.iterations must be non-negative.")
        _validate_step(specification.body, f"{path}.body")
        return
    if isinstance(specification, ParallelSpecification):
        if specification.execution not in {"sequential", "concurrent"}:
            raise ValueError(f"{path}.execution is invalid.")
        if specification.merge != "concatenate":
            raise ValueError(f"{path}.merge is not implemented.")
        if not specification.branches:
            raise ValueError(f"{path}.branches cannot be empty.")
        names = [branch.name for branch in specification.branches]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}.branches contain duplicate names.")
        for index, branch in enumerate(specification.branches):
            if not branch.name:
                raise ValueError(f"{path}.branches.{index}.name cannot be empty.")
            _validate_step(branch.body, f"{path}.branches.{index}.body")
        return
    if isinstance(specification, LocalEnumerationSpecification):
        if specification.trajectories < 1:
            raise ValueError(f"{path}.trajectories must be positive.")
        if (specification.iterations is None) == (specification.walk_time is None):
            raise ValueError(
                f"{path} requires exactly one of iterations or walk_time."
            )
        if specification.iterations is not None and specification.iterations < 1:
            raise ValueError(f"{path}.iterations must be positive.")
        if specification.walk_time is not None and specification.walk_time <= 0:
            raise ValueError(f"{path}.walk_time must be positive.")
        if specification.walker.kind != "walker":
            raise ValueError(f"{path}.walker must declare a walker.")
        if specification.generator.kind != "mutation_generator":
            raise ValueError(
                f"{path}.generator must declare a mutation generator."
            )
        _validate_step(specification.walker, f"{path}.walker")
        _validate_step(specification.generator, f"{path}.generator")
        _validate_step(specification.filters, f"{path}.filters")
        return
    raise TypeError(f"Unsupported pipeline specification at {path}: {specification!r}")
