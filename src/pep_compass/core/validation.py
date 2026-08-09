"""Semantic validation of neutral PepCompass pipeline specifications."""

from __future__ import annotations

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
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
    raise TypeError(f"Unsupported pipeline specification at {path}: {specification!r}")
