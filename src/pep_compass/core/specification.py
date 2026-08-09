"""Neutral declarations used to construct a PepCompass pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

from pep_compass.optimization.engine.execution.state import OptimizationLimits


ComponentKind = Literal["walker", "mutation_generator", "filter", "oracle"]
ParallelExecution = Literal["sequential", "concurrent"]


@dataclass(frozen=True, slots=True)
class ComponentSpecification:
    """Declare one registered computational component."""

    kind: ComponentKind
    method: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FlowSpecification:
    """Declare steps executed sequentially in configuration order."""

    steps: tuple["StepSpecification", ...]


@dataclass(frozen=True, slots=True)
class LoopSpecification:
    """Declare repeated execution of one nested flow."""

    iterations: int
    body: FlowSpecification


@dataclass(frozen=True, slots=True)
class BranchSpecification:
    """Declare one named parallel branch."""

    name: str
    body: FlowSpecification


@dataclass(frozen=True, slots=True)
class ParallelSpecification:
    """Declare parallel branches and their deterministic merge policy."""

    branches: tuple[BranchSpecification, ...]
    execution: ParallelExecution = "sequential"
    merge: str = "concatenate"


StepSpecification: TypeAlias = (
    ComponentSpecification | FlowSpecification | LoopSpecification | ParallelSpecification
)


@dataclass(frozen=True, slots=True)
class PipelineSpecification:
    """Declare a complete pipeline independently of YAML and runtime I/O."""

    root: FlowSpecification
    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
