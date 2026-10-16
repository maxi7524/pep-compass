"""Internal execution API used by the core builder and ``PepCompassPipeline``.

``execution`` owns the lifecycle of one step call and shared run state.
``operations`` owns composite steps that traverse other steps. Concrete
walkers, generators, filters, and oracles implement the same ``Step`` contract
under ``optimization.components``.
"""

from pep_compass.data_schemas.execution import OptimizationLimits
from pep_compass.data_schemas.results import OptimizationResult
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.optimization.engine.operations import (
    Flow,
    LocalEnumeration,
    Loop,
    Parallel,
    build_merger,
)

__all__ = [
    "Flow",
    "LocalEnumeration",
    "Loop",
    "OptimizationContext",
    "OptimizationLimits",
    "OptimizationResult",
    "OptimizationState",
    "Parallel",
    "Step",
    "build_merger",
]
