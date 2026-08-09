"""Construct executable PepCompass pipelines from neutral specifications."""

from __future__ import annotations

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)
from pep_compass.core.estimation import estimate_pipeline_stability
from pep_compass.core.validation import validate_pipeline_specification
from pep_compass.optimization.components.filters import FilterManager
from pep_compass.optimization.components.mutation_generators import (
    MutationGeneratorManager,
)
from pep_compass.optimization.components.oracles import OracleManager
from pep_compass.optimization.components.walkers import WalkerManager
from pep_compass.optimization.engine import Flow, Loop, Parallel, build_merger
from pep_compass.optimization.pipeline import PepCompassPipeline
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.engine.step import Step
from pep_compass.optimization.tracking import StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class PipelineBuilder:
    """Build and validate a :class:`PepCompassPipeline`.

    :param autoencoder: Initialized autoencoder injected into components that
        declare this service.
    """

    def __init__(self, autoencoder: Autoencoder) -> None:
        self.autoencoder = autoencoder
        self._load_builtin_components()

    def build(
        self,
        specification: PipelineSpecification,
        *,
        tracker: StepTracker | None = None,
        stability_monitor: StabilityMonitor | NullStabilityMonitor | None = None,
        initial_candidates: int | None = None,
    ) -> PepCompassPipeline:
        """Construct one executable pipeline from a neutral declaration.

        :param specification: Validatable computation-graph declaration.
        :param tracker: Optional runtime tracker.
        :param stability_monitor: Optional memory monitor.
        :return: Fully initialized executable pipeline.
        """
        validate_pipeline_specification(specification)
        root = self._build_step(specification.root)
        if initial_candidates is not None:
            estimate = estimate_pipeline_stability(
                specification,
                input_candidates=initial_candidates,
                latent_dimension=self.autoencoder.latent_dim,
            )
            logger.info(
                "Pipeline stability input_candidates=%s output_upper=%s "
                "peak_upper=%s latent_bytes_upper=%s warnings=%s.",
                estimate.input_candidates,
                estimate.output_candidates_upper,
                estimate.peak_candidates_upper,
                estimate.latent_bytes_upper,
                estimate.warnings,
            )
        logger.info("Constructed PepCompass pipeline root=%s.", root.name)
        return PepCompassPipeline(
            autoencoder=self.autoencoder,
            root=root,
            tracker=tracker,
            limits=specification.limits,
            stability_monitor=stability_monitor,
        )

    @staticmethod
    def _load_builtin_components() -> None:
        """Load built-in component registrations before resolution."""
        import pep_compass.optimization.components.filters.strategies  # noqa: F401
        import pep_compass.optimization.components.mutation_generators.strategies  # noqa: F401
        import pep_compass.optimization.components.oracles.strategies  # noqa: F401
        import pep_compass.optimization.components.walkers.strategies  # noqa: F401

    def _build_step(self, specification: StepSpecification) -> Step:
        """Recursively construct one declared computation node."""
        if isinstance(specification, ComponentSpecification):
            return self._build_component(specification)
        if isinstance(specification, FlowSpecification):
            return Flow([self._build_step(step) for step in specification.steps])
        if isinstance(specification, LoopSpecification):
            return Loop(
                self._build_step(specification.body),
                specification.iterations,
            )
        if isinstance(specification, ParallelSpecification):
            return Parallel(
                {
                    branch.name: self._build_step(branch.body)
                    for branch in specification.branches
                },
                execution=specification.execution,
                merger=build_merger(specification.merge),
            )
        raise TypeError(f"Unsupported pipeline specification: {specification!r}")

    def _build_component(self, specification: ComponentSpecification) -> Step:
        """Resolve and construct one registered computational component."""
        managers = {
            "walker": WalkerManager,
            "mutation_generator": MutationGeneratorManager,
            "filter": FilterManager,
            "oracle": OracleManager,
        }
        manager = managers[specification.kind]
        parameters = dict(specification.parameters)
        services = {"autoencoder": self.autoencoder}
        if specification.kind == "oracle":
            return manager.build(specification.method, **parameters)
        return manager.build(
            specification.method,
            services=services,
            **parameters,
        )
