"""Public factories for declarative nodes, distinct from executable steps."""

from collections.abc import Mapping

from pep_compass.data_schemas.graph import (
    LocalEnumerationSpec,
    LoopSpec,
    ParallelSpec,
    StepSpec,
    thaw,
)
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.loading import load_components
from pep_compass.utils.validation import validate_integer


def create_step(*, component, strategy, name=None, parameters=None, catalog=None):
    """Use the identical parameter-validation path for all public syntax styles."""
    if catalog is None:
        load_components()
        catalog = component_catalog
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("Step name must be a non-empty string or None.")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise ValueError("Step parameters must be a mapping.")
    values = thaw(parameters or {})
    catalog.validate(component, strategy, values)
    resolved = catalog.family(component).registry.configure(strategy, values)
    return StepSpec(
        component,
        catalog.family(component).registry.definition(strategy).name,
        resolved,
        name,
    )


class Step:
    """Create a registered component declaration without initializing models."""

    @staticmethod
    def create(*, component, strategy, name=None, parameters=None, catalog=None):
        """Return a StepSpec validated against the selected component registry.

        :param component: Family, such as walker, filter, oracle or save.
        :param strategy: Registered implementation name.
        :param name: Optional human-readable label.
        :param parameters: Public strategy parameters.
        :param catalog: Optional explicitly loaded custom catalog.
        :return: Immutable declaration with a stable identity.
        """
        return create_step(
            component=component,
            strategy=strategy,
            name=name,
            parameters=parameters,
            catalog=catalog,
        )


class Loop:
    """Declare repeated complete-batch feedback."""

    @staticmethod
    def create(*, iterations, steps, name=None):
        """Return a loop declaration; zero iterations is an identity operation."""
        validate_integer(iterations, "loop.iterations")
        return LoopSpec(iterations, tuple(steps), name)


class Parallel:
    """Declare named branches using the existing deterministic merge policy."""

    @staticmethod
    def create(*, branches, name=None, execution="sequential", merge="concatenate"):
        """Return named branch declarations without launching threads."""
        if not branches or any(not isinstance(key, str) or not key for key in branches):
            raise ValueError("Parallel requires non-empty named branches.")
        if execution not in {"sequential", "concurrent"} or merge != "concatenate":
            raise ValueError("Unsupported parallel execution or merge policy.")
        return ParallelSpec(branches, name, execution, merge)


class LocalEnumeration:
    """Declare private trajectories and a separate filtered mutation pool."""

    @staticmethod
    def create(
        *,
        walker,
        mutation_generator,
        filters=(),
        trajectories=1,
        iterations=None,
        walk_time=None,
        trajectory_execution="batched",
        include_walk_points=True,
        name=None,
    ):
        """Return the established local-enumeration structure.

        Exactly one of iterations and walk_time is required. Local filters do
        not receive arbitrary operator subgraphs.
        """
        from pep_compass.core.builder.schema import lower_node
        from pep_compass.core.builder.validation import validate_pipeline_specification
        from pep_compass.data_schemas.specification import (
            FlowSpecification,
            PipelineSpecification,
        )

        node = LocalEnumerationSpec(
            walker,
            mutation_generator,
            tuple(filters),
            trajectories,
            iterations,
            walk_time,
            trajectory_execution,
            include_walk_points,
            name,
        )
        if (
            not isinstance(walker, StepSpec)
            or not isinstance(mutation_generator, StepSpec)
            or any(
                not isinstance(child, StepSpec) or child.component != "filter"
                for child in node.filters
            )
        ):
            raise ValueError(
                "LocalEnumeration requires walker/generator declarations and filter components."
            )
        validate_pipeline_specification(
            PipelineSpecification(FlowSpecification((lower_node(node),)))
        )
        return node
