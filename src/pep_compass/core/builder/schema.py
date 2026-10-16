"""Adapt neutral public declarations to the established computation operators."""

from pep_compass.data_schemas.execution import OptimizationLimits
from pep_compass.data_schemas.graph import (
    LocalEnumerationSpec,
    LoopSpec,
    ParallelSpec,
    PipelineSchema,
    StepSpec,
    thaw,
)
from pep_compass.data_schemas.specification import (
    BranchSpecification,
    ComponentSpecification,
    FlowSpecification,
    LocalEnumerationSpecification,
    LoopSpecification,
    ParallelSpecification,
    PipelineSpecification,
)


def lower_node(node):
    """Translate declarations without changing existing operator semantics."""
    if isinstance(node, StepSpec):
        return ComponentSpecification(
            node.component, node.strategy, thaw(node.parameters)
        )
    if isinstance(node, LoopSpec):
        return LoopSpecification(
            node.iterations,
            FlowSpecification(tuple(lower_node(child) for child in node.steps)),
        )
    if isinstance(node, ParallelSpec):
        return ParallelSpecification(
            tuple(
                BranchSpecification(
                    name, FlowSpecification(tuple(lower_node(child) for child in steps))
                )
                for name, steps in node.branches.items()
            ),
            node.execution,
            node.merge,
        )
    if isinstance(node, LocalEnumerationSpec):
        return LocalEnumerationSpecification(
            node.trajectories,
            node.iterations,
            node.walk_time,
            lower_node(node.walker),
            lower_node(node.mutation_generator),
            FlowSpecification(tuple(lower_node(child) for child in node.filters)),
            node.trajectory_execution,
            node.include_walk_points,
        )
    raise TypeError(f"Expected a graph declaration, got {type(node).__name__}.")


def lower_schema(schema: PipelineSchema, limits=None):
    """Return the legacy-neutral shape consumed by established graph validation."""
    return PipelineSpecification(
        FlowSpecification(tuple(lower_node(node) for node in schema.steps)),
        limits or OptimizationLimits(),
    )


def bind_nodes(schema, root, catalog):
    """Attach stable identities and display metadata to constructed operators."""
    bound_objects = set()

    def bind(node, step):
        if id(step) in bound_objects:
            raise ValueError(
                "A factory reused one executable object for multiple graph nodes."
            )
        bound_objects.add(id(step))
        step.node_id = node.node_id
        step.display_name = node.name
        if isinstance(node, StepSpec):
            step.configuration = catalog.family(node.component).registry.configure(
                node.strategy, thaw(node.parameters)
            )
            step.display_parameter_names = tuple(
                catalog.family(node.component).registry.describe(node.strategy)[
                    "display_parameters"
                ]
            )
        elif isinstance(node, LoopSpec):
            for child, executable in zip(node.steps, step.body.steps, strict=True):
                bind(child, executable)
        elif isinstance(node, ParallelSpec):
            for name, children in node.branches.items():
                for child, executable in zip(
                    children, step.branches[name].steps, strict=True
                ):
                    bind(child, executable)
        elif isinstance(node, LocalEnumerationSpec):
            bind(node.walker, step.walker)
            bind(node.mutation_generator, step.mutation_generator)
            for child, executable in zip(node.filters, step.filters.steps, strict=True):
                bind(child, executable)

    for node, step in zip(schema.steps, root.steps, strict=True):
        bind(node, step)
