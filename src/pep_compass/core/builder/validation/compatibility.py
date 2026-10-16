"""Conservative field-flow checks driven by registered component contracts."""

from pep_compass.data_schemas.specification import (
    ComponentSpecification,
    FlowSpecification,
    LocalEnumerationSpecification,
    LoopSpecification,
    ParallelSpecification,
)


def validate_field_flow(specification, catalog, *, initial_fields=None):
    """Validate known field dependencies, including optional rows after merging.

    :param specification: Neutral graph declaration.
    :param catalog: Registered input/output contracts.
    :param initial_fields: Optional fields present at an isolated replay boundary.
    :raises ValueError: If a mandatory field is missing or possibly unavailable
        for a component that does not accept OptionalField inputs.
    """

    def declared(value, parameters):
        return set(value(parameters) if callable(value) else value)

    def merge(branches):
        names = set().union(*(set(branch) for branch in branches))
        return {
            name: all(branch.get(name, False) for branch in branches) for name in names
        }

    def visit(node, available):
        # Registered leaf contracts
        if isinstance(node, ComponentSpecification):
            registry = catalog.family(node.kind).registry
            definition = registry.definition(node.method)
            parameters = registry.configure(node.method, node.parameters)
            required = declared(definition.input_fields, parameters)
            missing = required - available.keys()
            if missing:
                raise ValueError(
                    f"{node.kind}:{node.method} requires unavailable fields: {sorted(missing)}."
                )
            optional = declared(definition.optional_inputs, parameters)
            uncertain = {
                name
                for name in required
                if not available[name] and name not in optional
            }
            if uncertain:
                raise ValueError(
                    f"{node.kind}:{node.method} does not accept optional fields: {sorted(uncertain)}."
                )
            retained = {
                name: guaranteed
                for name, guaranteed in available.items()
                if not any(
                    name.startswith(prefix)
                    for prefix in definition.invalidated_prefixes
                )
            }
            retained.update(
                {name: True for name in declared(definition.output_fields, parameters)}
            )
            return retained

        # Ordered composition and first/repeated-iteration compatibility
        if isinstance(node, FlowSpecification):
            for child in node.steps:
                available = visit(child, available)
            return available
        if isinstance(node, LoopSpecification):
            result = visit(node.body, dict(available))
            if node.iterations > 1:
                result = visit(node.body, result)
            return available if node.iterations == 0 else result

        # Unioned fields can be absent from individual branch emissions
        if isinstance(node, ParallelSpecification):
            return merge(
                [visit(branch.body, dict(available)) for branch in node.branches]
            )
        if isinstance(node, LocalEnumerationSpecification):
            points = visit(
                node.walker, {**available, "local_enumeration.center_sequence": True}
            )
            filtered = visit(node.filters, visit(node.generator, points))
            result = merge([filtered, points]) if node.include_walk_points else filtered
            from pep_compass.optimization.engine.operations.local_enumeration import (
                _TRANSIENT_WALKER_FIELDS,
            )

            return {
                name: guaranteed
                for name, guaranteed in result.items()
                if name not in _TRANSIENT_WALKER_FIELDS
            }
        raise TypeError(type(node).__name__)

    if initial_fields is None:
        available = {"lineage.candidate_id": True, "lineage.parent_candidate_id": True}
    else:
        from pep_compass.data_schemas.candidates import OptionalField

        available = (
            {
                name: not isinstance(value, OptionalField)
                for name, value in initial_fields.items()
            }
            if hasattr(initial_fields, "items")
            else {name: True for name in initial_fields}
        )
    return visit(specification.root, available)
