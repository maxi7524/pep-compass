"""Cardinality contracts supplied by implementations to generic graph tooling."""


def component_output_bound(kind, strategy, parameters, count, *, catalog=None):
    """Return a registered upper bound without constructing the implementation."""
    if catalog is not None and kind in catalog.names():
        registry = catalog.family(kind).registry
        if strategy in registry.names():
            definition = registry.definition(strategy)
            if definition.output_bound is not None:
                return definition.output_bound(parameters, count)
    if kind == "mutation_generator":
        maximum = parameters.get("maximum_candidates")
        return (
            count * maximum
            if isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum > 0
            else None
        )
    return count
