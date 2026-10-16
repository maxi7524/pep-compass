"""Resolve declared dependencies independently of the public wrapper."""

from dataclasses import replace

from pep_compass.data_schemas.graph import (
    LocalEnumerationSpec,
    LoopSpec,
    ParallelSpec,
    StepSpec,
    thaw,
)


def resolve_schema(schema, catalog):
    """Resolve all component defaults using their registered configuration types."""

    def resolve(node):
        if isinstance(node, StepSpec):
            return replace(
                node,
                parameters=catalog.family(node.component).registry.configure(
                    node.strategy, thaw(node.parameters)
                ),
            )
        if isinstance(node, LoopSpec):
            return replace(node, steps=tuple(resolve(child) for child in node.steps))
        if isinstance(node, ParallelSpec):
            return replace(
                node,
                branches={
                    name: tuple(resolve(child) for child in steps)
                    for name, steps in node.branches.items()
                },
            )
        if isinstance(node, LocalEnumerationSpec):
            return replace(
                node,
                walker=resolve(node.walker),
                mutation_generator=resolve(node.mutation_generator),
                filters=tuple(resolve(child) for child in node.filters),
            )
        raise TypeError(f"Unsupported graph declaration: {type(node).__name__}.")

    encoder = schema.autoencoder
    if encoder is not None:
        from pep_compass.registry.components.families.autoencoders import (
            AutoencoderRegistry,
        )

        if set(encoder.parameters) & {"device", "model", "model_name"}:
            raise ValueError(
                "Autoencoder model and device must use their dedicated declaration fields."
            )
        descriptor = AutoencoderRegistry.model(encoder.strategy, encoder.model)
        values = AutoencoderRegistry.configure(
            encoder.strategy,
            {
                "device": encoder.device,
                **descriptor.parameters,
                **thaw(encoder.parameters),
            },
        )
        values.pop("device", None)
        values.pop("model_name", None)
        encoder = replace(encoder, parameters=values)
    return replace(
        schema, steps=tuple(resolve(node) for node in schema.steps), autoencoder=encoder
    )


def resolve_autoencoder(schema, injected, execution):
    """Construct a declared shared model or validate an injected resource device."""
    if injected is not None:
        if (
            execution.device is not None
            and str(getattr(injected, "device", "cpu")) != execution.device
        ):
            raise ValueError(
                "Injected autoencoder device differs from runner settings."
            )
        return injected, schema
    declaration = schema.autoencoder
    if declaration is None:
        raise ValueError("Add a shared autoencoder before building the graph.")
    from pep_compass.autoencoder.factory import AutoencoderFactory

    device = execution.device or declaration.device
    model = AutoencoderFactory.build(
        declaration.strategy,
        model=declaration.model,
        device=device,
        **thaw(declaration.parameters),
    )
    return model, replace(schema, autoencoder=replace(declaration, device=device))
