"""Translate configuration mappings exclusively through the public wrapper."""

from collections.abc import Mapping

from pep_compass.core.wrapper import LocalEnumeration, Loop, Parallel, Pipeline, Step
from pep_compass.utils.validation import validate_keys


def create_pipeline(configuration, *, catalog=None, autoencoder=None, execution=None):
    """Create an editable pipeline; construction remains owned by its builder.

    :param configuration: Exactly one of predefined, schema or explicit steps.
    :param catalog: Optional component catalog for Python applications.
    :param autoencoder: Optional injected Python resource; local execution only.
    :param execution: Runner overrides, used for preset seed and resource device.
    :return: Public Pipeline object without initialized model weights.
    """
    if not isinstance(configuration, Mapping):
        raise ValueError("pipeline must be a mapping.")
    validate_keys(
        configuration,
        {"predefined", "parameters", "schema", "steps", "autoencoder"},
        "pipeline",
    )
    modes = [
        name for name in ("predefined", "schema", "steps") if name in configuration
    ]
    if len(modes) != 1:
        raise ValueError("Select exactly one of pipeline.predefined, schema or steps.")
    mode = modes[0]
    if mode == "predefined":
        if (
            "autoencoder" in configuration
            or catalog is not None
            or autoencoder is not None
        ):
            raise ValueError(
                "A predefined factory owns its catalog and autoencoder declaration."
            )
        parameters = dict(configuration.get("parameters", {}))
        if set(parameters) & {"seed", "device"}:
            raise ValueError(
                "Set seed/device through runner.execution, not pipeline.parameters."
            )
        from pep_compass.registry.pipelines.catalog import pipeline_registry
        from pep_compass.registry.pipelines.loading import load_predefined

        load_predefined()
        accepted = {
            p["name"]
            for p in pipeline_registry.describe(configuration[mode])["parameters"]
        }
        parameters.update(
            {
                key: value
                for key, value in (execution or {}).items()
                if key in {"seed", "device"} and key in accepted and value is not None
            }
        )
        pipeline = Pipeline.from_predefined(configuration[mode], **parameters)
    else:
        if "parameters" in configuration:
            raise ValueError(
                "pipeline.parameters applies only to a predefined factory."
            )
        pipeline = (
            Pipeline.from_schema(
                configuration[mode], catalog=catalog, autoencoder=autoencoder
            )
            if mode == "schema"
            else Pipeline.create(
                steps=[_node(node, catalog) for node in configuration[mode]],
                catalog=catalog,
                autoencoder=autoencoder,
            )
        )
        if "autoencoder" in configuration:
            pipeline.add_autoencoder(**configuration["autoencoder"])
    return pipeline


def _node(configuration, catalog):
    if not isinstance(configuration, Mapping):
        raise ValueError("Each step declaration must be a mapping.")
    options = dict(configuration)
    kind = options.pop("type", "step")
    if kind == "step":
        return Step.create(**options, catalog=catalog)
    if kind == "loop":
        options["steps"] = [_node(node, catalog) for node in options["steps"]]
        return Loop.create(**options)
    if kind == "parallel":
        options["branches"] = {
            name: tuple(_node(node, catalog) for node in nodes)
            for name, nodes in options["branches"].items()
        }
        return Parallel.create(**options)
    if kind == "local_enumeration":
        for name in ("walker", "mutation_generator"):
            options[name] = _node(options[name], catalog)
        options["filters"] = [
            _node(node, catalog) for node in options.get("filters", ())
        ]
        return LocalEnumeration.create(**options)
    raise ValueError(f"Unknown step type: {kind!r}.")
