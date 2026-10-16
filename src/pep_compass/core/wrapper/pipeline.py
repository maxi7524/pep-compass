"""Notebook-friendly facade over one schema, builder and graph runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import RLock

from pep_compass.core.builder import PipelineBuilder
from pep_compass.core.wrapper.scopes import SequenceBuilder
from pep_compass.data_schemas.execution import RunnerSettings
from pep_compass.data_schemas.graph import (
    AutoencoderSpec,
    PipelineSchema,
    StepSpec,
    node_dict,
    thaw,
)
from pep_compass.registry.components.catalog import component_catalog
from pep_compass.registry.components.loading import load_components
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class Pipeline(SequenceBuilder):
    """Edit declarative computations and execute isolated runs.

    :param steps: Optional ordered declarations.
    :param autoencoder: Optional initialized resource for manual Python use.
    :param catalog: Prepared custom catalog; defaults explicitly load built-ins.
    """

    def __init__(self, steps=None, *, autoencoder=None, catalog=None):
        if catalog is None:
            load_components()
            catalog = component_catalog
        self.catalog = catalog
        self._owner = self
        self._schema = PipelineSchema(tuple(steps or ()))
        self._provided_autoencoder = autoencoder
        self._compiled = None
        self._compiled_schema = None
        self._model_manifest = ()
        self._settings = RunnerSettings()
        self._running = False
        self._open_scopes = 0
        self._lock = RLock()
        self.last_result = None

    @classmethod
    def create(cls, steps=(), **kwargs):
        """Create the same editable pipeline as the incremental constructor."""
        return cls(steps, **kwargs)

    @classmethod
    def from_schema(cls, schema, **kwargs):
        """Restore a declaration without building models or running computation."""
        if isinstance(schema, str):
            schema = json.loads(schema)
        if isinstance(schema, dict):
            schema = PipelineSchema.from_dict(schema)
        if not isinstance(schema, PipelineSchema):
            raise TypeError("Expected PipelineSchema, JSON or a schema mapping.")
        pipeline = cls(**kwargs)
        pipeline._schema = schema
        return pipeline

    @classmethod
    def from_predefined(cls, name, **kwargs):
        """Create a fresh named preset; no weights are initialized here."""
        from pep_compass.predefined.factory import create

        return create(name, **kwargs)

    def _ensure_editable(self):
        if self._running:
            raise RuntimeError("An executing pipeline cannot be edited.")

    def _invalidate(self):
        self._compiled = None
        self._compiled_schema = None

    def _append(self, spec):
        with self._lock:
            self._ensure_editable()
            self._schema = replace(self._schema, steps=(*self._schema.steps, spec))
            self._invalidate()

    def add_autoencoder(
        self, strategy=None, *, model=None, parameters=None, device="cpu"
    ):
        """Configure one shared autoencoder, or inject an already initialized one.

        :param strategy: Registered method name, initialized encoder, or None for defaults.
        :param model: Named weights, required for non-default methods.
        :param parameters: Explicit constructor overrides.
        :param device: Resource device; changing it requires rebuilding.
        :return: This editable pipeline.
        """
        with self._lock:
            self._ensure_editable()
            if strategy is not None and not isinstance(strategy, str):
                if model is not None or parameters is not None:
                    raise ValueError(
                        "Injected autoencoders do not accept constructor parameters."
                    )
                self._provided_autoencoder = strategy
                self._schema = replace(self._schema, autoencoder=None)
            else:
                from pep_compass.predefined.default_params import default_autoencoder
                from pep_compass.registry.components.families.autoencoders import (
                    AutoencoderRegistry,
                )

                default = default_autoencoder()
                method = strategy or default.strategy
                selected_model = model or (
                    default.model if method == default.strategy else None
                )
                if selected_model is None:
                    raise ValueError("Select an explicit named model.")
                values = {
                    **(thaw(default.parameters) if method == default.strategy else {}),
                    **(parameters or {}),
                }
                if set(values) & {"device", "model_name", "model"}:
                    raise ValueError(
                        "Select model and device using add_autoencoder arguments, not parameters."
                    )
                descriptor = AutoencoderRegistry.model(method, selected_model)
                AutoencoderRegistry.validate(
                    method, {"device": device, **descriptor.parameters, **values}
                )
                self._provided_autoencoder = None
                self._schema = replace(
                    self._schema,
                    autoencoder=AutoencoderSpec(
                        method, selected_model, values, str(device)
                    ),
                )
            self._invalidate()
        return self

    def set_runner_settings(self, *, execution=None, storage=None, results=None):
        """Set immutable defaults; per-run overrides do not mutate these settings."""
        with self._lock:
            self._ensure_editable()
            updated = self._settings.update(
                execution=execution, storage=storage, results=results
            )
            if updated.execution.device != self._settings.execution.device:
                self._invalidate()
            self._settings = updated
        return self

    @property
    def runner_settings(self):
        """Return validated run defaults, separate from the algorithm schema."""
        return self._settings

    def _edit(self, target, **changes):
        with self._lock:
            self._ensure_editable()
            self._schema = self._schema.edit(target, **changes)
            self._invalidate()

    def insert_step(self, step, *, before=None, after=None):
        """Insert adjacent to one stable identity in the same parent sequence."""
        if (before is None) == (after is None):
            raise ValueError("Select exactly one of before or after.")
        self._edit(before or after, insert=step, before=before is not None)

    def remove_step(self, node_id):
        """Remove one declaration and its owned descendants."""
        self._edit(node_id)

    def replace_step(self, node_id, step):
        """Replace configuration while retaining the target identity."""
        self._edit(node_id, replacement=step)

    def build(self, *, execution=None):
        """Validate and construct an independent graph snapshot; return self.

        :param execution: Optional execution settings for resource initialization.
        """
        with self._lock:
            self._ensure_editable()
            if self._open_scopes:
                raise RuntimeError("Close all builder scopes before building.")
            builder = PipelineBuilder(self._provided_autoencoder, catalog=self.catalog)
            settings = self._settings.update(execution=execution)
            compiled = builder.build(self._schema, execution=settings.execution)
            self._compiled = compiled
            self._compiled_schema = builder.resolved_schema
            self._model_manifest = builder.model_manifest
            logger.info(
                "Built pipeline nodes=%s.", sum(1 for _ in self._compiled_schema.walk())
            )
        return self

    def run(
        self, sequences=None, *, execution=None, storage=None, results=None, seed=None
    ):
        """Run with fresh state, preserving defaults and previously returned results.

        :param sequences: Explicit input peptide strings.
        :param execution: Per-run execution overrides.
        :param storage: Per-run storage overrides.
        :param results: Per-run public return-mode overrides.
        :param seed: Convenient deterministic seed override.
        :return: RunResult, AnalysisReader or persistent run directory.
        """
        if (
            isinstance(sequences, (str, bytes))
            or sequences is None
            or not sequences
            or any(not isinstance(value, str) or not value for value in sequences)
        ):
            raise ValueError("Provide a non-empty collection of peptide strings.")
        with self._lock:
            self._ensure_editable()
            if self._open_scopes:
                raise RuntimeError("Close all builder scopes before execution.")
            options = dict(execution or {})
            if seed is not None:
                options["seed"] = seed
            settings = self._settings.update(
                execution=options, storage=storage, results=results
            )
            if self._compiled is None:
                self.build(execution=settings.execution)
            self._running = True
        try:
            from pep_compass.optimization.engine.runner import Runner

            result = Runner(
                self._compiled, self._compiled_schema, settings, self._model_manifest
            ).run(sequences, settings=settings)
            self.last_result = result
            if settings.results.return_mode == "reader":
                return result.reader
            if settings.results.return_mode == "path":
                return result.path
            return result
        finally:
            with self._lock:
                self._running = False

    def describe(
        self, component=None, strategy=None, *, display="display", technical=False
    ):
        """List available choices with automatic display by default.

        :param component: Optional family restriction, including autoencoder.
        :param strategy: Optional registered implementation restriction.
        :param display: raw records, pd DataFrame, or automatic display (default).
        :param technical: Include injected services and batch-field contracts.
        :return: Records, DataFrame, or None after displaying tables.
        """
        from pep_compass.core.wrapper.descriptions import present_description

        return present_description(
            self._describe(component, strategy), display, technical=technical
        )

    def _describe(self, component=None, strategy=None):
        """List available component choices, or describe a selected strategy.

        :param component: Optional family restriction, including autoencoder.
        :param strategy: Optional registered implementation restriction.
        :return: Structured catalog records; no model weights are loaded.
        """
        descriptions = (
            self.catalog.describe(component, strategy)
            if component != "autoencoder"
            else []
        )
        if component is None or component == "autoencoder":
            from pep_compass.registry.components.families.autoencoders import (
                AutoencoderRegistry,
            )

            encoders = []
            for method in (
                (strategy,) if strategy is not None else AutoencoderRegistry.methods()
            ):
                item = AutoencoderRegistry.describe(method)
                item.update(
                    component="autoencoder",
                    models=list(AutoencoderRegistry.models(method)),
                )
                from pep_compass.registry.models.catalog import model_catalog

                item["model_descriptions"] = model_catalog.describe(
                    f"autoencoder.{method}"
                )
                encoders.append(item)
            if strategy is not None:
                return encoders[0]
            descriptions.extend(encoders)
        return descriptions

    def _describe_node(self, node_id=None, *, component=None, strategy=None):
        """Read registry and placement details independently of presentation."""
        if node_id is None:
            if component is None or strategy is None:
                raise ValueError("Select node_id or both component and strategy.")
            return self._describe(component, strategy)
        if component is not None or strategy is not None:
            raise ValueError("Do not combine a node identity with strategy selectors.")
        for node, parent, position in self._schema.walk():
            if node.node_id == node_id:
                description = {
                    "node": node_dict(node),
                    "parent_id": parent,
                    "position": position,
                }
                prefix = position.rpartition(".")[0]
                siblings = [
                    candidate.node_id
                    for candidate, owner, location in self._schema.walk()
                    if owner == parent and location.rpartition(".")[0] == prefix
                ]
                offset = siblings.index(node_id)
                description["previous_id"] = siblings[offset - 1] if offset else None
                description["next_id"] = (
                    siblings[offset + 1] if offset + 1 < len(siblings) else None
                )
                if isinstance(node, StepSpec):
                    description["definition"] = self._describe(
                        node.component, node.strategy
                    )
                    description["resolved_parameters"] = self.catalog.family(
                        node.component
                    ).registry.configure(node.strategy, thaw(node.parameters))
                    definition = self.catalog.family(
                        node.component
                    ).registry.definition(node.strategy)
                    parameters = description["resolved_parameters"]
                    description["resolved_contract"] = {
                        "input_fields": list(
                            definition.input_fields(parameters)
                            if callable(definition.input_fields)
                            else definition.input_fields
                        ),
                        "output_fields": list(
                            definition.output_fields(parameters)
                            if callable(definition.output_fields)
                            else definition.output_fields
                        ),
                    }
                return description
        raise KeyError(node_id)

    def describe_component(
        self,
        component=None,
        strategy=None,
        *,
        node_id=None,
        display="display",
        technical=False,
    ):
        """Describe an available component or one placed graph node.

        :param component: Registered family; required together with strategy.
        :param strategy: Registered method in the selected family.
        :param node_id: Existing node identity, instead of family/strategy selectors.
        :param display: raw records, pd DataFrame, or automatic display (default).
        :param technical: Include injected services and batch-field contracts.
        :return: Detailed records, a DataFrame, or None after display.
        :raises ValueError: If selectors or display mode are invalid.
        :raises KeyError: If the requested node does not exist.
        """
        from pep_compass.core.wrapper.descriptions import present_description

        value = self._describe_node(node_id, component=component, strategy=strategy)
        return present_description(value, display, technical=technical)

    @classmethod
    def list_components(cls, *, display="display"):
        """List built-ins; instance.describe includes an instance's custom catalog."""
        return cls().describe(display=display)

    def to_schema(self, format="object"):
        """Return an immutable snapshot, JSON, text, Mermaid or SVG representation."""
        if format == "object":
            return PipelineSchema.from_dict(self._schema.to_dict())
        if format == "json":
            return json.dumps(self._schema.to_dict(), indent=2, allow_nan=False)
        from pep_compass.core.wrapper.visualization import render

        display = None
        if self._compiled is not None:
            from pep_compass.optimization.engine.inspection import find_step

            display = {}
            for node, _, _ in self._schema.walk():
                if isinstance(node, StepSpec):
                    executable = find_step(self._compiled.root, node.node_id)
                    important = executable.display_info()
                    if important:
                        display[node.node_id] = important
        return render(self._schema, self.catalog, format=format, display=display)

    def show(self, format="svg"):
        """Render graph structure without building or running components."""
        from pep_compass.core.wrapper.visualization import GraphView

        return GraphView(self.to_schema(format), format)

    show_pipeline_graph = show

    def save_schema(self, path):
        """Persist the declarative graph independently of run data."""
        target = Path(path)
        target.write_text(self.to_schema("json"), encoding="utf-8")
        return target

    def save_result(self, path):
        """Export the last run's retained data through its common saver backend."""
        if self.last_result is None:
            raise RuntimeError("The pipeline has no completed result to save.")
        return self.last_result.save(path)
