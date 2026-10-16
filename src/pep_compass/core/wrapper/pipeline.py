"""Convenient Python facade over graph construction and independent execution."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from pep_compass.core.builder import PipelineBuilder
from pep_compass.data_schemas.execution import RunnerSettings
from pep_compass.data_schemas.graph import PipelineSchema


class Pipeline:
    """Combine an editable builder, run defaults and the most recent result.

    :param steps: Ordered concrete Step objects, without initialized model weights.
    :param autoencoder: Shared resource declaration or initialized Python resource.
    :param schema: Alternative immutable declaration snapshot.
    """

    def __init__(self, steps=(), *, autoencoder=None, schema=None):
        self.builder = PipelineBuilder(steps, autoencoder=autoencoder, schema=schema)
        self._compiled = None
        self._compiled_schema = None
        self._compiled_device = None
        self._settings = RunnerSettings()
        self._lock = RLock()
        self._running = False
        self.last_result = None

    @classmethod
    def from_schema(cls, schema, *, autoencoder=None):
        """Restore a portable snapshot without loading model weights."""
        if isinstance(schema, str):
            schema = json.loads(schema)
        if isinstance(schema, dict):
            schema = PipelineSchema.from_dict(schema)
        if not isinstance(schema, PipelineSchema):
            raise TypeError("Expected PipelineSchema, JSON or a schema mapping.")
        return cls(schema=schema, autoencoder=autoencoder)

    def _edit(self, method, *args, **kwargs):
        with self._lock:
            if self._running:
                raise RuntimeError("An executing pipeline cannot be edited.")
            result = getattr(self.builder, method)(*args, **kwargs)
            self._compiled = None
            return result

    def add_component(self, component, *, name=None):
        """Append a concrete component and return its stable placement identity."""
        return self._edit("add", component, name=name)

    def add_autoencoder(self, component):
        """Attach a shared autoencoder declaration and return this wrapper."""
        self._edit("add_autoencoder", component)
        return self

    def insert_component(self, component, *, before=None, after=None):
        """Insert next to one placement and return the new stable identity."""
        return self._edit("insert", component, before=before, after=after)

    def replace_component(self, node_id, component):
        """Replace configuration while keeping the placement identity."""
        return self._edit("replace", node_id, component)

    def remove_component(self, node_id):
        """Remove a placement and its owned descendants."""
        return self._edit("remove", node_id)

    def loop(self, *, iterations, name=None):
        """Collect a loop using a child builder context manager."""
        return self._edit("loop", iterations=iterations, name=name)

    def validate(self, *, initial_fields=()):
        """Validate graph structure and known data dependencies without weights."""
        return self.builder.validate(initial_fields=initial_fields)

    def build(self, *, execution=None):
        """Return an independent ExecutablePipeline built from the current schema."""
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot build during execution.")
            settings = self._settings.update(execution=execution)
            self._compiled = self.builder.build(execution=settings.execution)
            self._compiled_schema = self.builder.schema
            self._compiled_device = settings.execution.device
            return self._compiled

    def set_runner_settings(self, *, execution=None, storage=None, results=None):
        """Set validated run defaults; per-run overrides leave these unchanged."""
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot change settings during execution.")
            self._settings = self._settings.update(
                execution=execution, storage=storage, results=results
            )
        return self

    @property
    def runner_settings(self):
        """Return immutable execution, storage and result defaults."""
        return self._settings

    def run(self, sequences, *, execution=None, storage=None, results=None, seed=None):
        """Run with fresh state; build again after graph or device changes.

        :param sequences: Non-empty collection of peptide strings.
        :param execution: Overrides for seed, device and complete-batch limits.
        :param storage: Overrides for memory/SQLite and retained observations.
        :param results: Overrides for the public return mode.
        :param seed: Convenient deterministic seed override.
        :return: RunResult, AnalysisReader or persistent output path.
        """
        from pep_compass.utils.validation import validate_sequences

        sequences = validate_sequences(sequences)
        with self._lock:
            if self._running:
                raise RuntimeError(
                    "Concurrent runs require independent pipeline instances."
                )
            self.builder._ensure_closed()
            options = dict(execution or {})
            if seed is not None:
                options["seed"] = seed
            settings = self._settings.update(
                execution=options, storage=storage, results=results
            )
            if (
                self._compiled is None
                or self._compiled_schema is not self.builder.schema
                or self._compiled_device != settings.execution.device
            ):
                self.build(execution=settings.execution)
            self._running = True
        try:
            result = self._compiled.run(sequences, settings=settings)
            self.last_result = result
            return {"result": result, "reader": result.reader, "path": result.path}[
                settings.results.return_mode
            ]
        finally:
            with self._lock:
                self._running = False

    def describe(self, category=None, *, display="display"):
        """List public importable components without constructing their models.

        :param category: Optional public steps namespace or autoencoder.
        :param display: ``raw``, ``pd`` or automatic formatted ``display``.
        :return: Records, a pandas DataFrame, or None after presentation.
        """
        from pep_compass.core.wrapper.descriptions import available, present_description

        return present_description(available(category), display)

    def describe_component(
        self, component=None, *, node_id=None, display="display", technical=False
    ):
        """Describe a concrete class, configured object or existing graph placement.

        :param component: Public component class, function or configured instance.
        :param node_id: Alternative stable identity of a placed graph node.
        :param display: ``raw``, ``pd`` or formatted ``display``.
        :param technical: Include resources and CandidateBatch field contracts.
        :return: Definition records, DataFrame or None after presentation.
        """
        from pep_compass.core.wrapper.descriptions import (
            definition,
            present_description,
        )

        if (component is None) == (node_id is None):
            raise ValueError("Select a component or a node_id.")
        if node_id is not None:
            component = self.builder.schema.find(node_id).instantiate()
        return present_description(definition(component), display, technical=technical)

    def to_schema(self, format="object"):
        """Return a schema snapshot, JSON or its text, Mermaid or SVG rendering."""
        from pep_compass.core.wrapper.visualization import render

        return render(self.builder.schema, format=format)

    def show(self, format="svg"):
        """Return a portable graph string with inline SVG support in notebooks."""
        from pep_compass.core.wrapper.visualization import graph_view

        return graph_view(self.builder.schema, format)

    def show_pipeline_graph(self, format="svg"):
        """Display the current computational graph using the selected representation."""
        return self.show(format)

    def save_schema(self, path):
        """Persist the constructor graph independently of run results."""
        path = Path(path)
        path.write_text(self.to_schema("json"), encoding="utf-8")
        return path

    def save_result(self, path):
        """Export the retained last result through the common storage backend."""
        if self.last_result is None:
            raise RuntimeError("No completed result is available.")
        return self.last_result.save(path)
