"""Own graph edits, immutable declarations and compilation into executable objects."""

from dataclasses import replace

from pep_compass.configuration.objects import Configurable, construct
from pep_compass.core.builder.validation import validate
from pep_compass.data_schemas.execution import ExecutionSettings, RunnerSettings
from pep_compass.data_schemas.graph import NodeSpec, PipelineSchema
from pep_compass.steps.flow.strategies.sequence import Flow
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

logger = get_custom_logger(__name__)


class PipelineBuilder:
    """Collect component placements; every build constructs an independent graph.

    :param steps: Ordered lightweight Step instances.
    :param autoencoder: Lightweight resource declaration or an initialized resource.
    :param schema: Alternative immutable graph declaration.
    """

    def __init__(self, steps=(), *, autoencoder=None, schema=None):
        if schema is not None and steps:
            raise ValueError("Select steps or schema, not both.")
        self._schema = schema or PipelineSchema(
            tuple(NodeSpec.capture(step) for step in steps)
        )
        self._provided_autoencoder = None
        self._open_scopes = 0
        if autoencoder is not None:
            self.add_autoencoder(autoencoder)

    @property
    def schema(self):
        """Return the current immutable declaration snapshot."""
        return self._schema

    def add(self, component, *, name=None):
        """Append a component; return its stable placement identity."""
        node = NodeSpec.capture(component, name=name)
        return self._append_node(node)

    def _append_node(self, node):
        self._schema = replace(self._schema, steps=(*self._schema.steps, node))
        return node.node_id

    def add_autoencoder(self, component):
        """Attach one shared resource declaration or initialized Python resource."""
        self._provided_autoencoder = (
            None if isinstance(component, Configurable) else component
        )
        self._schema = replace(
            self._schema,
            autoencoder=component.to_config()
            if isinstance(component, Configurable)
            else None,
        )
        return self

    def insert(self, component, *, before=None, after=None):
        """Insert in an ordered sequence; return the inserted placement identity."""
        if (before is None) == (after is None):
            raise ValueError("Select exactly one of before or after.")
        previous = {node.node_id for node, _, _ in self._schema.walk()}
        self._schema = self._schema.edit(
            before or after, insert=component, before=before is not None
        )
        return next(
            node.node_id
            for node, _, _ in self._schema.walk()
            if node.node_id not in previous
        )

    def replace(self, node_id, component):
        """Replace a declaration while retaining its identity."""
        self._schema = self._schema.edit(node_id, replacement=component)
        return node_id

    def remove(self, node_id):
        """Remove a placement and its owned descendants."""
        self._schema = self._schema.edit(node_id)

    def loop(self, *, iterations, name=None):
        """Collect a loop body through the same builder API in a context manager."""
        from pep_compass.steps.flow import Loop

        return _Scope(
            self, lambda steps: Loop(steps=steps, iterations=iterations), name
        )

    def _ensure_closed(self):
        if self._open_scopes:
            raise RuntimeError(
                "Close all construction scopes before validation or build."
            )

    def validate(self, *, initial_fields=()):
        """Check structure and field flow before initializing any resources."""
        self._ensure_closed()
        return validate(self._schema, initial_fields=initial_fields)

    def build(self, *, execution=None, initial_fields=()):
        """Validate and return a fresh ExecutablePipeline for this snapshot.

        :param execution: Resource initialization seed and device settings.
        :param initial_fields: Additional fields available to graph inputs.
        :return: Independent prepared computation graph.
        :raises ValueError: If declarations or required resource settings are invalid.
        """
        from pep_compass.autoencoder import HydrAMP
        from pep_compass.optimization.engine.pipeline import ExecutablePipeline
        from pep_compass.optimization.saving.artifacts import capture_model_artifacts

        self.validate(initial_fields=initial_fields)
        execution = execution or ExecutionSettings()
        schema = self._schema
        if schema.autoencoder is None and self._provided_autoencoder is None:
            schema = replace(schema, autoencoder=HydrAMP().to_config())
        with capture_model_artifacts() as artifacts, random_scope(execution.seed):
            resource = (
                construct(schema.autoencoder)
                if schema.autoencoder is not None
                else self._provided_autoencoder
            )
            autoencoder = (
                resource.build(device=execution.device)
                if isinstance(resource, Configurable)
                else resource
            )
            root = Flow([node.instantiate() for node in schema.steps])
            root.initialize(autoencoder)
        logger.info("Compiled graph nodes=%s.", sum(1 for _ in schema.walk()))
        return ExecutablePipeline(
            autoencoder=autoencoder,
            root=root,
            schema=schema,
            model_manifest=tuple(artifacts),
            settings=RunnerSettings(execution=execution),
        )


class _Scope:
    """Commit an isolated child builder only after a successful scope exit."""

    def __init__(self, owner, factory, name):
        self.owner, self.factory, self.name = owner, factory, name
        self.builder = PipelineBuilder()

    def __enter__(self):
        self.owner._open_scopes += 1
        return self.builder

    def __exit__(self, kind, value, traceback):
        self.owner._open_scopes -= 1
        if kind is None:
            node = NodeSpec.capture(
                self.factory(
                    [child.instantiate() for child in self.builder.schema.steps]
                ),
                name=self.name,
            )
            self.owner._append_node(
                replace(node, children={"steps": self.builder.schema.steps})
            )
        return False
