"""Immutable, serializable declarations shared by Python and future adapters."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4


def freeze(value):
    """Copy nested configuration containers into immutable values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Configuration mappings require string keys.")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(
                "Graph configuration requires finite numbers; use the strategy's unlimited sentinel."
            )
        return value
    if hasattr(value, "detach"):
        return freeze(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return freeze(value.tolist())
    raise TypeError(
        f"Graph configuration requires portable values, got {type(value).__name__}."
    )


def thaw(value):
    """Return independent JSON-shaped containers from frozen configuration."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return deepcopy(value)


def new_id():
    """Allocate an identity without consuming computational RNG streams."""
    return "node_" + uuid4().hex


@dataclass(frozen=True, slots=True)
class StepSpec:
    """Select a registered component independently of execution position."""

    component: str
    strategy: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    node_id: str = field(default_factory=new_id)

    def __post_init__(self):
        if not isinstance(self.parameters, Mapping):
            raise ValueError("Step parameters must be a mapping.")
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True, slots=True)
class LoopSpec:
    """Repeat one ordered sequence with complete-batch feedback."""

    iterations: int
    steps: tuple[NodeSpec, ...]
    name: str | None = None
    node_id: str = field(default_factory=new_id)

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    """Run named branches from one input with the existing merge semantics."""

    branches: Mapping[str, tuple[NodeSpec, ...]]
    name: str | None = None
    execution: str = "sequential"
    merge: str = "concatenate"
    node_id: str = field(default_factory=new_id)

    def __post_init__(self):
        object.__setattr__(
            self,
            "branches",
            MappingProxyType(
                {name: tuple(steps) for name, steps in self.branches.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalEnumerationSpec:
    """Keep private walker trajectories separate from mutation emissions."""

    walker: StepSpec
    mutation_generator: StepSpec
    filters: tuple[StepSpec, ...] = ()
    trajectories: int = 1
    iterations: int | None = None
    walk_time: float | None = None
    trajectory_execution: str = "batched"
    include_walk_points: bool = True
    name: str | None = None
    node_id: str = field(default_factory=new_id)

    def __post_init__(self):
        object.__setattr__(self, "filters", tuple(self.filters))


NodeSpec: TypeAlias = StepSpec | LoopSpec | ParallelSpec | LocalEnumerationSpec


@dataclass(frozen=True, slots=True)
class AutoencoderSpec:
    """Configure one shared model resource, not an execution-order node."""

    strategy: str
    model: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def __post_init__(self):
        object.__setattr__(self, "parameters", freeze(self.parameters))


def children(node):
    """Yield role-labelled children in deterministic structural order."""
    if isinstance(node, LoopSpec):
        return [("steps", child) for child in node.steps]
    if isinstance(node, ParallelSpec):
        return [
            (name, child) for name, steps in node.branches.items() for child in steps
        ]
    if isinstance(node, LocalEnumerationSpec):
        return [
            ("walker", node.walker),
            ("mutation_generator", node.mutation_generator),
            *[("filters", child) for child in node.filters],
        ]
    return []


def node_dict(node):
    """Serialize a node with stable identities and explicit operation tags."""
    common = {"node_id": node.node_id, "name": node.name}
    if isinstance(node, StepSpec):
        return {
            **common,
            "type": "step",
            "component": node.component,
            "strategy": node.strategy,
            "parameters": thaw(node.parameters),
        }
    if isinstance(node, LoopSpec):
        return {
            **common,
            "type": "loop",
            "iterations": node.iterations,
            "steps": [node_dict(child) for child in node.steps],
        }
    if isinstance(node, ParallelSpec):
        return {
            **common,
            "type": "parallel",
            "execution": node.execution,
            "merge": node.merge,
            "branches": {
                name: [node_dict(child) for child in steps]
                for name, steps in node.branches.items()
            },
        }
    if isinstance(node, LocalEnumerationSpec):
        return {
            **common,
            "type": "local_enumeration",
            "walker": node_dict(node.walker),
            "mutation_generator": node_dict(node.mutation_generator),
            "filters": [node_dict(child) for child in node.filters],
            "trajectories": node.trajectories,
            "iterations": node.iterations,
            "walk_time": node.walk_time,
            "trajectory_execution": node.trajectory_execution,
            "include_walk_points": node.include_walk_points,
        }
    raise TypeError(f"Unsupported graph declaration: {type(node).__name__}.")


def node_from_dict(value):
    """Decode only supported declarations; never import configuration targets."""
    raw = dict(value)
    operation = raw.pop("type")
    if operation == "step":
        return StepSpec(**raw)
    if operation == "loop":
        raw["steps"] = tuple(node_from_dict(child) for child in raw["steps"])
        return LoopSpec(**raw)
    if operation == "parallel":
        raw["branches"] = {
            name: tuple(node_from_dict(child) for child in steps)
            for name, steps in raw["branches"].items()
        }
        return ParallelSpec(**raw)
    if operation == "local_enumeration":
        raw["walker"] = node_from_dict(raw["walker"])
        raw["mutation_generator"] = node_from_dict(raw["mutation_generator"])
        raw["filters"] = tuple(node_from_dict(child) for child in raw["filters"])
        return LocalEnumerationSpec(**raw)
    raise ValueError(f"Unknown graph operation: {operation!r}.")


@dataclass(frozen=True, slots=True)
class PipelineSchema:
    """Immutable snapshot of a computation, without run or storage state."""

    steps: tuple[NodeSpec, ...] = ()
    autoencoder: AutoencoderSpec | None = None
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        if any(
            not isinstance(
                node, (StepSpec, LoopSpec, ParallelSpec, LocalEnumerationSpec)
            )
            for node in self.steps
        ):
            raise TypeError("PipelineSchema contains an unsupported graph node.")

    def walk(self):
        """Yield nodes with parent identities and unambiguous current positions."""

        def visit(nodes, parent, prefix):
            for index, node in enumerate(nodes):
                position = f"{prefix}.{index}" if prefix else str(index)
                yield node, parent, position
                if isinstance(node, LoopSpec):
                    yield from visit(node.steps, node.node_id, position + ".steps")
                elif isinstance(node, ParallelSpec):
                    for name, branch in node.branches.items():
                        yield from visit(
                            branch, node.node_id, position + ".branches." + name
                        )
                elif isinstance(node, LocalEnumerationSpec):
                    yield from visit((node.walker,), node.node_id, position + ".walker")
                    yield from visit(
                        (node.mutation_generator,),
                        node.node_id,
                        position + ".mutation_generator",
                    )
                    yield from visit(node.filters, node.node_id, position + ".filters")

        yield from visit(self.steps, None, "")

    def to_dict(self):
        """Return a complete independent JSON-compatible declaration."""
        encoder = self.autoencoder
        return {
            "version": self.version,
            "autoencoder": None
            if encoder is None
            else {
                "strategy": encoder.strategy,
                "model": encoder.model,
                "device": encoder.device,
                "parameters": thaw(encoder.parameters),
            },
            "steps": [node_dict(node) for node in self.steps],
        }

    @classmethod
    def from_dict(cls, value):
        """Load a versioned declaration while rejecting unknown schema fields."""
        if set(value) - {"version", "autoencoder", "steps"}:
            raise ValueError("Unknown PipelineSchema fields.")
        if value.get("version", 1) != 1:
            raise ValueError("Unsupported PipelineSchema version.")
        encoder = value.get("autoencoder")
        return cls(
            tuple(node_from_dict(node) for node in value["steps"]),
            AutoencoderSpec(**encoder) if encoder is not None else None,
        )

    def edit(self, target, *, replacement=None, insert=None, before=False):
        """Return an edited snapshot; identities are retained on replacement.

        :param target: Stable identity of an existing node.
        :param replacement: Replacement declaration, or ``None`` to remove.
        :param insert: Optional declaration inserted next to the target.
        :param before: Insert before rather than after the target.
        :raises KeyError: If the identity does not exist.
        :raises ValueError: If a required operator role would be removed.
        """
        found = [False]

        def update_sequence(nodes):
            result = []
            for node in nodes:
                if node.node_id == target:
                    found[0] = True
                    if insert is not None:
                        result.extend((insert, node) if before else (node, insert))
                    elif replacement is not None:
                        result.append(replace(replacement, node_id=target))
                    continue
                if isinstance(node, LoopSpec):
                    node = replace(node, steps=update_sequence(node.steps))
                elif isinstance(node, ParallelSpec):
                    node = replace(
                        node,
                        branches={
                            name: update_sequence(steps)
                            for name, steps in node.branches.items()
                        },
                    )
                elif isinstance(node, LocalEnumerationSpec):
                    roles = {}
                    for role in ("walker", "mutation_generator"):
                        values = update_sequence((getattr(node, role),))
                        if len(values) != 1 or not isinstance(values[0], StepSpec):
                            raise ValueError(
                                f"LocalEnumeration requires exactly one {role} component."
                            )
                        roles[role] = values[0]
                    node = replace(node, **roles, filters=update_sequence(node.filters))
                result.append(node)
            return tuple(result)

        updated = replace(self, steps=update_sequence(self.steps))
        if not found[0]:
            raise KeyError(target)
        return updated
