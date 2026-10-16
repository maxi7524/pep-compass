"""Transactional nested builders over the same immutable declaration model."""

from pep_compass.core.wrapper.declarations import (
    LocalEnumeration,
    Loop,
    Parallel,
    create_step,
)
from pep_compass.data_schemas.graph import (
    LocalEnumerationSpec,
    LoopSpec,
    ParallelSpec,
    StepSpec,
)


class SequenceBuilder:
    """Share collection operations between a pipeline, loop and branch."""

    def add(self, spec):
        """Append an existing declaration without constructing its component."""
        if not isinstance(
            spec, (StepSpec, LoopSpec, ParallelSpec, LocalEnumerationSpec)
        ):
            raise TypeError(
                "Expected a Step/Loop/Parallel/LocalEnumeration declaration."
            )
        self._append(spec)
        return self

    def add_step(self, component, strategy, name=None, parameters=None):
        """Create and append a declaration; return its stable node identity."""
        self._owner._ensure_editable()
        spec = create_step(
            component=component,
            strategy=strategy,
            name=name,
            parameters=parameters,
            catalog=self._owner.catalog,
        )
        self._append(spec)
        return spec

    add_component = add_step

    def add_loop(self, *, iterations, name=None):
        """Collect a loop body and commit it atomically on successful exit."""
        return LoopBuilder(self, iterations=iterations, name=name)

    def add_parallel(self, *, name=None, execution="sequential", merge="concatenate"):
        """Collect named branches and commit one parallel operator."""
        return ParallelBuilder(self, name=name, execution=execution, merge=merge)

    def add_local_enumeration(self, **settings):
        """Append a role-explicit local enumeration; it is not a generic scope."""
        self._owner._ensure_editable()
        spec = LocalEnumeration.create(**settings)
        self._append(spec)
        return spec


class _Scope:
    def __init__(self, parent):
        self.parent = parent
        self._owner = parent._owner
        self._entered = False
        self._closed = False

    def __enter__(self):
        self._owner._ensure_editable()
        if self._entered or self._closed:
            raise RuntimeError("Builder scopes cannot be entered twice.")
        self._entered = True
        self._owner._open_scopes += 1
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                self.parent._append(self.to_spec())
        finally:
            self._closed = True
            self._owner._open_scopes -= 1
        return False

    def _check_active(self):
        self._owner._ensure_editable()
        if not self._entered or self._closed:
            raise RuntimeError("Use a nested builder only inside its with block.")


class LoopBuilder(_Scope, SequenceBuilder):
    """Collect one ordered loop body without editing its parent prematurely."""

    def __init__(self, parent, *, iterations, name):
        super().__init__(parent)
        self.iterations, self.name, self.steps = iterations, name, []

    def _append(self, spec):
        self._check_active()
        self.steps.append(spec)

    def to_spec(self):
        """Return the same LoopSpec produced by Loop.create."""
        return Loop.create(iterations=self.iterations, steps=self.steps, name=self.name)


class ParallelBuilder(_Scope):
    """Collect branch scopes without a second graph-construction engine."""

    def __init__(self, parent, *, name, execution, merge):
        super().__init__(parent)
        self.name, self.execution, self.merge = name, execution, merge
        self.branches = {}

    def add_branch(self, name):
        """Open an independent branch sequence with a unique local name."""
        self._check_active()
        if not isinstance(name, str) or not name or name in self.branches:
            raise ValueError("Branch names must be non-empty and unique.")
        return BranchBuilder(self, name)

    def to_spec(self):
        """Return the same ParallelSpec produced by Parallel.create."""
        return Parallel.create(
            branches=self.branches,
            name=self.name,
            execution=self.execution,
            merge=self.merge,
        )


class BranchBuilder(_Scope, SequenceBuilder):
    """Commit one branch only after successful completion of its scope."""

    def __init__(self, parent, name):
        super().__init__(parent)
        self.name, self.steps = name, []

    def _append(self, spec):
        self._check_active()
        self.steps.append(spec)

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                if self.name in self.parent.branches:
                    raise ValueError(f"Duplicate branch: {self.name!r}.")
                self.parent.branches[self.name] = tuple(self.steps)
        finally:
            self._closed = True
            self._owner._open_scopes -= 1
        return False
