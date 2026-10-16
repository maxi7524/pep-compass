# Extending the library

A new implementation belongs under `steps/<category>/strategies/`. Export the
actual class or composition function in that category's explicit `__all__`.
Do not add registration decorators, manager branches, compatibility aliases or
model initialization to package imports.

```python
from pep_compass.steps.base import Step

class SequenceSuffix(Step):
    """Append a configured suffix without changing latent origins.

    :param suffix: Residues appended to every input sequence.
    """
    category = "filter"

    def __init__(self, suffix):
        self._configure({"suffix": suffix})
        self.suffix = suffix

    def _execute(self, batch, context):
        return batch.with_sequences([sequence + self.suffix for sequence in batch.sequences])
```

Use an importable module-level class so its constructor can be snapshotted and
rebuilt. Constructors validate portable arguments and call `_configure` with their
resolved values. Keep model loading in `_initialize(autoencoder)`; bind shared
resources through `Configurable.initialize`. Declare field contracts as class
metadata or override `contract()` when they depend on configuration. Select
compact graph information through `display_parameters` or `display_info()`.

`precompute(context)` resets per-run state. `prepare_call(batch, context)` refreshes
invocation state. `_execute` transforms the aligned batch. Never override
`Step.__call__`, which owns accounting and observation. Use the context RNG for
stochastic choices. Mark tensor dimensions after nontrivial transformations.

Internal strategies use their own explicit abstract algorithm interfaces and
`Configurable`; they do not inherit `Step` unless independently executed nodes.
Shared logic belongs in a real helper or base implementation, not empty family
markers. Data codec types are a fixed safe format, not a plugin registry. Custom
diagnostics should use supported dictionaries, arrays and tensors.

Run focused unit/workflow tests before the full suite. The direct API suites cover
ownership and edits, seeded numerical references, retention, replay, Hydra,
backend dispatch and lazy reads. Use small deterministic fixtures for interface
contracts; keep scientific numerical regressions and their original tolerances.
