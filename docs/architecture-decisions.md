# Architecture decisions

Construction, execution and experiment dispatch have separate contracts:

- `data_schemas` owns declarations, candidate data and run/result schemas.
- `registry` owns factory metadata, discovery, validation and model artifacts.
- `core.wrapper` presents the editable Python API and descriptions.
- `core.builder` resolves dependencies and validates the complete graph before
  constructing executable Steps.
- `optimization` owns scientific components, graph execution and result storage.
- `orchestrator` composes Hydra configs and dispatches independent wrapper runs
  through registered execution strategies.
- `analysis` reads common result tables and explicitly reconstructs retained
  histories. Historical archive reading is supported without reviving the old CLI.

The component registries do not depend on `core`. Default parameter resolution
belongs to registration metadata and preset factories. A builder receives a
prepared catalog; public wrapper construction explicitly loads built-in families.
Named models and their artifact requirements use one shared model catalog.

The editable schema and executable Step graph serve different lifecycle stages.
Every executable graph node follows Step's lifecycle; internal algorithm
strategies remain private to their scientific component. `prepare_call` owns
per-call preparation rather than external wrappers guessing component internals.

Use direct canonical module imports. Forwarding modules, old `runtime`/`data`
namespace aliases and the deprecated `describe_step` spelling are removed.
`describe_component` provides component metadata and placed-node details.

Result storage uses SQLite plus a JSON settings manifest with common memory and
disk semantics. Evaluated peptide archives differ from the terminal propagated
batch. Whole-batch budget semantics preserve observations already computed.
Retention controls reconstruction boundaries and component data independently
from the `base` archive. See the [wrapper](tutorial/wrapper.md) and
[orchestrator](tutorial/orchestrator.md) guides for the actual API.

PoGS remains a future scientific integration; no unimplemented workflow stub is
kept in the execution path. Concurrent experiment dispatch is separate from
parallel branches inside one computation graph.
