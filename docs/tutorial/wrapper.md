# Python pipeline wrapper

The wrapper assembles declarations, builds the existing computation operators and
executes one graph. It does not use YAML, Hydra or experiment dispatch. The
companion [notebook](wrapper.ipynb) executes a bounded example with bundled
HydrAMP weights and the local hydrophobicity oracle. Install the project with a
CPU or CUDA computation extra before executing it; model weights must be present.

## Inspect available components

```python
from pep_compass import Pipeline, Step, Loop, Parallel, LocalEnumeration

pipeline = Pipeline()
pipeline.describe()  # formatted display, no extra display(...) call
pipeline.describe(component="filter")
pipeline.describe_component(component="walker", strategy="sorbes")
choices = pipeline.describe(display="pd")
raw = pipeline.describe_component("walker", "sorbes", display="raw")
```

All description methods accept `display="display"` (default), `"pd"` or `"raw"`.
The default displays formatted tables in a notebook and prints readable tables in
plain Python, returning `None` to avoid duplicate notebook output. `"pd"` returns
a pandas DataFrame; `"raw"` returns the original dictionaries/lists for programmatic
use. A detailed DataFrame uses a `section` column for overview, parameters, models,
internal strategies; it does not discard nested
strategy parameters. Notebook display folds internal strategy sections. The catalog columns are
`component`, `strategy`, `description` and `internal_stages`. Detailed descriptions
render Sphinx prose and parameter help rather than showing raw `:param:` fields.

Set `technical=True` on `describe` or `describe_component` to inspect execution
contracts. `services` are shared objects injected by the builder, such as the
autoencoder; these are not user parameters. `input_fields` are required candidate
fields, `output_fields` are produced fields, and `optional_inputs` are read only
when present. These fields are hidden from the default user-facing tables.

`describe()` lists available choices, not the current graph. Use
`describe_component(component=..., strategy=...)` for a strategy's parameters,
defaults and internal choices. After adding a node,
`describe_component(node_id=node.node_id)` additionally includes its resolved
configuration, position, parent and adjacent nodes in the same sequence.
`describe_component` is now an instance method (the previous class helper only
saw built-ins). It uses the instance's catalog, including custom registrations;
use `Pipeline().describe_component(...)` for an independent built-in lookup.
`Pipeline.list_components(...)` lists
built-in choices and accepts the same display modes.

`pipeline.add_autoencoder(strategy="hydramp", model="article_25",
parameters={"jacobian_mode": "approx"})` configures a shared resource. Weights are
loaded during `build()`. An already initialized encoder can instead be supplied
with `Pipeline(autoencoder=encoder)` or `pipeline.add_autoencoder(encoder)`.

## Compose the graph

All syntax variants produce the same immutable node declarations. Adding a step
returns its declaration and stable `node_id`; `add(spec)` returns the pipeline.
`add_component(...)` is an alias of `add_step(...)`, including inside scopes.

```python
pipeline.add_autoencoder(parameters={"jacobian_mode": "approx"})
with pipeline.add_loop(iterations=2, name="optimization") as loop:
    loop.add_local_enumeration(
        trajectories=2,
        iterations=1,
        include_walk_points=False,
        walker=Step.create(component="walker", strategy="sorbes"),
        mutation_generator=Step.create(
            component="mutation_generator", strategy="mutang",
            parameters={"maximum_candidates": 4},
        ),
        filters=[Step.create(component="filter", strategy="deduplicate")],
    )
    loop.add_step("filter", "candidate_subset", parameters={"count": 3})
    loop.add_step("oracle", "hydrophobicity")
    loop.add_step("filter", "candidate_subset", parameters={
        "count": 1, "mode": "highest",
        "score_field": "oracle.hydrophobicity.score",
    })
```

Alternatively, create declarations separately or directly inside `Pipeline.create`:

```python
unique = Step.create(component="filter", strategy="deduplicate")
body = Loop.create(iterations=2, steps=[unique])
separate = Pipeline.create(steps=[body])
inline = Pipeline.create(steps=[Loop.create(
    iterations=2,
    steps=[Step.create(component="filter", strategy="deduplicate")],
)])
```

These last two examples only declare a graph; attach an encoder before running.
A loop feeds its entire output into the next iteration. Local enumeration keeps
walker trajectories separate from the emitted mutation pool. Exactly one of
`iterations` and `walk_time` bounds local enumeration.

Named branches use `with pipeline.add_parallel() as parallel` and
`with parallel.add_branch("name") as branch`. Every scope commits atomically on
successful exit. Failed scopes do not append an incomplete operator. Branches
share the run state and budget; `execution="concurrent"` retains the backend's
ordered access to shared resources and is not independent process execution.

## Inspect and edit

```python
schema = pipeline.to_schema()
text = pipeline.to_schema(format="text")
mermaid = pipeline.to_schema(format="mermaid")
pipeline.show()  # inline SVG; no Graphviz executable required
```

Views show identities, positions, structure, resource relationships and selected
parameters. SVG is a rendered vector image; Mermaid is textual diagram source
for a Mermaid-capable renderer (`print(pipeline.to_schema("mermaid"))`). JSON is
the machine-readable declaration, and text is a readable structure report.
Entering a nested body points right, returning to the parent points left, and
dashed feedback connects to the actual loop-body input or private walker.
Compact labels round floating-point text for readability; hover tooltips and
raw schemas retain exact configured values. Registered `display_parameters` specify the default compact fields;
a built component may override `Step.display_info()` from `optimization.engine.execution.step` for its own view.
`show_pipeline_graph()` aliases `show()`.

Use `insert_step(spec, before=id)` or `after=id`, `remove_step(id)` and
`replace_step(id, spec)`. Replacement preserves the target identity. Removing a
container removes its owned subgraph. Position indices may change; identities do
not. Each edit invalidates the previous build. Editing during execution and
building inside an open scope are forbidden.

`to_schema()` returns an independent immutable snapshot. `to_schema("json")` and
`save_schema(path)` export declarations, not live models. Restore them through
`Pipeline.from_schema(schema_or_json)`. Exported configuration contains portable
values; executable Python objects belong in registrations or injected services.

## Configure and execute

```python
pipeline.set_runner_settings(
    execution={"seed": 7, "oracle_calls": 6},
    storage={"backend": "memory", "profile": "base"},
    results={"return_mode": "result"},
)
pipeline.build()
result = pipeline.run(sequences=["FLYKWWIRIGRLKL"])
result.archive
result.summary
```

`run()` builds when needed. Every run receives fresh counters, observations, RNG
and context state. Oracle adapters reset their public evaluation budget and
observation cache between runs while retaining their model. Custom components
must keep run-dependent state in the context or reset it in `precompute()`.
`prepare_call()` runs before every invocation; overriding `__call__` is forbidden.
Automatic construction uses the effective run seed, including constructor RNGs.
An explicit `build()` uses configured execution settings; set its seed beforehand
or pass `build(execution={"seed": 7})`. Subsequent runs reuse constructed resources.

Per-run `execution`, `storage`, `results` mappings and the convenient `seed=`
argument override defaults without changing them. Device changes require
rebuilding resources. A single-graph runner accepts `max_workers=1`; experiment
parallelism is outside this API. Seeds, budgets and retention are separate from
loop iteration counts in the graph.

The oracle budget counts evaluated sequences, not Python calls. Complete batches
may cross the soft threshold; completed observations are never truncated.

## Retention and persistence

`base` retains the latest evaluation of each `(objective, sequence)` over the
whole run. Later candidate selection cannot delete an earlier evaluation. Ties
preserve first-observation order. Multiple objectives have separate archives.
The exact terminal batch is different from the evaluated archive. A graph with
no oracle has an empty evaluated archive.

Additional independent settings are:

| Setting | Meaning |
| --- | --- |
| `reconstruction="none/boundaries/trajectories"` | No reconstruction, input/RNG boundaries, or boundaries plus walker points |
| `component_data="none/summary/full"` | No component records, compact information, or retained output fields |
| `save_observation_history=True` | Every evaluation, including repeated sequences |
| `save_lineage=True` | Archive provenance identifiers |
| `save_terminal=True` | Exact terminal CandidateBatch |
| `flush_every=N` | Commit every N completed operator-iteration events; never sample away data |

Nested operators each emit their own iteration events. A batched walker advance
is one event with a trajectory count, not N independent flush events. The final
incomplete buffer is committed on completion or handled failure.

```python
stored = pipeline.run(
    sequences=["FLYKWWIRIGRLKL"],
    storage={
        "backend": "sqlite", "path": "/tmp/my-peptide-run",
        "reconstruction": "trajectories", "save_terminal": True,
    },
)
```

The output directory must not already contain another run bundle:

```text
my-peptide-run/
├── settings.json
└── results.sqlite
```

JSON contains the resolved graph, inputs, runner settings, actual model artifact
hashes and code/environment provenance. SQLite contains run status, evaluated
peptides, timing records and enabled optional categories. Tensor payloads use raw
binary blocks with dimensions, dtype and integrity checks; masks and shared
mutation-pool containers survive roundtrips. No arbitrary object unpickling is
used. Model weights are referenced, not copied into every bundle.

The memory backend uses the same relational schema in an in-memory SQLite
database. `result.save(path)` exports only the information actually retained.
`pipeline.save_result(path)` exports its last completed run.

## Read and reconstruct

```python
from pep_compass.optimization.saving import AnalysisReader

reader = AnalysisReader(stored.path)
archive = reader.get_archive(objective="hydrophobicity")
best = reader.get_best_candidate(objective="hydrophobicity")
terminal = reader.get_final_batch()  # requires save_terminal

for chunk in reader.tables["evaluated_peptides"].scan(
    ["sequence", "score"], chunk_size=1000
):
    display(chunk)
```

Reader construction does not load all tables or run models. Materialization is
explicit. Missing retention raises `DataUnavailableError` rather than returning
invented history. `results.return_mode` selects `result` (default), `reader`, or
`path`; the latter requires SQLite and returns the directory containing both
files. Returned results own their reader lifetime; call `result.close()` or
`reader.close()` when finished.

With reconstruction enabled, read `reader.tables["checkpoints"]` and call
`reader.reconstruct_local_enumeration(checkpoint_id)`. It builds the selected
fragment, restores the boundary batch, state, scope and RNGs, and returns a
`ReplayResult`. Check `replay.matches`: it compares ordered output sequences and
count with the original digest. Multiple LocalEnumeration nodes are addressed
by their persisted identities. For an originally injected encoder, pass it
explicitly using `autoencoder=...`; custom strategies require their catalog.

Exact replay requires matching package sources, model weights and the compatible
recorded environment. Custom dataclass payloads can register a codec through
`@register_payload("stable_name")` from `optimization.saving.serialization`.
A component with custom scientific payloads needs such a codec; unsupported
payloads fail explicitly. Prepared run-dependent values needed for boundary replay
must live in `context.step_state`; arbitrary component attributes are not
implicitly snapshotted. Replay is explicit and can be expensive.
Retained history can be read without recomputation; replay reconstructs supported
boundaries. General interrupted-run continuation is not provided by this API.

Computational exceptions propagate. SQLite runs record a failed status and the
last committed information; they are never marked completed before finalization.
Historical CSV/JSON/PT archives remain readable and replayable through
`analysis.reader`; the former runtime executor and import paths are removed. Canonical imports
use `data_schemas` and the thematic registry packages; obsolete aliases are removed.

## Register and reuse

Use a central family's decorator, for example
`registry.components.families.filters.FilterManager.register`, optionally with a dataclass `config_type`,
validator, normalizer, field contracts and `display_parameters`. Registration
stores metadata and a factory; it does not construct the component. A new
strategy does not require a manager branch. `pipeline.describe()` exposes these
same definitions. `predefined/default_params.py` selects default resources, while
component configuration types own parameter defaults.

`Pipeline.from_predefined("lebo", device="cpu")` returns an editable, bounded
preset based on the repository reference. Its explicit mutation allocation cap
is an operational addition; this is not a claim of publication reproduction.

## Presets and experiment orchestration

`Pipeline.from_predefined("lebo", ...)` delegates to the decorated preset registry.
Explicit imports such as `from pep_compass.predefined.lebo import create` are also
supported. See the [Hydra orchestration guide](orchestrator.md) for group overrides,
independent runs and server-specific execution strategies.
