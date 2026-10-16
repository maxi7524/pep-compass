# Direct Python components

Import actual implementations from `pep_compass.steps.<category>` and shared
resources from `pep_compass.autoencoder`. Category `__init__.py` files explicitly
export their implementations under `strategies/`. Imports do not register objects
or load model weights. Ordinary preset functions live in `pep_compass.predefined`.

```python
from pep_compass import Pipeline
from pep_compass.autoencoder import HydrAMP
from pep_compass.steps.filters import CandidateSubset
from pep_compass.steps.oracles import Hydrophobicity

pipeline = Pipeline(autoencoder=HydrAMP(jacobian_mode="approx"))
node_id = pipeline.add_component(CandidateSubset(count=3))
pipeline.add_component(Hydrophobicity())
pipeline.describe()
pipeline.describe_component(node_id=node_id)
pipeline.replace_component(node_id, CandidateSubset(count=2))
compiled = pipeline.build()
result = pipeline.run(["FLYKWWIRIGRLKL"], seed=7)
print(result.reader.get_archive())
result.close()
```

`Pipeline` owns a `PipelineBuilder`; editing is implemented by that builder.
`PipelineSchema` stores immutable constructor snapshots, child ownership and
stable placement IDs. `build()` validates input-field dependencies, initializes
shared resources, creates fresh components and returns `ExecutablePipeline`.
Editing a builder does not change an already returned executable graph.
`compiled.run()` uses the same Runner and saver as `pipeline.run()`.

`Pipeline([...])`, incremental `add_component(...)`, and a standalone
`PipelineBuilder` are equivalent construction styles. `builder.loop(...)` is a
context manager for collecting a body; successful scope exit preserves child IDs.
A failed scope commits nothing. Required roles such as a local walker can be
replaced but cannot be removed. Insertions target ordered sequences.

`describe(display="display")` automatically renders formatted tables;
`display="pd"` returns a pandas DataFrame, and `display="raw"` returns records.
`describe_component(Class)` shows constructor defaults; passing a configured
instance also shows resolved internal stages. `technical=True` exposes field and
resource contracts. Constructors are the source of defaults; no parallel config
classes maintain a second set.

`show()` returns an inline SVG. Mermaid is textual diagram source requiring a
Mermaid renderer; SVG is the ready picture. `to_schema()` returns the immutable
schema, while `json`, `text`, `mermaid` and `svg` select portable representations.
Declared and compiled graphs share placement IDs. Candidate lineage is a separate
run-time structure, not a graph position.

Runner defaults are grouped under `execution`, `storage` and `results`. The new
notebook documents every setting and default. Per-run overrides do not modify
wrapper defaults. Oracle limits apply before a new evaluation batch; an entire
completed batch remains in the archive even if it crosses the threshold.

The run bundle contains `settings.json` and `results.sqlite`. The `base` profile
retains evaluated sequences and compact execution/iteration statistics. These
are independent options:

| Option | Default | Retained information |
| --- | --- | --- |
| `save_terminal` | `False` | Outgoing aligned batch, separate from the evaluated archive |
| `save_observation_history` | `False` | Repeated evaluations, not only latest sequence/objective values |
| `save_lineage` | `False` | Archive candidate and parent identities |
| `component_data` | `none` | `summary`: display metadata; `full`: metadata and aligned output fields |
| `reconstruction` | `none` | `boundaries`: input/state/RNG replay checkpoints; `trajectories`: also walker outputs |
| `flush_every` | `1` | Transaction interval, without sampling or dropping rows |

`Snapshot("label")` explicitly retains a batch at its graph position.
`RunResult.save(path)` exports already retained data. Memory and persistent storage
use the same SQL schema. `AnalysisReader` opens one bundle; `ExperimentReader`
discovers collections and exposes lazy projected scans through the shared table
contract. Missing retained data raises `DataUnavailableError`.

Replay is explicit: `reader.reconstruct_local_enumeration(checkpoint_id)` builds
only that fragment, restores saved state and RNGs, and verifies ordered output
against the recorded digest. It requires the recorded package source and model
identity. Locality analyses additionally require trajectories and a retained
candidate batch. Old CSV/PT readers and declaration formats are removed.

The full runnable sequence is in [wrapper_new.ipynb](wrapper_new.ipynb).
