# Reading and analyzing runs

The supported bundle is `settings.json` plus `results.sqlite`. Both memory and disk
storage use the same relational schema and lossless tensor payload codec. The
reader never regenerates missing data implicitly.

```python
from pep_compass.optimization.saving import AnalysisReader
from pep_compass.analysis.reader import ExperimentReader

with AnalysisReader("results/example") as reader:
    archive = reader.get_archive()
    summary = reader.summary
    for frame in reader.tables["evaluated_peptides"].scan(["sequence", "score"], chunk_size=1000):
        print(frame)

collection = ExperimentReader("results")
selected = collection.select(seeds=[7])
frame = selected.collect("evaluated_peptides", columns=["sequence", "score", "seed"])
```

`base` retains all completed evaluated sequence/objective pairs and compact
statistics. It does not imply an outgoing tensor batch. `save_terminal=True`
retains that separate aligned batch. Observation history, evaluated lineage,
component diagnostics and reconstruction are independent retention options.
`Snapshot(name)` explicitly captures a batch at a graph position.

`get_checkpoints()` lists retained boundaries. `get_boundary_input(execution_id)`
loads the original input batch. `get_trajectories()` returns aligned walker point
metadata and latent positions, with zero-based walker call indices per trajectory.
`reconstruct_local_enumeration(checkpoint_id)` explicitly rebuilds and verifies one
fragment without rerunning downstream oracles. Missing data raises
`DataUnavailableError`.

Locality analyses consume canonical batch fields and trajectory payloads, not old
CSV/PT layouts. Enable trajectory reconstruction and retain the candidate batch
needed by the analysis. Candidate and parent identities remain distinct from
stable computation-node IDs. TableHandle projections and bounded scans provide
one read contract for external analyses; MetricsStore caches completed analyses
using run/selection/parameter fingerprints.
