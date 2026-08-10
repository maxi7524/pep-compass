# Analysis Guide

`pep_compass.analysis` reads results persisted by `runtime` (see
[User Guide](user-guide.md#output-files)) and computes derived tables and
plots. It is a **separate, read-only consumer**: it never imports
`runtime`, `core`, or the execution engine, and it cannot change what a run
produced — only interpret it after the fact. If you are looking for how a
pipeline runs, see [Technical Architecture](technical-architecture.md).

## `analysis.reader`

`analysis.reader` discovers, selects and lazily reads tracking output
without loading whole tables into memory unless asked to. A complete,
runnable walkthrough of every entry point below lives in
[`assets/experiments/example_notebook..ipynb`](../assets/experiments/example_notebook..ipynb)
— open it rather than re-deriving these examples by hand.

### `ExperimentReader`

```python
from pep_compass.analysis.reader import ExperimentReader

reader = ExperimentReader("experiments/results/reference_lebo/run")
reader.experiments   # discovered experiments, tables not loaded yet
reader.runs           # every discovered ExperimentRun (run/grid/tracking-path identity + metadata)
reader.dataset         # ExperimentDataset: schema_version, runs, lazy per-table handles
reader.cached_analyses()
```

`ExperimentReader(path)` accepts a collection root, an experiment directory,
or a single tracking-run directory, and discovers runs from whichever
on-disk layout is present (the current versioned `variants/*/runs/*/
result.json` layout, or an older `run_manifest.csv`/`tracking_metadata.json`
layout) without loading any tracking table. It also opens a
`.pep_compass_analysis.sqlite` cache next to the opened root (see
[`MetricsStore`](#metricsstore)).

### `ExperimentSelection`

```python
selection = reader.select(seeds=[1234], peptides=["FLYKWWIRIGRLKL"])
selection = selection.rows(candidate_index=0)   # deferred row filter

selection.count_rows("candidates")
selection.paths("candidates")
selection.collect("candidates")                 # materialize
for chunk in selection.scan("trajectory_points", chunk_size=1000):
    ...
selection.specification()                        # stable cache key: run ids, paths, filters, file fingerprints
```

`select(...)` filters *runs* by manifest/result metadata (`experiments`,
`methods`, `grid_ids`, `peptides`, `seeds`, `parameters`); it is immutable
and chainable. `.rows(**filters)` adds a deferred *row*-level equality/
membership filter, applied only to tables whose source columns contain the
filter key, and only when a table is actually scanned. Logical table names
are registered in `analysis.reader.data_schemas.registry.SCHEMAS` (`steps`,
`candidates`, `trajectory_points`, `local_enumerations`, `stability`, plus
BO-loop tables not produced by the composable-pipeline reference runs).

### `RunReplay`

```python
replay = reader.replay("run_00000")
replay.result                 # terminal run metadata
replay.resolved_configuration # exact configuration this run used
replay.final_candidates       # sequences + every oracle.*.score column
replay.final_latents           # (N, D) tensor
replay.trajectory_points, replay.trajectory_latents
points, latents = replay.trajectory(trajectory_id)

sequences, latents = replay.local_enumeration_input(execution_id)
verification = replay.verify_local_enumeration(execution_id, reconstructed_sequences)
verification.matches
```

`reader.replay(run_id)` resolves one unambiguous run and exposes its final
results, exact resolved configuration, and tracking checkpoints, including
the ones needed to reconstruct one `local_enumeration` execution's exact
input (see
[Technical Architecture](technical-architecture.md#local-enumeration)).
`local_enumeration_input`/`verify_local_enumeration` only return the stored
input and compare a *reconstruction* you supply against the stored count and
SHA-256 digest — they do not recompute the pipeline themselves. Producing a
real reconstruction means rebuilding the same `walker`/`mutation_generator`/
`filters` `Step`s from `resolved_configuration["pipeline"]` through
`core.builder.PipelineBuilder` and running them against that input; see
`tests/analysis/reader/test_reader.py::test_reader_validates_local_enumeration_replay_checkpoint`
for the pattern (there built from mock components).

### `MetricsStore`

```python
reader.metrics.get_or_compute_metrics("apex", sequences, evaluator, metric_version="1")
reader.metrics.put_analysis("my_analysis", frame, metadata={}, selection=selection.specification(), parameters={})
reader.metrics.get_analysis("my_analysis", selection=selection.specification(), parameters={})
reader.cached_analyses()   # == reader.metrics.list_analyses()
```

`reader.metrics` is a `MetricsStore` backed by
`<root>/.pep_compass_analysis.sqlite`, with two independent caches: per-
sequence metric values (`put_metrics`/`get_metrics`/`get_or_compute_metrics`,
useful for caching an expensive oracle call across notebooks) and exact
aggregate analysis results keyed by a stable hash of the selection and
parameters (`put_analysis`/`get_analysis`/`list_analyses`/`load_analysis`).

## `analysis_types`, `resampling`, `visualization`

- **`analysis_types/locality`** (`LocalityAnalysis`, `parameter_selection`,
  and per-component analyses in `sorbes.py`/`mutang.py`/
  `latent_geometry.py`) computes locality-experiment summaries directly from
  an `ExperimentSelection`, streaming over chunks (`_streaming.py`:
  `ProgressReporter`, `Reservoir`, `RunningMoments`) rather than materialising
  full tables, and returns an `AnalysisResult` (`data`, `metadata`,
  `diagnostics`).
- **`resampling`** (`bootstrap.py`: `ClusterSampler`, `RowSampler`,
  `BootstrapEngine`) provides resampling utilities for uncertainty estimates
  over analysis outputs.
- **`visualization`** (`locality.py`: `LocalityVisualizer`, `theme.py`:
  `PlotTheme`) renders `AnalysisResult` tables produced by `analysis_types`.
- **`experiment.py`** (`VisualizationCollection`) is a thin composition
  facade wiring one visualizer per registered analysis type.

This layer is under active revision — `src/pep_compass/analysis/README.md`
tracks open restructuring notes for the visualization/analysis split. Treat
new analyses the same way `locality` is built: accept an `ExperimentSelection`
or `ExperimentDataset`, never a raw path or an ad hoc dict, and return an
`AnalysisResult`.
