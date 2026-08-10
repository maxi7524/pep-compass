# Architecture Decisions

This document records *why* PepCompass is split the way it is, what was
deliberately deferred, and the currently known open issues. It condenses the
implementation discussion that drove the `core`/`optimization`/`runtime`/
`autoencoder`/`data`/`analysis` restructuring into decisions and current
status — read [Technical Architecture](technical-architecture.md) for how
the result actually works, and [`developer/przeniesienie
modeli.md`](developer/przeniesienie%20modeli.md) for the file-by-file
old→new mapping used during that migration.

## Package split rationale

The starting problem was four responsibilities mixed into two packages:
describing/validating an experiment, building an executable optimisation
program, running one optimisation and a whole experiment plan, and reading
persisted results back for analysis. The chosen split makes each
responsibility a package with a one-directional dependency:

```text
runtime  ──►  core  ──►  optimization  ──►  data, autoencoder
                                 ▲
analysis ──► data (result_schema, dataset)   (never → runtime/core/optimization)
```

- **`optimization`** is the engine and the science. It must run from plain
  Python with a manually constructed `PepCompassPipeline` — no `core`, no
  YAML, no CLI — so a component can be tested and used without the
  configuration layer around it.
- **`core`** builds and validates only. `PipelineBuilder` turns a parsed
  `PipelineSpecification` plus an already-constructed autoencoder into a
  `PepCompassPipeline`. It does not read YAML, does not know about the CLI,
  and does not persist anything.
- **`runtime`** is everything from a configuration file to persisted
  results: loading, validation, input/grid/plan materialisation, workflow
  selection, backend execution, and output writing. It is the layer that
  changes when you add a new way to *run* PepCompass (a backend, a
  workflow), not a new way to *compute*.
- **`autoencoder`** (not `models` — see below) and **`data`** are shared,
  dependency-light foundations used by both `optimization` and `runtime`.
- **`analysis`** only ever depends on `data` (the result schema and the
  logical dataset view) and its own `reader`. It cannot import `runtime` or
  `optimization` — it consumes what they produced, after the fact.

## Naming decisions

### `autoencoder`, not `models`

A generic top-level `models` package was rejected: it re-creates the kind of
undifferentiated bucket this restructuring was meant to remove. The package
is named for what it concretely is — an autoencoder contract, its geometry
operations, and its registered implementations — not a place for arbitrary
future model types.

### `method` vs `model`

`autoencoder.method` selects the *implementation* (`hydramp`);
`autoencoder.model` selects a *named checkpoint variant* of that
implementation (`article_25`). This was a deliberate two-axis split so a
future fine-tuned checkpoint (for example a locality-fine-tuned HydrAMP
variant) can be added as a new registry entry under `model` without writing
a new `method` strategy. See `autoencoder/registry.py`/`factory.py` and
[Developer Guide](developer-guide.md#adding-an-autoencoder).

### `core` builds; it does not run

An earlier proposal collected CLI, planning, backends and the engine under
one `core`. That was rejected because it made `core` mean "everything
executable," which is exactly the ambiguity this restructuring set out to
remove. `core` is deliberately narrow: specification → validated,
constructed `PepCompassPipeline`, nothing else. Everything about *running*
that pipeline (once, or as a planned experiment) belongs to `optimization`
(for the single run) or `runtime` (for the experiment).

### Latent geometry folded into `autoencoder`

The former `core/latent_geometry.py` (Jacobian SVD / tangent-space
decomposition) was moved into `autoencoder/geometry.py`. It is not a
pipeline-construction concern — it is an operation derived directly from the
decoder Jacobian, used by SORBES and MUTANG, and belongs next to the
autoencoder contract that produces that Jacobian.

## Resource control

Two independent, complementary mechanisms were designed and are now
implemented as `optimization/stability_estimation/`:

- **Static estimation** (`estimation.py`) — per-node candidate-count and
  byte-size bounds computed from declared cardinality behaviour, run during
  `dry-run` before any model is constructed. A node whose output cannot be
  bounded from configuration alone (a data-dependent filter's pass-through
  rate, for example) is reported as an explicit unknown rather than a
  guessed number, so a false sense of precision is never presented.
- **Runtime monitoring** (`monitoring.py`) — actual process RSS and CUDA
  allocator sampling at step boundaries (not inside per-operation tensor
  code, to avoid forcing GPU synchronisation mid-operation), written to
  `tracking/stability.csv`.

See [Technical Architecture](technical-architecture.md#stability-estimation)
for the implemented mechanism.

## Deferred work

These were discussed as follow-on improvements to the resource-control
design above and are **explicitly not implemented** — documented here so
they are not mistaken for silently dropped scope:

- **Field-lifetime pruning**: dropping `CandidateBatch` fields once no later
  step needs them (an explicit liveness analysis over declared
  `requires`/`provides`/`retains` field sets), instead of every step
  implicitly propagating every field it received.
- **Candidate-viability selectors**: an explicit, configurable filter family
  for dropping candidates that provably cannot improve the objective (trust-
  region exit, Pareto dominance, stagnation), kept deliberately separate from
  memory management — candidate removal is an algorithmic decision with
  scientific consequences, not a housekeeping operation, and must never
  happen implicitly.

## PoGS

PoGS takes a different input shape than the composable pipeline and was
therefore deliberately **not** inserted into the existing `Loop`/`Flow`
graph. The agreed integration shape is a second, independent
`RuntimeWorkflow` implementation (`runtime/workflows/base.py`'s
`RuntimeWorkflow` protocol, the same contract `ComposableWorkflow`
implements), selected the same way at the runtime boundary rather than
branching inside the engine.

**Current status**: `runtime/workflows/pogs.py` declares `PogsWorkflow` and
unconditionally raises `NotImplementedError`; no `pep-compass` command
selects it yet. See [User Guide](user-guide.md#pogs) for the placeholder and
[Developer Guide](developer-guide.md#adding-a-workflow) for how to implement
it.

## Known issues (verified against the current tree)

The pre-restructuring code review (`tmp/REVIEW.md`, since superseded by this
document and deleted) flagged several items against an older package layout.
Re-checked against the current tree:

- **Resolved** — duplicate class-and-factory registration under the same
  manager key (`sorbes`, `mutang`): the current
  `walkers/strategies/__init__.py`/`mutation_generators/strategies/__init__.py`
  each register exactly one factory per name.
- **Resolved** — the old ablation walker's constructor mismatch: SORBES's
  `position_update` sub-strategies (`main`/`article`/`without_acceleration`)
  now share `MainPositionUpdate`'s constructor with no signature drift.
- **Resolved** — empty filter subpackages (`biology/`, `constraints/`,
  `latent_geometry/`, `sequence_geometry/` with no implementation): the
  `filters/direct/{constraints,controls,optimization,structural}/` and
  `filters/ranked/{scoring,selection}/` split is populated and is the
  current, intentional structure (see
  [Developer Guide](developer-guide.md#adding-a-filter)).
- **Resolved** — `oracles/strategies/README.md` claiming EIPred and
  MBC-Attention were not yet wired in: both are registered
  (`oracles/strategies/__init__.py`).
- **Still open** — `Parallel` merge policies `interleave`, `select_best` and
  `weighted_sample` are declared (`MergeMethod` literal) but raise
  `NotImplementedError` (`optimization/engine/operations/parallel/merge.py`).
- **Downgraded, not fully resolved** — the `field_eps` inconsistency:
  `Autoencoder.field_derivative` now consistently uses `self.field_eps`
  (the original bug there is fixed), but
  `Autoencoder.get_ambient_covariant_derivative` still takes an independent
  local `eps: float = 0.1` parameter unrelated to `field_eps`. It currently
  has **no callers** anywhere in the codebase, so this is dead code rather
  than an active bug — reconcile the `eps` parameter with `field_eps` before
  the method is used.
- **New, found during this pass** — `MainBoundary.__call__`
  (`walkers/strategies/sorbes/boundary/main.py`) contains an explicit
  `REMARK`/`TODO`: it provides no computable `W_kappa` boundary projection;
  an "article" boundary variant is pending on `W_kappa` being fully
  specified. `position_update/article.py` similarly has an open TODO
  confirming whether its Gamma term equals the acceleration term used by
  `main`. Both are open scientific questions for the walker's author, not
  implementation bugs.
- Everything else in the superseded review (old-path bug locations, a
  duplicated-`materialize_variants`-call performance note in a
  `composable.py` that no longer exists under that name, an unconfirmed
  `should_start_tracking` heuristic) referenced files or structures that no
  longer exist under the current package layout and were not re-verified;
  they are not carried forward as current facts.
