# Technical Architecture

This document describes how PepCompass constructs and executes composable
optimisation graphs. 

> Remarks:
> - Public configuration belongs to the [User Guide](user-guide.md), 
> - modification procedures belong to the [Developer Guide](developer-guide.md).

## Table of Contents

- [Technical Architecture](#technical-architecture)
  - [Table of Contents](#table-of-contents)
  - [System Architecture](#system-architecture)
    - [Core](#core)
    - [Experiment System](#experiment-system)
    - [Optimisation Engine](#optimisation-engine)
    - [Strategy Modules](#strategy-modules)
    - [Tracking and Analysis](#tracking-and-analysis)
  - [Runtime Construction](#runtime-construction)
    - [Construction Flow](#construction-flow)
    - [Configuration Loading](#configuration-loading)
    - [Configuration Validation](#configuration-validation)
    - [Strategy Resolution](#strategy-resolution)
    - [Execution-Graph Construction](#execution-graph-construction)
    - [Runner Construction](#runner-construction)
  - [Pipeline Elements](#pipeline-elements)
    - [Candidate Data Model](#candidate-data-model)
      - [Candidate](#candidate)
      - [Candidate Batch](#candidate-batch)
      - [Batch Fields](#batch-fields)
      - [Candidate Selection](#candidate-selection)
      - [Candidate Expansion](#candidate-expansion)
      - [Batch Concatenation](#batch-concatenation)
    - [Step Contract](#step-contract)
      - [Precomputation](#precomputation)
      - [Iteration Preparation](#iteration-preparation)
      - [Execution](#execution)
      - [Automatic Tracking](#automatic-tracking)
    - [Optimisation State](#optimisation-state)
      - [Oracle Observations](#oracle-observations)
      - [Trust-Region State](#trust-region-state)
      - [Resource Counters](#resource-counters)
      - [Stop Requests](#stop-requests)
  - [Pipeline Construction and Execution](#pipeline-construction-and-execution)
    - [Execution Graph](#execution-graph)
      - [Flow Execution](#flow-execution)
      - [Loop Execution](#loop-execution)
      - [Parallel Execution](#parallel-execution)
      - [Batch Merging](#batch-merging)
    - [Latent-Space Trajectories](#latent-space-trajectories)
      - [Initial Encoding](#initial-encoding)
      - [SORBES Execution](#sorbes-execution)
      - [MUTANG Expansion](#mutang-expansion)
      - [Latent-Origin Propagation](#latent-origin-propagation)
      - [Repeated SORBES–MUTANG Execution](#repeated-sorbesmutang-execution)
      - [Candidate-Pool Formation](#candidate-pool-formation)
    - [Replicated Trajectories](#replicated-trajectories)
      - [Replica Materialisation](#replica-materialisation)
      - [Replica Identification](#replica-identification)
      - [Replica Random States](#replica-random-states)
      - [Replica Result Merging](#replica-result-merging)
  - [Runtime Implementation](#runtime-implementation)
    - [Automatic Execution Selection](#automatic-execution-selection)
    - [Batched GPU Execution](#batched-gpu-execution)
    - [Concurrent Execution](#concurrent-execution)
    - [Sequential Fallback](#sequential-fallback)
    - [Encoder-Decoder Reuse](#encoder-decoder-reuse)
    - [Oracle Batching](#oracle-batching)
    - [Candidate-Growth Control](#candidate-growth-control)
    - [Failure Semantics](#failure-semantics)
  - [Tracking](#tracking)
    - [Tracking Separation](#tracking-separation)
    - [Execution Scope](#execution-scope)
    - [Loop Identification](#loop-identification)
    - [Replica Identification](#replica-identification-1)
    - [Step Identification](#step-identification)
    - [Tracking Levels](#tracking-levels)
    - [Experiment Isolation](#experiment-isolation)
    - [Run Isolation](#run-isolation)
    - [Storage Model](#storage-model)
    - [Tracking Performance](#tracking-performance)

## System Architecture

PepCompass isolates experiment construction from the individual pipeline elements it composes. Walkers, mutation generators, filters and oracles are interchangeable strategy families behind abstract contracts. 

The optimisation engine and experiment system assemble them into a run without depending on any concrete scientific method. This lets different experiments be built from the same components, with construction (how steps are wired and looped) varying independently of the strategies plugged into them.



### Core

`pep_compass.core` is the composition root. It validates configuration,
constructs the configured encoder-decoder and converts the recursive
`optimization.steps` structure into `Step` objects. Core owns wiring; it does
not implement walking, mutation, filtering or scoring methods.

### Experiment System

`pep_compass.experiments` loads input sequences, expands repetitions and grid
variants, assigns stable plan indices and persists run results. Backends decide
where plan entries execute. Each plan entry receives a fresh optimisation
state, tracker and derived seed.

### Optimisation Engine

`pep_compass.optimization` defines `CandidateBatch`, `Step`, `Flow`, `Loop`,
`Parallel`, runtime context, state, limits, tracking contracts and result
summarisation. These objects contain no method-specific peptide logic.

### Strategy Modules

`walkers`, `mutation_generators`, `filters` and `oracles` contain independent
strategy families. Each family exposes a base contract, a manager and a
`strategies` package. Managers map public method names to factories.

The encoder-decoder family resides below `core/encoder_decoder` because the
configured model is a runtime service shared by multiple strategy families.

### Tracking and Analysis

Tracking observes step execution and writes runtime events. It is not stored in
`CandidateBatch`. Analysis reads persisted results after execution and does not
participate in candidate transformations.

## Runtime Construction

### Construction Flow

```text
configuration file
       │
       ▼
load_configuration
       │
       ▼
validate_configuration ──► materialize grid variants
       │
       ▼
EncoderDecoderManager ───► shared encoder-decoder
       │
       ▼
PepCompassCore ──────────► recursive Step tree
       │
       ▼
OptimizationRunner ──────► CandidateBatch execution
       │
       ▼
ComposableExperiment ───► persisted run result and manifest
```

### Configuration Loading

`load_configuration` reads YAML or JSON. Relative paths are retained until the
experiment plan or output directory is resolved against one explicit working
directory.

### Configuration Validation

Validation runs before model construction for every materialised grid variant.
It verifies encoder and strategy names, parameter contracts, step-tree shape,
loop counts, parallel forms, merge support, backend settings and limits.

Each step mapping contains one operation key. A parallel node contains either
explicit `branches` or `replicas` with shared `steps`.

### Strategy Resolution

Importing each built-in `strategies` package registers factory functions with
its manager. A factory declares accepted and required parameters through
`parameter_contract`. Runtime services such as `encoder_decoder` are injected
by core and excluded from user parameter validation.

### Execution-Graph Construction

Core recursively maps configuration nodes to objects:

- a list of steps becomes `Flow`;
- `loop` becomes `Loop` containing another flow;
- `parallel` becomes `Parallel` containing named flows;
- leaf operations are constructed by their strategy managers.

Replica syntax materialises separate step instances. Ten replicas therefore do
not share mutable strategy objects, even though they use the same configuration
and encoder-decoder service.

### Runner Construction

`OptimizationRunner` receives the encoder-decoder, root step, optional tracker
and safety limits. At `run()`, it seeds NumPy and Torch, encodes the starting
sequences once, constructs the initial batch and creates an isolated
`OptimizationState`.

## Pipeline Elements

### Candidate Data Model

#### Candidate

`Candidate` contains `sequence` and `latent_origin`. `latent_origin` is the
concrete latent point that generated the sequence. It is not replaced by the
mean obtained from re-encoding that sequence.

#### Candidate Batch

`CandidateBatch` stores sequences as one tuple, latent origins as a tensor with
batch dimension first, and algorithm data as named fields. Construction checks
that every candidate-aligned column has the same length.

The batch is immutable. Transformations construct another batch while tensors
may remain shared when no change is required.

#### Batch Fields

- `TensorField` stores candidate-aligned tensors;
- `ObjectField` stores candidate-aligned Python objects;
- `SharedField` stores one value shared by the batch;
- `OptionalField` adds a validity mask when merged branches do not all provide
  the same field.

#### Candidate Selection

`select(indices)` applies one index set to sequences, latent origins and every
field. Filters therefore cannot silently misalign a score or geometry column.

#### Candidate Expansion

`repeat_from_parents(parent_indices)` duplicates all parent-aligned columns.
A mutation generator then replaces sequences while retaining each duplicated
parent latent origin.

#### Batch Concatenation

`CandidateBatch.concatenate` combines branch outputs in branch order. Fields
present in only a subset of branches become optional fields. Shared fields must
have equal values to remain shared.

### Step Contract

#### Precomputation

`precompute(context)` prepares state that is reusable for the entire run.
Composite steps forward precomputation to their children before execution.

#### Iteration Preparation

`prepare_iteration(batch, context)` is available for state depending on the
current batch. The base implementation performs no work.

#### Execution

Every concrete step implements `_execute(batch, context) -> CandidateBatch`.
`Step.__call__` supplies a child execution scope, invokes preparation, surrounds
execution with tracking and propagates exceptions after recording failure.

#### Automatic Tracking

Tracking is inherited from `Step`; strategy implementations do not call the
tracker. When a scope is disabled, the wrapper performs no serialisation.

### Optimisation State

#### Oracle Observations

Oracle results are stored by objective and sequence. ROBOT consumes this state
instead of requiring observation columns to remain in every future batch.

#### Trust-Region State

Each objective owns its centre sequence, centre latent position, best score,
radius and success/failure counters. `trust_region` reads this state;
`trust_region_update` changes it after oracle evaluation.

#### Resource Counters

State counts evaluated sequences and generated candidates. Counters are
included in tracking records before and after each step.

#### Stop Requests

Reaching a limit sets `stop_requested`. Loops test it before another iteration.
Loops also stop when the current batch is empty.

## Pipeline Construction and Execution

### Execution Graph

#### Flow Execution

`Flow` passes the output of each child to the next child. It does not interpret
the fields or insert operations.

#### Loop Execution

`Loop` executes one child tree for a fixed maximum number of iterations.
Nested loop indices are appended to the execution scope, so repeated calls are
distinguishable without changing candidate data.

#### Parallel Execution

`Parallel` gives the same immutable input batch to every branch. `sequential`
executes branches in order. `concurrent` uses one worker thread per branch.
`auto` currently selects the concurrent fallback and is the stable public mode
for future batched execution.

Branch completion order does not affect merged candidate order. Futures are
read in configuration order.

#### Batch Merging

Every parallel operation must return one batch before the next step can run.
`ConcatenateMerger` is currently implemented. Alternative policies remain
explicit `NotImplementedError` extension points.

### Latent-Space Trajectories

#### Initial Encoding

The runner encodes every starting sequence once. All later positions originate
from walker updates or parent duplication.

#### SORBES Execution

SORBES receives each latent origin, computes tangent-space information, samples
a position update and decodes the resulting position. It returns new sequences
and latent origins plus singular values, left vectors, adjusted time steps and
tangent-space objects.

#### MUTANG Expansion

MUTANG reads SORBES singular values and left vectors. It selects residue options
from significant tangent directions and materialises the Cartesian product for
each parent. Every child retains the exact latent origin of its parent.

#### Latent-Origin Propagation

Sequence and latent position remain related but serve different purposes. The
sequence is the discrete candidate passed to filters and oracles. The latent
origin preserves trajectory locality for the next walker step and latent-space
selectors.

#### Repeated SORBES–MUTANG Execution

After MUTANG, the expanded batch enters the next SORBES step. SORBES walks from
every retained latent origin and decodes new sequences at the resulting points.
Repeated execution may grow the batch because mutation expansion is one-to-many.

#### Candidate-Pool Formation

A trajectory loop returns its complete current batch. Parallel trajectories
form independent pools, which are concatenated before shared mutation filters,
deduplication, trust-region selection, ROBOT and oracle evaluation.

### Replicated Trajectories

#### Replica Materialisation

`replicas: N` expands shared steps into `N` separately constructed branch
flows. Replica names use zero-padded indices: `replica_000`, `replica_001`, ….

#### Replica Identification

The execution scope stores branch names and numeric branch indices separately.
Nested parallel sections append to both tuples. Tracking therefore identifies
the complete branch hierarchy without adding fields to candidate batches.

#### Replica Random States

Each branch context derives a seed from the run seed and its branch index and
constructs a separate NumPy generator. Strategies using the context generator
therefore receive reproducible branch-local randomness.

#### Replica Result Merging

Replica outputs are merged by replica index, independently of completion order.
No deduplication or selection occurs unless a later configured step performs it.

## Runtime Implementation

### Automatic Execution Selection

`execution: auto` decouples configuration from the concrete execution method.
The current implementation uses concurrent branch execution. A batched GPU
executor can replace that choice without changing experiment files.

### Batched GPU Execution

True batched trajectory execution requires walkers and encoder-decoders to
accept an additional trajectory batch dimension while retaining per-replica
random states, activity masks and tracking identities. This optimisation is not
implemented in the current runtime.

### Concurrent Execution

Concurrent branches reuse the encoder-decoder object and have separate strategy
instances and NumPy generators. The optimisation state remains run-scoped, so
steps that mutate shared state must be placed after branch merging unless their
updates are explicitly designed for concurrency.

### Sequential Fallback

`execution: sequential` preserves the same branch and merge semantics without
simultaneous model calls. It is the diagnostic mode for strategies that cannot
share a model safely.

### Encoder-Decoder Reuse

One core instance reuses its encoder-decoder across run construction. Each run
still receives independent state, tracking and candidate data.

### Oracle Batching

`BlackBoxOracle` can split evaluated sequences by `evaluation_batch_size`. It
does not call the model for an empty batch and attaches an empty score column so
the batch schema remains valid.

### Candidate-Growth Control

MUTANG reports generated candidate counts to state. Mutation-choice strategies
also bound local Cartesian products through their `maximum_candidates`
parameters. These controls do not change the requirement that each stage
returns one aligned batch.

### Failure Semantics

The tracker records a failed step and re-raises the exception. The experiment
runner persists a failed status. Its `stop` or `continue` policy determines
whether later independent plan entries execute.

## Tracking

### Tracking Separation

Tracking records execution metadata and optional snapshots in a separate data
stream. It cannot change optimisation fields, candidate count or step order.

### Execution Scope

An `ExecutionScope` contains the hierarchical path, nested loop indices,
parallel branch names and parallel branch indices. Entering a step, iteration
or branch returns a new immutable scope.

### Loop Identification

Every nested loop appends an integer to `loop_indices` and an `iteration[N]`
segment to the path.

### Replica Identification

Every parallel level appends a stable name to `branch_names` and its numeric
position to `branch_indices`. CSV tracking serialises both as JSON arrays.

### Step Identification

`execution_id` identifies one tracker event. `step_name` identifies the class
or configured step name; `path` identifies its location in the complete tree.

### Tracking Levels

The tracker filters events by level and maximum scope depth. Candidate snapshots
are written for all tracked steps only at level `all`; oracle outputs are
written at every level.

### Experiment Isolation

Input repetition and grid expansion produce independent plan entries. Each
entry has a separate output directory and result status.

### Run Isolation

Optimisation state, counters, trust regions and trackers are constructed for one
run. Subprocess and Slurm backends additionally isolate interpreter state.

### Storage Model

`steps.csv` stores execution events. `candidates.csv` references events by
`execution_id`. Final candidates and latent origins are stored separately from
tracking snapshots.

### Tracking Performance

Serialising SORBES left-vector matrices for every candidate can dominate output
size. `store_fields` should be enabled only when those values are required for
analysis; `max_depth` should exclude internal steps not needed by the study.
