# Developer Guide

This guide defines how to add and maintain PepCompass components. Runtime
relationships are documented in
[Technical Architecture](technical-architecture.md); public configuration is
documented in [User Guide](user-guide.md).

## Table of Contents

- [Library Structure](#library-structure)
  - [Package Map](#package-map)
  - [Module Responsibilities](#module-responsibilities)
  - [Naming Conventions](#naming-conventions)
  - [Public and Internal APIs](#public-and-internal-apis)
- [Adding Components](#adding-components)
  - [Implementation Procedure](#implementation-procedure)
  - [Component Placement](#component-placement)
  - [Common Component Contract](#common-component-contract)
  - [Adding an Encoder-Decoder](#adding-an-encoder-decoder)
  - [Adding a Walker](#adding-a-walker)
  - [Adding a Mutation Generator](#adding-a-mutation-generator)
  - [Adding a Filter](#adding-a-filter)
  - [Adding a Selector](#adding-a-selector)
  - [Adding an Oracle](#adding-an-oracle)
  - [Adding a Merge Policy](#adding-a-merge-policy)
  - [Strategy Registration](#strategy-registration)
  - [Parameter Validation](#parameter-validation)
- [Mandatory Requirements](#mandatory-requirements)
  - [Type and Batch Safety](#type-and-batch-safety)
  - [Empty-Batch Handling](#empty-batch-handling)
  - [Deterministic Randomness](#deterministic-randomness)
  - [Tracking Compatibility](#tracking-compatibility)
  - [Logging](#logging)
  - [Docstrings](#docstrings)
  - [Documentation](#documentation)
  - [Unit Tests](#unit-tests)
  - [Contract Tests](#contract-tests)
  - [Universal Registry Tests](#universal-registry-tests)
  - [Integration Configurations](#integration-configurations)
- [Extension Checklist](#extension-checklist)

## Library Structure

### Package Map

```text
src/pep_compass/
  core/
    builder.py
    validation.py
    encoder_decoder/
      base.py
      manager.py
      strategies/
  experiments/
    runner/
    analysis/
    reader/
  optimization/
    batch.py
    context.py
    flow.py
    runner.py
    state.py
    step.py
    tracking.py
  walkers/
    base.py
    manager.py
    strategies/
  mutation_generators/
    base.py
    manager.py
    strategies/
  filters/
    base.py
    manager.py
    strategies/
      biology/
      constraints/
      decision_models/
      latent_geometry/
      selectors/
      sequence_geometry/
  oracles/
    base.py
    manager.py
    strategies/
  utils/
```

Configurations belong under `experiments/configs/`. Executable repository
scripts belong under `assets/scripts/`. Tests for the composable runtime belong
under `tests/`.

### Module Responsibilities

`base.py` defines a family contract. `manager.py` stores named factories.
`strategies/` contains implementations and registration factories. Do not add a
new top-level package for one strategy when it belongs to an existing family.

Core wires components but does not implement their algorithms. Optimisation
modules define generic execution and data contracts. Experiments orchestrate
runs and persistence but do not select scientific strategies implicitly.

### Naming Conventions

Use singular class names and snake-case public registry names. A strategy name
must identify behaviour rather than the experiment in which it was introduced.

Use the established suffixes when applicable:

- `Walker` for latent-position transitions;
- `Generator` for one-to-many candidate generation;
- `Filter` for admissibility transformations;
- `Selector` for policies selecting a subset;
- `Oracle` or `BlackBox` for objective evaluation;
- `Manager` for a strategy registry.

### Public and Internal APIs

Registry names and YAML parameters are public APIs. `CandidateBatch`, batch
field classes, `Step`, execution context and manager build methods are developer
contracts. Strategy helper functions are internal unless exported explicitly.

Changing a public method name or parameter requires updating validation
configurations and the [User Guide](user-guide.md).

## Adding Components

### Implementation Procedure

1. Select the existing component family responsible for the behaviour.
2. Read its base class, manager and one current strategy.
3. Implement the smallest class satisfying the family contract.
4. Add a factory and public registry name.
5. Declare accepted and required parameters.
6. Add contract and failure tests.
7. Add a validation configuration that executes the real runner.
8. Document public parameters in the User Guide.

Do not modify `PepCompassCore._build_step` when adding another implementation
of an existing family. Core changes are required only for a new operation type.

### Component Placement

Place a small implementation directly below its family `strategies/` package.
Create a subpackage when the implementation contains multiple cohesive modules,
model files or adapters. Reuse existing semantic groups under filters instead
of creating one directory per filter.

Scientific model code retained for comparison should remain inside the owning
strategy package. Backup source trees are not importable runtime modules.

### Common Component Contract

A pipeline component accepts one `CandidateBatch` and returns one
`CandidateBatch`. It may change the number of rows, sequences, latent origins or
fields only through batch operations that preserve column alignment.

Algorithm data needed by later steps belongs in `CandidateBatch.fields`.
Execution diagnostics belong in tracking. Cross-iteration observations and
control state belong in `OptimizationState`.

### Adding an Encoder-Decoder

Implement `EncoderDecoder` in
`core/encoder_decoder/strategies/<method>/`. The implementation must provide
the encoding, decoding, decoder Jacobian and field-derivative operations used by
configured walkers and geometry filters.

Register a factory in `core/encoder_decoder/strategies/__init__.py`:

```python
@EncoderDecoderManager.register("method_name")
@parameter_contract(
    accepted={"device", "required_parameter"},
    required={"required_parameter"},
)
def build_method_name(*, device: str = "cpu", **parameters):
    return MethodEncoderDecoder(device=device, **parameters)
```

The core owns this instance and injects it into factories declaring an
`encoder_decoder` service.

### Adding a Walker

Subclass `Walker` and implement `_execute`. A walker normally replaces latent
origins and decoded sequences while preserving compatible incoming fields.

Fields required by MUTANG use the established names:

```text
walker.singular_values
walker.left_vectors
walker.adjusted_time_step
walker.tangent_space
```

If a new walker cannot produce these values, document that it is not compatible
with the current MUTANG strategy rather than creating placeholder fields.

### Adding a Mutation Generator

Subclass `MutationGenerator`. For one-to-many generation:

1. compute generated sequences per parent;
2. build one parent-index list;
3. call `batch.repeat_from_parents(parent_indices)`;
4. replace sequences with `with_sequences`;
5. attach reusable generation metadata;
6. call `context.state.record_generated_candidates`.

Do not re-encode generated sequences when their required origin is the concrete
parent trajectory position.

### Adding a Filter

Subclass `Filter` for an admissibility transformation. Calculate any reusable
scores before selecting rows, attach them as a candidate-aligned field, and use
`select(indices)` to return accepted candidates.

A filter may return an empty batch. It must not raise only because no candidate
passed its rule.

### Adding a Selector

Subclass `Selector` when the component chooses candidates using an optimisation
policy, observation history, diversity rule or resource target. State-dependent
selectors read `context.state`; they must not encode persistent state into
tracking fields.

Register selectors through `FilterManager`, because `filter` is the public step
operation for both filters and selectors.

### Adding an Oracle

Subclass `Oracle` or adapt an existing POLI black box with `BlackBoxOracle`.
An oracle must:

- attach a one-dimensional `oracle.<name>.score` tensor;
- attach objective name and direction fields;
- record observations through `OptimizationState`;
- respect the remaining oracle-call budget;
- return a correctly shaped empty score field without calling its model for an
  empty batch.

Import model-specific dependencies lazily in the registered factory so unused
oracles do not prevent library import.

### Adding a Merge Policy

Implement `BatchMerger.__call__(batches) -> CandidateBatch` in
`optimization/flow.py` and add it to `build_merger`. Define stable ordering,
missing-field behaviour, required score fields, output size and random-state
usage before implementation.

A merge policy must return one batch. Deduplication remains a separate filter
unless deduplication is explicitly part of the merge policy contract.

### Strategy Registration

Managers expose `register(name)`, `build(name, ...)`, `methods()` and
`factory(name)`. Built-in strategy packages are imported by core validation and
construction. Registration names must be unique inside their family.

Prefer a factory when construction requires injected services or adapter
objects:

```python
@WalkerManager.register("new_walker")
@parameter_contract(source=NewWalkerImplementation)
def build_new_walker(encoder_decoder, **parameters):
    return NewWalker(NewWalkerImplementation(encoder_decoder, **parameters))
```

### Parameter Validation

Use `parameter_contract(source=Type)` when the source constructor accurately
declares public parameters. Use explicit `accepted` and `required` sets when a
factory translates parameters or uses `*args` and `**kwargs`.

Injected service names are excluded during configuration validation. Do not
accept arbitrary unknown user parameters only to ignore them.

Every configuration grid is validated after value substitution. A method grid
therefore requires its unchanged parameter mapping to be accepted by every
selected method.

## Mandatory Requirements

### Type and Batch Safety

- Return `CandidateBatch` from every `_execute` method.
- Keep the batch dimension first in tensor fields.
- Use `select` for row filtering.
- Use `repeat_from_parents` for one-to-many expansion.
- Preserve device compatibility for concatenated tensors.
- Use optional fields only when a field is genuinely absent from a branch.

### Empty-Batch Handling

Test empty input when the component can follow a filter or selector. Components
must either return an aligned empty batch or document and validate a required
non-empty precondition. Oracle adapters must not call third-party models with
empty input.

### Deterministic Randomness

Use `context.rng` for NumPy sampling. Do not construct an unseeded generator in
the strategy. Parallel identity is provided through
`context.scope.branch_names` and `branch_indices`.

Torch sampling currently uses the run-level Torch seed. A component requiring
strict branch-local Torch randomness must accept or derive an explicit
`torch.Generator` rather than relying on thread scheduling.

### Tracking Compatibility

Do not call tracker lifecycle methods from a strategy. Inherit `Step` and allow
the base wrapper to record execution. Add reusable algorithm values to batch
fields only when later computation consumes them.

### Logging

Nontrivial computation and orchestration modules initialise
`logger = get_custom_logger(__name__)`. Use lazy interpolation. Log stage-level
progress at `info`, dimensions and parameters at `debug`, recoverable anomalies
at `warning`, and contextualised handled failures at `error` with exception
information.

### Docstrings

Public Python classes and methods use Sphinx-compatible reStructuredText
docstrings. Document parameters, return values, raised exceptions, assumptions,
side effects and input constraints.

### Documentation

Update the document responsible for the change:

- public YAML method or parameter: [User Guide](user-guide.md);
- runtime relationship or data flow: [Technical Architecture](technical-architecture.md);
- extension or maintenance procedure: this guide.

Do not duplicate the same explanation in all three documents.

### Unit Tests

Test method-specific computation with small deterministic tensors or mock
sequences. A bug fix includes a regression test reproducing the original
failure.

### Contract Tests

Every pipeline component test verifies:

- returned type;
- sequence and latent-origin alignment;
- field lengths;
- parent-origin propagation for expansion;
- expected candidate-count behaviour;
- empty-batch behaviour where applicable.

### Universal Registry Tests

Manager contract tests iterate over every registered strategy family and verify
that registered implementations inherit the required base type. Extend these
tests when introducing another manager or operation family.

### Integration Configurations

Add or extend a small file in `experiments/configs/validation/`. Each file
should isolate one component family and use a grid for comparable methods or
parameters. Use the mock sequence CSV under `data/peptides/`.

Validate it before a full run:

```bash
uv run --extra cu118 python \
  assets/scripts/runner/run_composable_optimization.py \
  --config experiments/configs/validation/<configuration>.yaml \
  --dry-run
```

Then execute the smallest real variant needed to exercise the component.

## Extension Checklist

- [ ] The implementation belongs to an existing component family or a new
      family is justified.
- [ ] Public code, identifiers, comments, logs and configuration are English.
- [ ] The class satisfies its base contract.
- [ ] Candidate columns remain aligned after every transformation.
- [ ] Empty batches are handled or rejected by an explicit precondition.
- [ ] Random operations use controlled random state.
- [ ] The strategy has a unique registry name.
- [ ] Public parameters have an explicit contract.
- [ ] Unit and regression tests cover computation.
- [ ] Contract tests cover batch behaviour.
- [ ] Registry tests recognise the strategy.
- [ ] A validation YAML exercises the real runner.
- [ ] The responsible documentation file is updated.
- [ ] Focused tests, full applicable tests and configuration dry-runs pass.
