# PepCompass Documentation

Pep-compass 

The documentation is separated by reader intent. Usage, runtime internals and
extension procedures are documented independently so that each document stays
at one abstraction level.

## Documentation Overview

### User Guide

[User Guide](user-guide.md) explains how to configure, execute and inspect a
PepCompass experiment. Start here when preparing YAML input or selecting an
execution backend.

### Technical Architecture

[Technical Architecture](technical-architecture.md) describes the candidate
data model, step tree, latent-origin propagation, parallel execution and
tracking lifecycle.

### Developer Guide

[Developer Guide](developer-guide.md) defines where components belong, how to
register them and which implementation, documentation and testing requirements
must be satisfied.

### Analysis Documentation

[Locality Analysis](locality_analysis.md) documents the analysis interface for
locality experiment results. It is separate from optimisation execution.

## Optimisation Process Overview

An experiment reads one or more starting sequences and encodes each sequence
once. The configured step tree transforms a columnar candidate batch. Walkers
move latent positions, mutation generators expand sequences from those
positions, filters and selectors reduce candidate pools, and oracles attach
objective values.

Parallel replicas may execute the same trajectory definition with independent
identities and random seeds. Their results are merged into one candidate batch
before subsequent shared steps run.

## Documentation Navigation

- To run an experiment, continue with the [User Guide](user-guide.md).
- To understand execution and data flow, continue with
  [Technical Architecture](technical-architecture.md).
- To add or modify a component, continue with the
  [Developer Guide](developer-guide.md).
