# PepCompass Documentation

<!-- REMARK: To jest po to jakby ktoś chciał docsy stworzyć jako strone na githubie -->

PepCompass composes latent-space peptide optimisation pipelines from
interchangeable walkers, mutation generators, filters and oracles, executes
them across configurable backends, and exposes the persisted results to a
separate analysis layer.

Documentation is separated by reader intent, along the same boundary the
library itself uses: running a pipeline, understanding how it executes,
extending it, and analysing what it produced. Each document stays at one
abstraction level; do not duplicate an explanation across documents.

## Contents

- [User Guide](user-guide.md)
- [Technical Architecture](technical-architecture.md)
- [Developer Guide](developer-guide.md)
- [Analysis Guide](analysis-guide.md)
- [Architecture Decisions](architecture-decisions.md)
- [Migration Notes](developer/przeniesienie%20modeli.md)

## Documentation Overview

### User Guide

[User Guide](user-guide.md) explains how to configure and run a PepCompass
experiment: 
- the `pep-compass` CLI commands, 
- the YAML configuration schema 
- the full component reference. 

### Technical Architecture

[Technical Architecture](technical-architecture.md) describes how a
configuration becomes an executable pipeline: 
- the candidate data model, 
- the step tree (`Flow`/`Loop`/`Parallel`/`LocalEnumeration`), 
- runtime construction,
- tracking and the on-disk result layout. 

### Developer Guide

[Developer Guide](developer-guide.md) defines the package layout: 
- where a new component belongs, 
- how to register it, 
- the implementation, 
- documentation and testing requirements it must satisfy.

### Analysis Guide

[Analysis Guide](analysis-guide.md) documents `pep_compass.analysis`: 
- reading persisted results (`analysis.reader`) 
- the analyses built on top of it (locality, resampling, visualisation). 

It is a separate, read-only consumer of runtime output — it does not participate in execution.

### Architecture Decisions

[Architecture Decisions](architecture-decisions.md) records why the library is
split the way it is, what was deliberately deferred, and the currently known
open issues. Read it before proposing another restructuring.

## Optimisation Process Overview

An experiment reads one or more starting sequences and encodes each sequence
once. The configured step tree transforms a columnar candidate batch: walkers
move latent positions, mutation generators expand sequences from those
positions, filters and selectors reduce candidate pools, and oracles attach
objective values. `LocalEnumeration` composes a walker and a mutation
generator into one bounded trajectory-exploration operation used by the
reference LE-BO pipeline (it is optimized and works like in initial library).

Parallel branches may execute independent trajectory definitions with their
own identity and random seed. Their results are merged into one candidate
batch before subsequent shared steps run.
