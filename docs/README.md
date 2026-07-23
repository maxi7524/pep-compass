# PepCompass documentation

The documentation is split by responsibility:

- [Library architecture](library_architecture.md) explains the runtime data
  flow, package structure, module contracts, and extension points.
- [LE-BO implementation](lebo_implementation.md) explains the integrated
  algorithms, historical object migration, tensor shapes, and method-specific
  candidate selection.
- [Optimization runner](optimization_runners.md) explains configuration,
  parameter grids, local and Slurm execution, outputs, and aggregation.

Read the architecture first when changing library code. Read the runner guide
when adding an experiment that only composes existing components.
