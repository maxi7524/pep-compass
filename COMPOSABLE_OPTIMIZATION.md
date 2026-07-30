# Composable optimization engine

The engine executes exactly the steps listed in `optimization.steps`. It does
not insert a walker, mutation generator, filter, or oracle implicitly.

Every step receives and returns one immutable, columnar `CandidateBatch`.
Sequences stay paired with the concrete `latent_origin` from which they were
generated. A mutation does not replace that origin by re-encoding its sequence.

## Compositions

- `Flow` executes configured steps in order.
- `Loop` repeats one nested flow and records every nested iteration index.
- `Parallel` gives the same immutable input to independent branches and merges
  their outputs into one batch.

`parallel.execution` controls computation independently of merge semantics:

- `sequential` evaluates logical branches one after another and is the safe
  default for shared GPU models;
- `concurrent` uses one worker thread per branch and retains configuration order
  in the merged result.

`parallel.merge: concatenate` is implemented. It preserves duplicates and adds
validity masks when a field exists only in selected branches. `interleave`,
`select_best`, and `weighted_sample` are explicit placeholders documented in
`pep_compass.optimization.flow`; they raise `NotImplementedError` until their
selection parameters and deterministic behavior are specified.

## Runtime and tracking data

`CandidateBatch.fields` contains reusable algorithm data such as SORBES
singular values, model scores, and oracle values. Tracking is held separately
and cannot enlarge or alter the optimization batch. The `Step.__call__` template
adds tracking around every concrete or composite step. A disabled tracker or a
scope below the configured depth performs no serialization or tensor transfer.

## Experiments without an oracle

Oracle is optional. A finite configured tree can test an individual filter,
walker, generator, loop, or branch composition. Such a result contains final
candidates and latent origins, while objective and best-score fields are null.

Run the example configuration with:

```bash
uv run python scripts/runner/run_composable_optimization.py \
  --config configs/optimization/composable_example.yaml
```
