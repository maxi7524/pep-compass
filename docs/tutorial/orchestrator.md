# Hydra experiment orchestration

The orchestrator composes configuration, freezes jobs and selects an execution
strategy. Each job calls the same public `Pipeline` wrapper used from Python.
The builder and computation engine have no Hydra dependency. Install the project
with a computation extra, for example `uv sync --extra cpu`.

## Commands and substitutions

```bash
pep-compass list
pep-compass plan pipeline=simple ++runner.execution.seed=7
pep-compass run pipeline=simple orchestrator.output_dir=results
pep-compass run --multirun pipeline=simple ++runner.execution.seed=7,8
pep-compass run backend=subprocess backend.parameters.max_workers=2 input.mode=individual
```

`plan` resolves defaults and validates the graph without loading weights or
checking GPU availability. `run` loads resources through the wrapper. A Hydra
sweep creates a separate resolved plan for each configuration. Input mode
`batch` (default) sends all peptides to one wrapper run; `individual` creates one
job per peptide. Relative paths are based on the original invocation directory,
including when Hydra changes the job directory.

The packaged defaults list selects `pipeline=simple`, `runner=default`,
`input=example`, `orchestrator=default`, and `backend=local`. Pipeline groups are
derived from decorated factories under `predefined`; backend groups are derived
from decorated strategy constructors. YAML groups can replace either.

## Configuration

A custom `experiment.yaml` can use registered groups and ordinary Hydra defaults:

```yaml
defaults:
  - pipeline: simple
  - runner: default
  - backend: local
  - _self_
input:
  sequences: [FLYKWWIRIGRLKL, GIGKFLHSAKKFGKAFVGEIMNS]
  mode: individual
runner:
  execution:
    seed: 7
  storage:
    backend: sqlite
    reconstruction: boundaries
orchestrator:
  name: demo
  output_dir: results
  skip_completed: false
  continue_on_error: false
plugins: []
```

```bash
pep-compass plan --config-path /absolute/path/to/configs --config-name experiment
pep-compass run --config-path /absolute/path/to/configs --config-name experiment pipeline=lebo pipeline.parameters.mutation_candidates=4 pipeline.parameters.oracle_calls=3
```

The LEBO preset still selects its actual APEX model and optimization components;
small limits do not replace that model. Model artifacts must be installed before
execution.

`runner` is a patch over the preset's `RunnerSettings`. An omitted value preserves
its preset default. Explicit `null` removes a threshold. Use
`++runner.execution.seed=...` when adding a key absent from the composed mapping.
Set resource device and seed in `runner.execution`; do not duplicate them in
`pipeline.parameters`. The orchestrator assigns a unique `storage.path` to each
SQLite job. `storage.backend=memory` returns a status/summary and closes the
transient result; use SQLite for later analysis. Retention options and defaults
are described in the [wrapper guide](wrapper.md).

Input can instead be `{path: peptides.csv, column: sequence, mode: individual}`;
CSV, FASTA and one-sequence-per-line text are supported. When overriding the
packaged inline example, set `input.sequences=null` as well as `input.path=...`.
`input.limit` optionally bounds the input collection before batching.

An explicit graph uses the wrapper's names:

```yaml
pipeline:
  autoencoder:
    strategy: hydramp
    model: article_25
    parameters: {jacobian_mode: approx}
  steps:
    - component: filter
      strategy: deduplicate
    - component: oracle
      strategy: hydrophobicity
```

Composite nodes use `type: loop`, `type: parallel` or `type: local_enumeration`,
with the public `Loop.create`, `Parallel.create` and `LocalEnumeration.create`
parameters. `pipeline.schema` also accepts a saved schema mapping or JSON string.
Select exactly one of `predefined`, `steps`, or `schema`.

## Python composition and custom registrations

```python
from pep_compass.orchestrator.configuration import compose_configuration
from pep_compass.orchestrator.planning import plan_configuration
from pep_compass.orchestrator.runner import Orchestrator

configuration = compose_configuration(["pipeline=simple", "++runner.execution.seed=7"])
plan = plan_configuration(configuration)
records = Orchestrator().run(plan)
```

Register a preset in an importable application module:

```python
from pep_compass import Pipeline
from pep_compass.predefined import register

@register("my_pipeline")
def create(*, count=3):
    pipeline = Pipeline()
    pipeline.add_autoencoder()
    pipeline.add_component("filter", "candidate_subset", parameters={"count": count})
    pipeline.add_component("oracle", "hydrophobicity")
    return pipeline
```

Use `pep-compass run --plugin my_project.presets pipeline=my_pipeline`, or
`compose_configuration(..., plugins=["my_project.presets"])`. The explicit plugin
imports run before group discovery. A YAML `plugins` list is also loaded before
planning; a plugin defining a new Hydra group must be imported with `--plugin`
before composition. Workers import the recorded plugin list.

Execution strategies use the common registry decorator:

```python
from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.registry.execution.backends import register

@register("my_server")
class ServerBackend(ExecutionBackend):
    def __init__(self, queue="cpu"):
        self.queue = queue

    def run(self, orchestrator, plan, plan_path):
        # Submit canonical workers, or invoke orchestrator.run_entry locally.
        return tuple(orchestrator.run_entry(plan, entry) for entry in plan.entries)
```

Backend constructor defaults become Hydra parameters. Strategy selection does
not require adding a conditional branch to the orchestrator. For Python-local
work, a prepared catalog or initialized autoencoder can be injected into planning
and `Orchestrator`. Isolated workers require importable plugin registrations.

## Server-specific Slurm scripts

The packaged default script is `orchestrator/templates/slurm_array.sh`.
A server configuration group can select the built-in Slurm strategy with a site
shell template. For example, create `backend/cluster.yaml` in a custom config tree:

```yaml
strategy: slurm
parameters:
  template_path: configs/slurm/cluster.sh
  settings:
    partition: cpu
    cpus_per_task: 4
    memory: 8G
    time: '00:20:00'
```

The template may load modules, activate an environment or define site directives:

```bash
#!/bin/bash
#SBATCH --array=0-${array_max}
${directives}
set -euo pipefail
# Add site-specific module/environment setup here.
cd ${working_directory}
${command}
```

Supported substitutions are `${array_max}`, `${directives}`,
`${working_directory}`, `${plan_path}` and `${command}`. The worker command uses
the planning interpreter and quotes paths; `${SLURM_ARRAY_TASK_ID}` remains a
shell expression. Change site environment setup in the template when the compute
node uses another interpreter or container. `${command}` and `${array_max}` are
required. The template content is frozen into the plan at planning time.

```bash
pep-compass run --config-path /absolute/path/to/configs --config-name experiment backend=cluster
sbatch results/demo/plans/<fingerprint>.slurm.sh
```

Generation never calls `sbatch`. The shared plan, installation, input models and
working directory must be reachable from every compute node. Slurm submission
and remote availability require validation on the actual server.

## Artifacts, skipping and replay

```text
results/<experiment>/
  plans/<fingerprint>.json
  plans/<fingerprint>.launch-<digest>.json  # changed dispatch settings
  plans/<plan-name>.slurm.sh      # Slurm strategy only
  statuses/<job_id>.json
  runs/<job_id>/
    settings.json
    results.sqlite
```

Plans include the resolved configuration, schema, input batches, runner settings,
package/plugin source digests and dispatch settings. A digest guards accidental plan edits; it is
not a signature. Repeated equivalent plans reuse persisted node identities.
`job_id` identifies an orchestrator task; the stored `run_id` identifies the actual
wrapper execution. Operational backend and skip/failure choices do not change
computation identity. A changed launch policy gets its own immutable launch
artifact, sharing the original schema identities and job outputs. Registered
plugin module files are hashed; environment dependencies and files imported by
those modules must still be versioned by the application owner.

`orchestrator.skip_completed=true` validates an existing completed SQLite bundle,
its schema, inputs, settings, source digest and recorded model-file hashes before
skipping it. Failed, partial or mismatching output is not overwritten. A job lock
prevents concurrent output ownership. A stale lock after a killed process needs
manual inspection/removal. This is completed-job skipping, not continuation from
an interrupted computation.

Read each bundle using `AnalysisReader`; retention and replay remain owned by the
shared saving/analysis contract. Old CSV/JSON/PT archives can still be read and
replayed by `analysis.reader`. The removed `runtime`, `data` and forwarding
modules have no import aliases. Historical YAML execution commands must be
rewritten to the new configuration grammar.

Hydra APIs follow the official [Compose API](https://hydra.cc/docs/1.3/advanced/compose_api/)
and [ConfigStore](https://hydra.cc/docs/1.3/tutorials/structured_config/config_store/)
contracts. The CLI uses `@hydra.main` to retain standard multirun behavior.
