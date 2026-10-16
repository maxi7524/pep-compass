# User Guide

Use the [Python wrapper guide](tutorial/wrapper.md) for component discovery, graph
construction and individual runs. Use the [Hydra orchestrator guide](tutorial/orchestrator.md)
for YAML defaults, overrides, sweeps, execution strategies and server templates.
The canonical CLI is `pep_compass.orchestrator.cli`; obsolete runtime commands
and import aliases have been removed.

## Available Components

Registry names and parameters below are public API — changing one requires
updating the responsible strategy's registration metadata and this table (see
[Developer Guide](developer-guide.md#documentation)).

### Autoencoders

- `hydramp` — peptide autoencoder with strict or approximate decoder
  Jacobians; model variants selected through `autoencoder.model` (currently
  `article_25`).

### Walkers

- `sorbes` — one SORBES step: latent geometry, direction sampling, an
  adaptive position update and a boundary check, in that fixed order. Accepts
  `geometry`, `directions`, `scaling`, `position_update` and `boundary` as
  `{method, parameters}` sub-strategies. The built-in methods are
  `kappa_stable`, `active_inactive`, `stable_dimension`, `main`, and, for
  `position_update`, also `article` and `without_acceleration`. A legacy flat
  form (`horizontal_threshold`, `time_step`, `max_horizontal_update_norm`,
  `vertical_movement`) is converted to the corresponding nested defaults.

### Mutation Generators

- `mutang` — reads SORBES singular values and left vectors, selects
  residue substitutions from significant tangent directions and materialises
  their Cartesian product per parent. Accepts `maximum_candidates`,
  `strategies.geometry.method` (`shared` or `kappa_stable`), `max_len`,
  `direction_significance_threshold`, `min_number_of_directions`,
  `token_threshold`, `alphabet`. For HydrAMP, `max_len` must match the model
  sequence length (25); it is not a candidate limit.

MUTANG requires the `walker.singular_values`/`walker.left_vectors` fields
produced by SORBES, so a `mutation_generator` step must follow a `walker`
step (or run inside `local_enumeration`, which wires this automatically).

### Filters

Filters split into two extension styles
(`optimization/components/filters/{direct,ranked}/`):

- **Direct filters** are self-contained batch transformations:

  | method | parameters | description |
  |---|---|---|
  | `deduplicate` | `key`: `sequence` \| `sequence_and_latent` | keeps the first candidate per identity key |
  | `candidate_subset` | `count`, `mode`: `random`\|`highest`\|`lowest`, `score_field` (required unless `random`) | caps batch size by random or score-ranked subset |
  | `levenshtein` | `reference` or `reference_field` (exactly one), `maximum_distance` | keeps sequences within an edit-distance radius of a reference |
  | `sequence_length` | `minimum`, `maximum` | keeps sequences within an inclusive length interval |
  | `robot` | `objective`, `batch_size`, `maximize`, `diversity_threshold`, `acquisition_batch_size`, `standardize`, `device` | Bayesian ROBOT selection: GP + LogEI over molecular fingerprints, with a diversity threshold |
  | `trust_region` | `objective`, `geometry`: `sequence`\|`latent`, `initial_radius` | retains candidates inside the current per-objective trust-region radius |
  | `trust_region_update` | `objective`, `initial_radius`, `minimum_radius`, `maximum_radius`, `expand_factor`, `shrink_factor`, `success_tolerance`, `failure_tolerance`, `maximize` | expands or shrinks the trust-region radius from oracle scores already on the batch |
  | `random_walker` / `random_mutang` | `selection_fraction`, `temperature`, `maximum_positions`, `residues_per_position`, `maximum_candidates`, `alphabet` | random-baseline ablation controls (walker-position vs. MUTANG-pool randomisation) |

- **Ranked filters** compose one `ScoreFunction` with one `SelectionRule`
  (`filters/ranked/base.py`). Use `ranked` directly for a custom
  combination, or one of the pre-built method names that already pair a
  specific score with a specific selection rule:

  | method | scoring | selection | parameters |
  |---|---|---|---|
  | `ranked` | any of `tandem`, `lams`, `move`, `decoder_likelihood`, `esm` | any of `threshold`, `nucleus`, `top_k` | `scoring: {method, parameters}`, `selection: {method, parameters}` |
  | `lpbebo` | decoder log-probability | nucleus top-p | `top_p`, `temperature`, `maximum_candidates`, `alphabet` |
  | `lams` | worst compatible mutation-pair cosine similarity | threshold | `similarity_threshold`, `horizontal_threshold`, `maximum_candidates`, `alphabet` |
  | `tandem` | projected pairwise mutation-direction similarity | nucleus top-p | `top_p`, `temperature`, `horizontal_threshold`, `maximum_candidates`, `alphabet` |
  | `move` | negative predicted net latent displacement | nucleus top-p | `top_p`, `temperature`, `maximum_candidates`, `alphabet` |
  | `esm_plausibility` | ESM2 pseudo-log-likelihood | threshold XOR top-k | `threshold`, `top_k`, `model_name`, `device` |

  Mutation-choice filters (`lpbebo`, `lams`, `tandem`, `move`,
  `random_walker`, `random_mutang`) consume MUTANG's mutation-choice fields,
  so they must follow a `mutation_generator` step.

### Oracles

Built-in methods are `apex`, `apex_original` (preserved pre-refactor APEX for
comparison), `battleamp`, `eipred`, `hydrophobicity`, `mbc_attention` and
`toxipep`. Every oracle is a `BlackBoxOracle` adapter and accepts optional
POLI controls (`batch_size`, `parallelize`, `num_workers`,
`evaluation_budget`, `force_isolation`) plus `evaluation_batch_size`. APEX
accepts `model` as a named weight variant and defaults to `default` when it is
omitted. BattleAMP and MBC-Attention require TensorFlow in the runtime
environment. Model-specific parameters are validated before model
construction; a lazily-imported oracle module does not prevent library import
when its dependency is missing.
