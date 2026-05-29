# eps_greedy_learn — Q-learning over MUTANG++ action spaces

This folder documents the **tabular Q-learning** epsilon-greedy agent implemented
in [scripts/eps_greedy_learn.py](../scripts/eps_greedy_learn.py).

It is the learning counterpart to [`eps_greedy_rl.py`](../scripts/eps_greedy_rl.py)
(documented in [`basic_eps_greedy_rl/README.md`](../basic_eps_greedy_rl/README.md)):
that earlier baseline calls APEX directly during exploitation and never updates
a value function. `eps_greedy_learn.py` keeps the same action space, but its
exploit step uses a **learned** Q-table that is refined by TD(0) updates from
observed rewards.

---

## Three core components

| Component | Role |
|---|---|
| **HydrAMP encoder-decoder** | Peptide ↔ 64-dim latent space; provides the decoder Jacobian used to define the local tangent space |
| **APEX predictor** | 8-model ensemble that predicts MIC (µg/mL) per species; drives reward and Q-table warm-start |
| **MutationEnumerationInTangentSpace** | Identifies positions / amino acids reachable along the latent manifold |
| **ProjectedDirectionPairwiseSimilarityPotential** | MUTANG++ scoring of combined mutations in the horizontal sub-Riemannian tangent space |

These are built **once per run** (or once per multi-agent batch) and held in
memory. Encoder/decoder weights live on the chosen device (default `cuda`),
APEX runs the full 8-model ensemble per `predict` call.

---

## MDP definition

| Element | Definition |
|---|---|
| **State** `s` | Current peptide string |
| **Action** `a` | A neighbour peptide chosen from the MUTANG++-filtered action set at `s` |
| **Transition** | Deterministic: `s' = a` |
| **Reward** | `r = log₂MIC(s) − log₂MIC(s')` — positive when MIC decreases (better antimicrobial activity) |
| **Discount** | `γ` (default `0.99`) |
| **Policy** | Epsilon-greedy over the learned Q-table |

Non-finite rewards are clamped to 0.

### Q-learning update (TD(0))

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a' ∈ A(s')} Q(s', a') − Q(s, a) ]
```

`A(s')` is the MUTANG++-filtered candidate set at the next state. When that
set is empty (the agent is **stuck**), the bootstrap collapses to the
self-loop value `Q(s', s')`.

### Warm-start (initial Q values)

Unseen `(s, a)` pairs are initialised as:

```
Q(s, a)  ←  −APEX_score(a)        (mean log₂MIC over E. coli, sign-flipped)
```

This seeds the table with the APEX prior — "if you stop at `a`, your expected
future return equals how good `a` is right now." TD updates then refine these
values based on observed downstream trajectories. Sign is flipped because the
agent *minimises* MIC but Q values represent return (positive = better).

---

## Episode flow

An episode starts from a single peptide drawn from the pool and runs for
`n_steps` (default 20) transitions.

```
for each step t in 0 .. n_steps-1:

    1. Look up cand_seqs, cand_probs ← candidate_cache[current_peptide]
       If absent, run generate_candidates_for_rl:
           a) encode current peptide → latent z  (cached)
           b) decoder Jacobian J at z; SVD J = U S Vᵀ
           c) MutationEnumerationInTangentSpace(S, U) → positions + AA candidates
           d) cap cartesian product to MAX_CANDIDATES_PER_STEP (6 000)
           e) ProjectedDirectionPairwiseSimilarityPotential.compute → log potentials
           f) top_p (nucleus, default 0.90) OR top_n (hard cap, default 20) filter
           g) drop the parent sequence
           h) softmax of remaining log_potentials → cand_probs (used only for explore)

    2. If only the parent survives  → mark step as "stuck", reward 0, continue.

    3. Batch-score every uncached candidate through APEX in ONE call.
       Populates score_cache[a] = log₂MIC(a) for all a ∈ cand_seqs.

    4. Epsilon-greedy action selection:
        • explore  (prob ε):  sample a ~ Categorical(cand_probs)
        • exploit  (prob 1-ε): a = argmax_a Q(s, a)
                                using LEARNED Q values, NOT APEX directly
       Q(s, a) is read from the table; absent entries are warm-started to
       −score_cache[a] (a single dict lookup — APEX was already batched in step 3).

    5. Reward r = score_cache[current] − score_cache[next].

    6. Build next-state candidate set A(s'); batch-score it through APEX.

    7. Apply the Q-learning update:
           old   = Q(s, a)             (or −score_cache[a] if unseen)
           targ  = r + γ · max_{a'} Q(s', a')
           Q(s, a) ← old + α · (targ − old)

    8. Advance: s ← s', track best peptide so far in the episode.
```

The episode return is `Σ r_t`. The global best peptide (lowest log₂MIC) is
tracked across all episodes and all agents.

---

## Epsilon schedule

Epsilon decays once **per episode** (not per step), after the episode ends:

```
ε_{ep+1} = max(epsilon_end, ε_ep × epsilon_decay)
```

Default schedule: `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay=0.9940`.
Roughly exponential decay from full exploration to 5 % exploration over the
first ~500 episodes.

---

## Caching architecture

Four caches drive most of the speedup:

| Cache | Key | Value | Lifetime |
|---|---|---|---|
| `latent_cache` | peptide | encoder mean (np.float32, dim 64) | per run; shared across agents in `multi` |
| `score_cache` | peptide | mean log₂MIC over E. coli (float) | per run; shared across agents in `multi` |
| `candidate_cache` | peptide | `(cand_seqs, cand_probs)` | per run; shared across agents in `multi` |
| `q_table` | `(s, a)` | float Q value | **per agent** — never shared, that is what makes each agent learn independently |

The encoder, decoder Jacobian + SVD, and APEX outputs are deterministic
per peptide, so any of them computed in agent 1 is reused by agents 2…N for
free. The Q-table is intentionally NOT shared — multiple agents are run
precisely to diversify the learned policy.

### Batching (the big perf lever)

Per step, the agent needs APEX scores for the entire current candidate set
(for the exploit argmax / Q init) and the entire next-state candidate set
(for the bootstrap max). The implementation collects every uncached peptide
in those sets and submits them as a **single** batch to APEX. Each APEX
`predict` call runs 8 ensemble models, so batching collapses what would
otherwise be hundreds of single-peptide GPU passes into two batched calls.

Helper: `_batch_score_into_cache(apex_predictor, score_cache, peptides)` in
[scripts/eps_greedy_learn.py](../scripts/eps_greedy_learn.py).

---

## Action-space filtering

Set via `--action_filter`:

| Mode | Behaviour |
|---|---|
| `top_p` (default) | Nucleus sampling. Sort candidates by potential, keep the smallest prefix whose cumulative softmax mass ≤ `top_p` (default 0.90). At least one candidate is always kept. |
| `top_n` | Hard cap. Keep the `top_n` candidates with the highest potential (default 20). |

`top_p` adapts the action-set size to the local geometry — wide when many
directions look promising, narrow when one dominates. `top_n` gives a
predictable per-step cost.

The parent sequence is always excluded from the action set. If filtering
leaves the parent only, the step is marked **stuck**: reward 0, no Q update,
loop continues from the same state.

---

## Start-peptide selection

Each episode draws its starting peptide from a pool:

- `start_selection=random` (default): uniform random draw each episode
- `start_selection=cycle`: deterministic round-robin

The pool is loaded from a plain-text file (one peptide per line, default
`basic_eps_greedy_rl/inputs/sampled_500_peptides.txt`). If the file is
missing or empty, the built-in 10-peptide `STARTING_PEPTIDES` fallback is
used.

All start peptides are batch-encoded and batch-scored before episode 1.

---

## Single agent vs multi-agent

### Single agent (`run`)

One Q-table, one seed (default 0). Caches are private to the run.

### Multi-agent (`multi`)

`run_multi_agent` runs `n_agents` agents **sequentially** with seeds
`17, 18, 19, …`. The encoder, APEX, and enumerator objects are built once.
The three peptide-keyed caches (`latent_cache`, `score_cache`,
`candidate_cache`) are **shared** across agents — agents 2…N reuse all
encoder / APEX work agent 1 already did. Each agent gets its own
**independent** Q-table.

GPU memory is freed between agents with `gc.collect()` +
`torch.cuda.empty_cache()`. The CPU dict caches persist.

---

## Inference

There is no separate "inference" mode. The "trained" artifact of a run is
the agent's final Q-table together with the best-peptide log.

To use a trained agent for greedy rollout:

1. Persist the Q-table at end of training (currently only its size is saved
   in `*_results.json`; serialising the full dict is a one-line addition if
   needed for downstream use).
2. Reload it, set `epsilon = 0`, run additional episodes from any start
   peptide. The agent will pick `argmax_a Q(s, a)` at every step, falling
   back to the APEX warm-start `−score_cache[a]` for any `(s, a)` it has
   never seen.

The global-best peptide tracked in `*_results.json` is the optimisation
output you typically care about.

---

## CLI

```
# single agent
uv run python scripts/eps_greedy_learn.py run [options]

# multiple agents (models built once, caches shared)
uv run python scripts/eps_greedy_learn.py multi [options]
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--n_episodes` | 500 | Episodes per agent |
| `--n_steps` | 20 | Steps per episode |
| `--epsilon_start` | 1.0 | Initial exploration rate |
| `--epsilon_end` | 0.05 | Minimum exploration rate |
| `--epsilon_decay` | 0.9940 | Multiplicative decay per episode |
| `--alpha` | 0.1 | Q-learning step size |
| `--gamma` | 0.99 | Discount |
| `--action_filter` | `top_p` | `top_p` (nucleus) or `top_n` (hard cap) |
| `--top_p` | 0.90 | Nucleus mass threshold |
| `--top_n` | 20 | Hard candidate cap |
| `--device` | `cuda` | PyTorch device |
| `--start_selection` | `random` | `random` or `cycle` |
| `--start_peptides_file` | `basic_eps_greedy_rl/inputs/sampled_500_peptides.txt` | Peptide pool file |
| `--output_dir` | `results/eps_greedy_learn` | Output directory |
| `--n_agents` (`multi` only) | 5 | Number of independent agents |
| `--run_name` (`run` only) | `""` | Tag in output filenames |
| `--seed` (`run` only) | 0 | RNG seed (per-agent in `multi`: 17 + i) |

---

## Output files

Each agent writes two files in `--output_dir`:

### `<run_name>_qlearn_<unix_ts>_log.csv`

One row per episode:

| Column | Meaning |
|---|---|
| `episode` | 1-indexed episode number |
| `epsilon` | Exploration rate used this episode |
| `start_peptide`, `start_log2mic` | Episode start state and its score |
| `best_peptide`, `best_log2mic` | Best peptide reached *within this episode* |
| `global_best_peptide`, `global_best_log2mic` | Best across all episodes so far |
| `episode_return` | `Σ r_t` for the episode |
| `n_stuck_steps` | Steps that produced no action |
| `q_table_size` | Number of `(s, a)` pairs known after this episode |
| `action_filter` | `top_p` or `top_n` |

The CSV is flushed every 50 episodes (and at end of run) so partial progress
is durable without per-row syscall overhead.

### `<run_name>_qlearn_<unix_ts>_results.json`

Full run metadata: hyperparameters, per-episode arrays (`episode_returns`,
`episode_best_scores`, `episode_best_peptides`, `q_table_sizes`), and the
final global best.

The Q-table itself is **not** currently serialised — only its size. Add a
`json.dumps({f"{s}|{a}": v for (s, a), v in q_table.items()})` step at the
end of `_run_with_prebuilt_models` if you need it.

---

## Reproducibility

Seeds set per run: `numpy.random.default_rng(seed)`, `random.seed(seed)`,
`torch.manual_seed(seed)`. In `multi`, agent `i` uses `seed = 17 + i`.

Caches are not part of the seed; sharing them across agents in `multi` does
not affect determinism of any single agent's trajectory because the cached
values are deterministic functions of the peptide string.

---

## Performance notes

The dominant per-step cost is APEX (8 ensemble models, GPU). Two changes
relative to the original implementation give the bulk of the speedup:

1. **Batched APEX per step** — every uncached candidate at the current and
   next state is scored in two batched calls (one for `cand_seqs`, one for
   `next_cands`). Previously each `get_q` / `update_q` lookup triggered a
   single-peptide forward pass.
2. **Cross-agent cache sharing in `multi`** — agents 2…N reuse all
   encoder + APEX outputs agent 1 produced. Caches live for the full
   `run_multi_agent` call.

Together these typically reduce wall-time by an order of magnitude on a
500-episode × 20-step × 5-agent run with `top_p=0.90`. Q-learning
trajectories are unchanged — only the order of work is.
