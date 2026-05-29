# basic_eps_greedy_rl

This folder documents the epsilon-greedy RL baseline used in this repository:
an **epsilon-greedy policy over a MUTANG++-structured action space** that
minimizes APEX mean log₂(MIC) over E. coli.

---

## Algorithm overview

The optimizer is a model-free, tabular-style epsilon-greedy agent.  There is no
learned value function or policy network — the action space at each step is
constructed on-the-fly from the geometry of the latent space around the current
peptide, and the greedy choice is made by directly querying the APEX predictor.

### Three core components

| Component | Role |
|---|---|
| **HydrAMP encoder-decoder** | Maps peptides ↔ 64-dim latent space; provides decoder Jacobian |
| **APEX predictor** | Predicts MIC values (µg/mL) for E. coli; drives reward and greedy selection |
| **MutationEnumerationInTangentSpace** | Identifies which positions and amino acids are reachable along the latent manifold |

---

## Episode flow

An episode starts from a single peptide drawn from the pool and runs for
`n_steps` (default 20) transitions.

```
for each step t in 0 .. n_steps-1:

    1. Encode current peptide → latent vector z  (cached across episodes)

    2. Compute decoder Jacobian J at z  (shape: seq_positions × latent_dim)

    3. SVD:  J = U S Vᵀ
       Principal directions U, singular values S, right-singular vectors V
       define the sub-Riemannian tangent space (SORBES structure)

    4. Enumerate mutations
       MutationEnumerationInTangentSpace uses S and U to identify positions
       whose tangent-space projections exceed the direction_significance_threshold,
       and the amino acids reachable at each such position.

    5. Cap the Cartesian product to MAX_CANDIDATES_PER_STEP = 6 000
       (prevents OOM when SVD yields many mutable positions)

    6. Score candidates with ProjectedDirectionPairwiseSimilarityPotential
       compose_mutant_distribution returns (sequences, log_potentials) ranked
       by the MUTANG++ pairwise similarity score in the horizontal tangent space.

    7. Filter the action set (controlled by --action_filter):
       • top_p  (default 0.90) — nucleus sampling: sort by potential desc,
         keep the smallest prefix whose cumulative softmax mass ≤ top_p.
         At least one candidate is always kept.
       • top_n  (default 20)  — hard cap: keep the top-N candidates by potential.

    8. Exclude the parent sequence from the action set.
       If nothing survives filtering → emit (stuck) step, reward = 0, continue.

    9. Epsilon-greedy selection:
       • explore  (prob ε):   sample an action proportionally to
                              softmax(log_potentials) of the filtered set
       • exploit  (prob 1-ε): score all filtered candidates with APEX,
                              pick the one with minimum mean log₂(MIC) over E. coli

   10. Reward:  r_t = log₂MIC_t − log₂MIC_{t+1}
       Positive reward means MIC decreased (better antimicrobial activity).
       Non-finite rewards are clamped to 0.
```

The episode return is `Σ r_t`.  The global best peptide (lowest log₂MIC) is
tracked across all episodes and all agents.

---

## Epsilon schedule

Epsilon decays once per episode (not per step), after the episode ends:

```
ε_{ep+1} = max(epsilon_end, ε_{ep} × epsilon_decay)
```

Default schedule: `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay=0.9940`.  
This gives a roughly exponential decay from full exploration to 5 % exploration
over the first ~500 episodes.

---

## Caching

Three shared caches persist across all episodes within a single run:

- **latent_cache** `{peptide → z}` — avoids re-encoding visited peptides
- **score_cache** `{peptide → log₂MIC}` — avoids re-querying APEX
- **candidate_cache** `{peptide → (sequences, probs)}` — avoids re-running
  the full MUTANG++ pipeline for peptides visited in earlier episodes

All starting peptides are pre-encoded and pre-scored before episode 1.

---

## Start-peptide selection

Each episode draws its starting peptide from a pool:

- `start_selection=random` (default): uniform random draw each episode
- `start_selection=cycle`: deterministic round-robin over the pool

The pool is loaded from a plain-text file (one peptide per line).  If the file
does not exist or is empty, the built-in 10-peptide `STARTING_PEPTIDES` fallback
is used.

---

## Multi-agent mode

`run_multi_agent` runs `n_agents` independent agents **sequentially**, but all
agents share the same pre-built model objects (encoder-decoder, APEX, enumerator)
to avoid repeated model loading.  Agents use distinct seeds `17, 18, 19, …`
and write separate output files.

GPU memory is freed between agents with `gc.collect()` + `torch.cuda.empty_cache()`.

---

## Reward / loss proxy

There is no learned neural approximator, so there is no gradient-based loss.
The logged loss proxy is:

```
epoch_loss = −mean(step_rewards over the episode)
```

More negative values indicate larger average MIC reduction per step.
Episode return (`Σ r_t`) is the primary performance signal.

---

## CLI

```
# single agent
uv run python scripts/eps_greedy_rl.py run [options]

# multiple agents (models built once, shared)
uv run python scripts/eps_greedy_rl.py multi [options]
```

Key options (both commands):

| Flag | Default | Meaning |
|---|---|---|
| `--n_episodes` | 500 | Episodes per agent |
| `--n_steps` | 20 | Steps per episode |
| `--epsilon_start` | 1.0 | Initial exploration rate |
| `--epsilon_end` | 0.05 | Minimum exploration rate |
| `--epsilon_decay` | 0.9940 | Multiplicative decay per episode |
| `--action_filter` | `top_p` | `top_p` (nucleus) or `top_n` (hard cap) |
| `--top_p` | 0.90 | Nucleus mass threshold |
| `--top_n` | 20 | Hard candidate cap |
| `--device` | `cuda` | PyTorch device |
| `--start_selection` | `random` | `random` or `cycle` |
| `--start_peptides_file` | `basic_eps_greedy_rl/inputs/sampled_500_peptides.txt` | Peptide pool file |
| `--output_dir` | `results/eps_greedy_rl` | Output directory |

---

## Training setup (benchmark runs)

- Number of agents: `5`
- Episodes per agent: `500`
- Steps per episode: `20`
- Device: `cuda` on remote server
- Start pool: `~500` pre-sampled peptides

---

## 500 random peptides + 10 runs each (separate SLURM submission)

This pipeline samples ~500 unique peptides from CSV files in
`results/mutants/mutants/**` and runs 10 independent episodes per peptide
using **6 agents** (SLURM array workers).

Sample the pool:

```bash
<<<<<<< HEAD
uv run python scripts/rl_peptide_optimizer.py run_five_agents --n_agents 5 --n_epochs 1500 --max_steps 200 --device cuda --output_dir results/basic_eps_greedy_rl --start_selection random
```

Generate plots/report files:

```bash
uv run python scripts/basic_eps_greedy_rl_report.py --results_dir results/basic_eps_greedy_rl
```

Outputs:
- `*_log.csv` per agent
- `*_results.json` per agent
- `losses_returns.png`
- `summary.txt`

## 500 random peptides + 10 runs each (separate submission)

This pipeline samples ~500 unique peptides from CSV files in
`results/mutants/mutants/**` and runs
10 independent runs per peptide using **6 agents** (SLURM array workers).

Sampling command (standalone):

```bash
python scripts/sample_peptides_from_csvs.py --root /home/kjurasz/pep-compass --dataset_subdir results/mutants/mutants --sample_size 500 --max_len 25 --seed 2026 --output_file basic_eps_greedy_rl/inputs/sampled_500_peptides.txt --meta_file basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json
=======
python scripts/sample_peptides_from_csvs.py \
  --root /home/kjurasz/pep-compass \
  --dataset_subdir results/mutants/mutants \
  --sample_size 500 --max_len 25 --seed 2026 \
  --output_file basic_eps_greedy_rl/inputs/sampled_500_peptides.txt \
  --meta_file basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json
>>>>>>> 92ef3a097c8c62818d8cb00fa8411f164cac50ca
```

Queue the dataset run:

```bash
sbatch scripts/basic_eps_greedy_500x10_gpu_array.sh
```

Outputs:

- `basic_eps_greedy_rl/inputs/sampled_500_peptides.txt`
- `basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json`
- `results/basic_eps_greedy_rl_500x10/*_results.json`
- `results/basic_eps_greedy_rl_500x10/*_log.csv`
- `results/basic_eps_greedy_rl_500x10/worker_*_summary.json`
- `results/basic_eps_greedy_rl_500x10/worker_*_manifest.json`

---

## Promising start strategy (separate SLURM submission)

Strategy: **coverage-first curriculum starts** — broad peptide pool with
`start_selection=cycle` so each agent traverses distant areas of peptide space
before converging.

```bash
sbatch scripts/basic_eps_greedy_promising_start_gpu.sh
```

Uses **6 agents**, writes to `results/basic_eps_greedy_rl_promising_start`.

---

## Remote (bury.mimuw.edu.pl) workflow

```bash
ssh kjurasz@bury.mimuw.edu.pl
cd ~/pep-compass
git switch rl_trials
git pull
uv run python scripts/eps_greedy_rl.py multi \
  --n_agents 5 --n_episodes 500 --n_steps 20 \
  --device cuda --output_dir results/basic_eps_greedy_rl \
  --start_selection random
```

---

## Output files

Each agent writes:

<<<<<<< HEAD
More negative loss indicates better average improvement per step.
z
=======
- `*_log.csv` — one row per episode: epsilon, start/best peptide, log₂MIC values,
  episode return, stuck-step count, action filter used
- `*_results.json` — full run metadata including all trajectories and hyperparameters
>>>>>>> 92ef3a097c8c62818d8cb00fa8411f164cac50ca
