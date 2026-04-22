# basic_eps_greedy_rl

This folder documents the simplest RL baseline used in this repository now:
a **basic epsilon-greedy policy** that minimizes APEX mean log2(MIC) over E. coli.

## Policy definition

At each step:
1. Enumerate local mutation candidates from the current peptide.
2. Score candidate mutation tuples with similarity potential and softmax.
3. Keep actions with probability `p(action) > 2 / N` where `N` is candidate count.
4. If nothing passes threshold, keep only the top-probability action.
5. Choose action with epsilon-greedy:
   - explore (`epsilon`): sample from softmax probabilities
   - exploit (`1-epsilon`): select candidate with smallest APEX score

### No-mutation handling

If mutation enumeration yields no valid actions, the policy emits the current
peptide as the only action, reward is `0`, and training continues safely.

## Training setup requested

- Number of agents: `5`
- Epochs per agent: `1500`
- Steps per epoch: `200`
- Device: `cuda` on remote server
- Start peptides: multiple peptides (default: six benchmark seeds)

## Local commands

Run 5 agents:

```bash
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

This pipeline samples ~500 unique peptides from CSV files and runs
10 independent runs per peptide using **6 agents** (SLURM array workers).

Sampling command (standalone):

```bash
python scripts/sample_peptides_from_csvs.py --root /home/kjurasz/pep-compass --sample_size 500 --max_len 25 --seed 2026 --output_file basic_eps_greedy_rl/inputs/sampled_500_peptides.txt --meta_file basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json
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

## Promising start strategy (separate submission)

Proposed strategy: **coverage-first curriculum starts**.  
Use a broad peptide pool and cycle starts deterministically (`start_selection=cycle`)
so each agent repeatedly traverses distant areas of peptide space before converging.

Queue this run:

```bash
sbatch scripts/basic_eps_greedy_promising_start_gpu.sh
```

This run uses **6 agents**, `1500` epochs, `200` steps, and writes to:
- `results/basic_eps_greedy_rl_promising_start`

## Remote (bury.mimuw.edu.pl) workflow

```bash
ssh kjurasz@bury.mimuw.edu.pl
cd ~/pep-compass
git switch rl_trials
git pull
uv run python scripts/rl_peptide_optimizer.py run_five_agents --n_agents 5 --n_epochs 1500 --max_steps 200 --device cuda --output_dir results/basic_eps_greedy_rl --start_selection random
uv run python scripts/basic_eps_greedy_rl_report.py --results_dir results/basic_eps_greedy_rl
```

## Loss definition

There is no learned neural approximator in this baseline, so we log a direct
policy proxy loss:

`epoch_loss = -mean(step_rewards)`, where `step_reward = log2MIC_t - log2MIC_{t+1}`.

More negative loss indicates better average improvement per step.
z