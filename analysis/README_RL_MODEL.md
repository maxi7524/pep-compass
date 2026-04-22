# Basic epsilon-greedy RL model

Current RL baseline:

- `scripts/rl_peptide_optimizer.py` (`run_basic_epsilon_greedy`, `run_five_agents`)
- `scripts/basic_eps_greedy_rl_report.py` (loss/return plotting)

The previous DQN/A2C policies were removed and replaced by a minimal
epsilon-greedy strategy over mutation candidates.

## Objective

Minimize mean `log2(MIC)` on selected E. coli targets from APEX.

Lower score is better.

## Action space and filtering

Candidates are generated from local mutation enumeration and scored by
softmaxed similarity potential. Actions are filtered with:

`p(action) > 2 / N`

where `N` is number of candidates. If no action survives, the argmax action is
retained. If there are no mutations at all, the parent peptide is used as a
safe fallback action (reward `0`).

## Reward and logged loss

- Step reward: `r_t = log2MIC_t - log2MIC_{t+1}`
- Epoch return: `sum_t r_t`
- Logged policy loss proxy: `-mean_t r_t`

## Multi-start and 5-agent runs

Use multiple start peptides (inline or file), random/cycle start selection, and
run 5 independent agents:

```bash
uv run python scripts/rl_peptide_optimizer.py run_five_agents --n_agents 5 --n_epochs 1500 --max_steps 200 --device cuda --output_dir results/basic_eps_greedy_rl --start_selection random
```
