# RL Multi-Start Peptide Optimizer (Similarity-Potential Variant)

This document describes the updated RL methodology implemented in:

- `scripts/rl_peptide_optimizer.py` (DQN)
- `scripts/rl_actor_critic_optimizer.py` (A2C)

## Objective

Optimize peptides by minimizing mean `log2(MIC)` over selected E. coli targets.

Lower score is better.

## Multi-Start Training

Runs can start from multiple seed peptides in one training session.

Inputs:

- `start_peptides` (inline CSV/space/newline list)
- `start_peptides_file` (dataset file)
- `start_selection` (`cycle` or `random`)

If not provided, `start_peptide` is used for backward compatibility.

## State / Action / Transition

- **State**: current peptide and latent vector (HydrAMP 64-D).
- **Actions**: local MutAgg++ mutation candidates.
- **Transition**: selected candidate becomes next peptide/state.

## Reward and Return

Step reward:

`r_t = log2MIC_t - log2MIC_{t+1}`

- Positive: improved potency
- Negative: worsened potency

Discounted return:

`G_t = Σ_{i=t}^{T-1} gamma^(i-t) * r_i`

Default `gamma = 0.99`, default horizon `max_steps = 20`.

## Action Filtering Rule

Let mutation dictionary be `mutations[pos] -> candidate_aas`.

- `k = Π_pos |mutations[pos]|` (cartesian mutation count)
- Convert potentials to softmax probabilities
- Keep actions with `p(action) > 1/k`
- If none pass, keep argmax-probability action

## Similarity Potential (Default)

Default action scoring uses:

- `ProjectedDirectionPairwiseSimilarityPotential`

Optional fallback:

- `DecoderLogProbPotential` via `potential_type=decoder_logprob`

## Performance Safeguards

To control combinatorial cost:

- normalize/deduplicate mutation sets
- cap mutable positions (`similarity_max_positions`)
- cap mutations per position (`similarity_max_mutations_per_position`)
- apply combination budget (`similarity_max_combinations`)
- optional sampling cap (`similarity_sample_combinations`)

## Algorithms

### DQN

- Q-network over `(state_latent, action_latent)`
- replay buffer + target network
- epsilon-greedy action choice over filtered candidates

### A2C

- actor outputs latent direction
- candidate selection by direction alignment
- critic estimates state value
- Monte Carlo returns + entropy regularization

## Outputs

JSON/CSV outputs include:

- `start_peptides`, `start_selection_mode`
- `reward_formula`
- `discount_gamma`
- `episode_rewards`
- `episode_discounted_returns`
- `potential_type`

Saved under `results/*_results.json` and `results/*_log.csv`.
