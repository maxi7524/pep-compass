# Reinforcement Learning for Peptide Optimization in HydrAMP Latent Space

## Overview

This document describes two reinforcement learning approaches for optimizing antimicrobial peptides (AMPs) by navigating the 64-dimensional latent space of the HydrAMP variational autoencoder. Both methods share a common candidate generation pipeline (**MUTANG++**) and scoring function (**APEX MIC predictor**), but differ fundamentally in how the RL agent learns to select among candidates.

| Property | DQN | A2C |
|----------|-----|-----|
| **Script** | `scripts/rl_peptide_optimizer.py` | `scripts/rl_actor_critic_optimizer.py` |
| **Core idea** | Estimate $Q(s, a)$ for each (state, candidate) pair | Learn a direction $\mathbf{d}$ in latent space; score candidates by alignment |
| **Learning signal** | TD(0) with experience replay | Monte Carlo returns with advantage estimation |
| **Exploration** | $\varepsilon$-greedy (guided random) | Temperature-scaled softmax + entropy bonus |
| **Networks** | Q-network + target network | Actor network + critic network |

---

## 1. Shared Components

### 1.1 HydrAMP Latent Space

Both methods operate in the **64-dimensional latent space** of HydrAMP, a conditional VAE for antimicrobial peptides. The ambient (sequence) space is 525-dimensional ($25 \text{ positions} \times 21 \text{ tokens}$, where the 21 tokens are the 20 standard amino acids plus a padding symbol).

- **Encoder**: peptide sequence $\rightarrow$ $\mathbf{z} \in \mathbb{R}^{64}$
- **Decoder**: $\mathbf{z} \in \mathbb{R}^{64}$ $\rightarrow$ probability distribution over sequences
- **Condition**: fixed at $[1.0, 1.0]$ (antimicrobial + active)

### 1.2 MUTANG++ Candidate Generation Pipeline

Both methods use the same candidate generation procedure (the `generate_candidates` function). At each step, given the current peptide $p_t$ and its latent vector $\mathbf{z}_t$:

**Step 1 — Decoder Jacobian SVD.** Compute the Jacobian of the decoder at $\mathbf{z}_t$:

$$J = \frac{\partial \text{Decoder}}{\partial \mathbf{z}} \bigg|_{\mathbf{z}_t} \in \mathbb{R}^{525 \times 64}$$

Then perform a thin SVD: $J = U \Sigma V^\top$, yielding:
- $U \in \mathbb{R}^{525 \times 64}$ — left singular vectors (tangent directions in ambient space)
- $\Sigma \in \mathbb{R}^{64 \times 64}$ — singular values (directional sensitivities)
- $V \in \mathbb{R}^{64 \times 64}$ — right singular vectors (latent directions)

**Step 2 — Tangent-space mutation enumeration.** `MutationEnumerationInTangentSpace` uses the SVD to identify which amino-acid substitutions at each position correspond to significant tangent-space perturbations. Configuration:
- `direction_significance_threshold = 1e-3` (minimum singular value ratio)
- `min_number_of_directions = 5` (always consider at least 5 directions)
- `token_threshold = 0.1` (minimum contribution to consider a token substitution)

**Step 3 — Log-probability ranking.** `DecoderLogProbPotential` scores each candidate mutation by how likely it is under the decoder. The `compose_mutant_distribution` function produces a ranked list of candidate peptides with associated log-potentials.

**Step 4 — Parent removal and softmax filtering.** The parent sequence is excluded (the decoder assigns it the highest log-probability, dominating the distribution). A softmax is applied to the remaining log-potentials, and candidates below the uniform threshold $\frac{1}{1.2k}$ are filtered out, keeping only above-average mutations.

**Step 5 — Encoding.** Surviving candidate peptides are re-encoded to obtain their latent vectors $\{\mathbf{z}_i\}_{i=1}^{n}$ (typically $n \leq 40$).

### 1.3 APEX MIC Scoring

Both methods evaluate candidates using the **APEX MIC predictor**, which estimates the minimum inhibitory concentration (MIC) against pathogens. The score for a peptide is:

$$\text{score}(p) = \frac{1}{3} \sum_{k \in \{1,2,3\}} \log_2\!\bigl(\text{MIC}_k(p)\bigr)$$

where indices 1, 2, 3 correspond to three *E. coli* strains. **Lower score = more potent peptide** (lower MIC).

### 1.4 Reward Design

Both methods use the same reward signal. At each step $t$ within an episode:

$$r_t = \max\!\bigl(0,\; \text{best}_t - \text{score}(p_{t+1})\bigr)$$

where $\text{best}_t$ is the best (lowest) score seen so far in the current episode. The reward is **non-zero only when the agent discovers a new episode-best candidate**, encouraging continued frontier exploration. The cumulative episode reward equals the total $\log_2(\text{MIC})$ improvement achieved:

$$R_{\text{episode}} = \text{score}(p_0) - \min_{t} \text{score}(p_t)$$

### 1.5 Seed Peptides

Six benchmark antimicrobial peptides are used as starting points:

| Name | Sequence | Length |
|------|----------|--------|
| middle-1 | `FLYKWWIRIGRLKL` | 14 |
| jurand-4 | `KYCRRFRWLTFRWL` | 14 |
| jurand-2 | `KFRNRHRWKFKLIFRN` | 16 |
| jurand-7 | `KKYWLIRKWIRLWFLT` | 16 |
| mammuthusin-3 | `KTLKIIRLLF` | 10 |
| hydrodamin-2 | `RMARNLVRYVQGLKKKKVI` | 19 |

---

## 2. Method 1: Deep Q-Network (DQN)

### 2.1 Architecture

The **Q-network** estimates the action-value function $Q(\mathbf{s}, \mathbf{a})$, where both the state $\mathbf{s}$ and action $\mathbf{a}$ are 64-D latent vectors. The input is their concatenation:

$$\text{input} = [\mathbf{s} \| \mathbf{a}] \in \mathbb{R}^{128}$$

**Network structure** (fully connected):

```
Linear(128 → 256) → LayerNorm(256) → ReLU
Linear(256 → 128) → LayerNorm(128) → ReLU
Linear(128 → 64)  → LayerNorm(64)  → ReLU
Linear(64 → 1)
```

Output: scalar Q-value estimate. Two copies are maintained:
- **Online network** $Q_\theta$ — updated via gradient descent
- **Target network** $Q_{\theta^-}$ — periodically synced from online network (hard update every 50 steps)

### 2.2 Experience Replay

Transitions $(\mathbf{s}_t, \mathbf{a}_t, r_t, \mathbf{s}_{t+1}, \text{done}, \mathcal{C}_{t+1})$ are stored in a **circular replay buffer** of capacity 10,000. Each transition also stores the set of candidate latent vectors $\mathcal{C}_{t+1}$ available at the next state (needed for computing $\max_{a'} Q(\mathbf{s}', \mathbf{a}')$).

### 2.3 Training Procedure

At every step, a mini-batch of 32 transitions is sampled uniformly from the replay buffer. The **TD(0) loss** is:

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \bigl(Q_\theta(\mathbf{s}_i, \mathbf{a}_i) - y_i\bigr)^2$$

where the target is:

$$y_i = r_i + \gamma \cdot (1 - \text{done}_i) \cdot \max_{\mathbf{a}' \in \mathcal{C}_{i}^{\text{next}}} Q_{\theta^-}(\mathbf{s}_i', \mathbf{a}')$$

Optimisation details:
- **Optimizer**: Adam with $\text{lr} = 10^{-3}$
- **Discount factor**: $\gamma = 0.99$
- **Gradient clipping**: max norm 1.0
- **Target network sync**: hard copy every 50 update steps

### 2.4 Exploration Strategy — $\varepsilon$-Greedy with Guided Random

The DQN agent uses $\varepsilon$-greedy exploration with a crucial twist: the random fallback is **guided** rather than uniform.

- With probability $\varepsilon$: **guided exploration** — sample a candidate index proportional to the MUTANG++ softmax probabilities (the decoder log-potential distribution). This biases exploration toward structurally plausible mutations.
- With probability $1 - \varepsilon$: **exploitation** — select $\arg\max_{\mathbf{a}} Q_\theta(\mathbf{s}, \mathbf{a})$ over all candidates.

**Epsilon schedule** (per-episode linear decay):

$$\varepsilon_e = \varepsilon_{\text{start}} + \frac{e}{E-1} \cdot (\varepsilon_{\text{end}} - \varepsilon_{\text{start}})$$

- $\varepsilon_{\text{start}} = 1.0$ (fully random in episode 1)
- $\varepsilon_{\text{end}} = 0.05$ (5% random by the last episode)

Additionally, within each episode, a multiplicative per-step decay is applied: $\varepsilon \leftarrow \max(\varepsilon_{\text{end}},\; \varepsilon \times 0.995)$.

### 2.5 Hyperparameter Summary

| Parameter | Value |
|-----------|-------|
| Latent dimension | 64 |
| Q-network hidden layers | 256 → 128 → 64 |
| Learning rate | $10^{-3}$ |
| Discount factor $\gamma$ | 0.99 |
| $\varepsilon_{\text{start}}$ | 1.0 |
| $\varepsilon_{\text{end}}$ | 0.05 |
| $\varepsilon_{\text{decay}}$ (per-step) | 0.995 |
| Batch size | 32 |
| Replay buffer capacity | 10,000 |
| Target network sync | every 50 updates |
| Gradient clip (max norm) | 1.0 |
| Episodes | 100 |
| Steps per episode | 20 |
| Max candidates per step | 40 |

---

## 3. Method 2: Advantage Actor-Critic (A2C)

### 3.1 Architecture

The A2C approach uses two separate networks:

**Actor Network** — outputs a unit direction vector $\mathbf{d} \in \mathbb{R}^{64}$ on the unit sphere $\mathbb{S}^{63}$:

```
Linear(64 → 256) → LayerNorm(256) → ReLU
Linear(256 → 256) → LayerNorm(256) → ReLU
Linear(256 → 64) → L2-normalize
```

$$\mathbf{d} = \frac{f_\phi(\mathbf{z}_t)}{\|f_\phi(\mathbf{z}_t)\|_2}$$

**Critic Network** — estimates the state-value function $V(\mathbf{s})$:

```
Linear(64 → 256) → ReLU
Linear(256 → 128) → ReLU
Linear(128 → 1)
```

Output: scalar state value.

### 3.2 Action Selection — Direction Alignment

Instead of evaluating Q-values for each candidate independently, the actor provides a **direction of desired movement** in latent space. Candidates are scored by how well their displacement from the current state aligns with this direction:

$$\text{score}_i = (\mathbf{z}_i - \mathbf{z}_t) \cdot \mathbf{d}_t = \langle \Delta\mathbf{z}_i,\, \mathbf{d}_t \rangle$$

A temperature-scaled softmax converts scores to a categorical distribution:

$$\pi(a_i \mid \mathbf{z}_t, \mathcal{C}_t) = \frac{\exp\!\bigl(\text{score}_i / \tau\bigr)}{\sum_{j} \exp\!\bigl(\text{score}_j / \tau\bigr)}$$

The action is **sampled** from this distribution during training (or $\arg\max$ during evaluation).

### 3.3 Training Procedure

A2C uses **Monte Carlo returns** (no TD bootstrapping). At the end of each episode, the full trajectory is used for a single batch update.

**Monte Carlo returns:**

$$G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} \, r_{t'}$$

**Advantages:**

$$A_t = G_t - V_\psi(\mathbf{z}_t)$$

**Actor loss** (REINFORCE with baseline):

$$\mathcal{L}_{\text{actor}} = -\sum_{t=0}^{T-1} A_t \cdot \log \pi_\phi(a_t \mid \mathbf{z}_t, \mathcal{C}_t)$$

**Critic loss** (value function regression):

$$\mathcal{L}_{\text{critic}} = \sum_{t=0}^{T-1} \bigl(V_\psi(\mathbf{z}_t) - G_t\bigr)^2$$

**Entropy bonus** (encourages exploration):

$$H_t = -\sum_i \pi(a_i \mid \mathbf{z}_t) \log \pi(a_i \mid \mathbf{z}_t)$$

**Total loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + 0.5 \cdot \mathcal{L}_{\text{critic}} - \beta \sum_t H_t$$

Optimisation details:
- **Optimizer**: Adam with $\text{lr} = 3 \times 10^{-4}$ (joint for actor and critic)
- **Discount factor**: $\gamma = 0.99$
- **Gradient clipping**: max norm 0.5
- No replay buffer — on-policy learning, trajectories discarded after each update

### 3.4 Exploration Strategy — Temperature and Entropy Scheduling

Two mechanisms control the exploration-exploitation trade-off, both annealed linearly over episodes:

**Softmax temperature** $\tau$ — controls the sharpness of the action distribution:

$$\tau_e = \tau_{\text{start}} + \frac{e}{E-1}(\tau_{\text{end}} - \tau_{\text{start}})$$

- $\tau_{\text{start}} = 2.0$ (very exploratory — nearly uniform selection)
- $\tau_{\text{end}} = 0.2$ (sharply peaked — nearly greedy)

**Entropy bonus coefficient** $\beta$ — penalises low-entropy policies:

$$\beta_e = \beta_{\text{start}} + \frac{e}{E-1}(\beta_{\text{end}} - \beta_{\text{start}})$$

- $\beta_{\text{start}} = 0.1$ (strong entropy incentive)
- $\beta_{\text{end}} = 0.01$ (weak entropy incentive)

### 3.5 Hyperparameter Summary

| Parameter | Value |
|-----------|-------|
| Latent dimension | 64 |
| Actor hidden layers | 256 → 256 (LayerNorm + ReLU) |
| Critic hidden layers | 256 → 128 (ReLU) |
| Learning rate | $3 \times 10^{-4}$ |
| Discount factor $\gamma$ | 0.99 |
| Temperature $\tau$ | 2.0 → 0.2 (linear) |
| Entropy $\beta$ | 0.1 → 0.01 (linear) |
| Critic loss weight | 0.5 |
| Gradient clip (max norm) | 0.5 |
| Episodes | 100 |
| Steps per episode | 20 |
| Max candidates per step | 40 |

---

## 4. Comparison

### 4.1 Key Architectural Differences

| Aspect | DQN | A2C |
|--------|-----|-----|
| **What the network learns** | $Q(\mathbf{s}, \mathbf{a}) \in \mathbb{R}$ for each candidate | Direction $\mathbf{d} \in \mathbb{S}^{63}$ + value $V(\mathbf{s}) \in \mathbb{R}$ |
| **Input to network** | $[\mathbf{s} \| \mathbf{a}] \in \mathbb{R}^{128}$ (evaluated per candidate) | $\mathbf{s} \in \mathbb{R}^{64}$ (one forward pass) |
| **Action selection cost** | $O(n)$ forward passes through Q-network | 1 forward pass + $O(n)$ dot products |
| **Learning paradigm** | Off-policy (replay buffer) | On-policy (trajectory-based) |
| **Temporal credit** | TD(0) bootstrapping | Monte Carlo full returns |
| **State representation** | Q-value landscape over action space | Geometric direction in latent space |
| **Normalisation** | LayerNorm in all hidden layers | LayerNorm in actor only |

### 4.2 Theoretical Advantages and Disadvantages

**DQN Advantages:**
- **Sample efficiency**: Off-policy learning with experience replay reuses past transitions, extracting more learning signal per interaction.
- **Stability**: Target network prevents oscillating Q-value estimates.
- **Fine-grained discrimination**: Evaluates each candidate independently, can distinguish subtle differences between similar candidates.

**DQN Disadvantages:**
- **Computational cost**: Requires $n$ forward passes per action selection (one per candidate).
- **Variable action space**: The set of candidates changes at every step, so Q-values cannot be precomputed or generalised across steps easily.
- **No principled exploration**: $\varepsilon$-greedy with softmax-guided random is heuristic; no explicit incentive for policy entropy or uncertainty.

**A2C Advantages:**
- **Geometric interpretability**: The actor learns a direction in latent space, which has a clear geometric meaning — "move this way to improve".
- **Efficient action selection**: A single forward pass through the actor produces scores for all candidates via dot products ($O(n \cdot d)$ vs $O(n)$ forward passes).
- **Principled exploration**: Entropy bonus in the loss function explicitly encourages diverse action selection during early training.
- **Smooth policy updates**: Policy gradient with advantage estimation provides lower-variance gradients than naive REINFORCE.

**A2C Disadvantages:**
- **Sample inefficiency**: On-policy — each trajectory is used once and discarded. No replay buffer.
- **High variance**: Monte Carlo returns (no bootstrapping) can have high variance, especially with sparse rewards.
- **Direction assumption**: The dot-product scoring assumes that "good" candidates lie in a consistent direction from the current state, which may not hold in regions with high curvature in the latent space.

### 4.3 Conceptual Framing

The DQN approach treats the problem as a **discrete choice** at each step: "which candidate has the highest long-term value?" It learns a value landscape over the joint state-action space.

The A2C approach treats the problem as **navigation on a manifold**: "which direction should I move in latent space to improve the peptide?" It learns a flow field $\mathbf{d}(\mathbf{z})$ that guides exploration, with MUTANG++ providing the discrete candidate set to choose from.

---

## 5. Experimental Results (100 Episodes, 20 Steps/Episode)

### 5.1 DQN Results

| Seed Peptide | Start Sequence | Start $\log_2(\text{MIC})$ | Start MIC (µM) | Best Sequence Found | Best $\log_2(\text{MIC})$ | Best MIC (µM) | Improvement ($\Delta \log_2$) | Fold Improvement |
|---|---|---|---|---|---|---|---|---|
| middle-1 | `FLYKWWIRIGRLKL` | 4.001 | 16.01 | `RKYWWKVRRMRWWL` | 2.560 | 5.90 | 1.441 | 2.71× |
| jurand-4 | `KYCRRFRWLTFRWL` | 2.879 | 7.36 | `KRFRRFRWLKFRWL` | 2.443 | 5.44 | 0.436 | 1.35× |
| jurand-2 | `KFRNRHRWKFKLIFRN` | 3.386 | 10.46 | `WFRKRIRWKFKLVFRL` | 2.266 | 4.81 | 1.120 | 2.17× |
| jurand-7 | `KKYWLIRKWIRLWFLT` | 2.709 | 6.54 | `KKYWLIRKWIRLWFKL` | 2.086 | 4.25 | 0.623 | 1.54× |
| mammuthusin-3 | `KTLKIIRLLF` | 4.304 | 19.76 | `KTLKIIRLLF` | 4.304 | 19.76 | 0.000 | 1.00× |
| hydrodamin-2 | `RMARNLVRYVQGLKKKKVI` | 6.962 | 124.64 | `RLGRKLVRFVKGLKKWKWW` | 2.858 | 7.25 | 4.104 | 17.20× |

### 5.2 A2C Results

| Seed Peptide | Start Sequence | Start $\log_2(\text{MIC})$ | Start MIC (µM) | Best Sequence Found | Best $\log_2(\text{MIC})$ | Best MIC (µM) | Improvement ($\Delta \log_2$) | Fold Improvement |
|---|---|---|---|---|---|---|---|---|
| middle-1 | `FLYKWWIRIGRLKL` | 4.001 | 16.01 | `RWYWWKVRRMRWWL` | 2.457 | 5.49 | 1.544 | 2.92× |
| jurand-4 | `KYCRRFRWLTFRWL` | 2.879 | 7.36 | `KRFRRFRWLKFRWL` | 2.443 | 5.44 | 0.436 | 1.35× |
| jurand-2 | `KFRNRHRWKFKLIFRN` | 3.386 | 10.46 | `WFRKRIRLWFKLVRRL` | 2.155 | 4.45 | 1.231 | 2.35× |
| jurand-7 | `KKYWLIRKWIRLWFLT` | 2.709 | 6.54 | `RWCWKIRKWKWLKWWL` | 1.983 | 3.95 | 0.726 | 1.65× |
| mammuthusin-3 | `KTLKIIRLLF` | 4.304 | 19.76 | `KTLKIIRLLF` | 4.304 | 19.76 | 0.000 | 1.00× |
| hydrodamin-2 | `RMARNLVRYVQGLKKKKVI` | 6.962 | 124.64 | `RLSRKLVRYVKGLWKWKRW` | 2.434 | 5.40 | 4.528 | 23.07× |

### 5.3 Head-to-Head Comparison

| Seed Peptide | DQN Best MIC (µM) | A2C Best MIC (µM) | Winner | DQN Fold | A2C Fold |
|---|---|---|---|---|---|
| middle-1 | 5.90 | **5.49** | A2C | 2.71× | **2.92×** |
| jurand-4 | 5.44 | 5.44 | Tie | 1.35× | 1.35× |
| jurand-2 | 4.81 | **4.45** | A2C | 2.17× | **2.35×** |
| jurand-7 | 4.25 | **3.95** | A2C | 1.54× | **1.65×** |
| mammuthusin-3 | 19.76 | 19.76 | Tie | 1.00× | 1.00× |
| hydrodamin-2 | 7.25 | **5.40** | A2C | 17.20× | **23.07×** |

**Summary**: A2C outperformed or matched DQN on all six seed peptides. The largest difference was on hydrodamin-2, where A2C achieved a 23.07× fold improvement vs. DQN's 17.20×. Neither method was able to improve mammuthusin-3, a short 10-residue peptide whose local neighbourhood in latent space appears to be a basin with no improving mutations.

### 5.4 Notable Observations

1. **hydrodamin-2** showed the most dramatic improvements for both methods (17–23× fold), likely because the seed peptide had the highest initial MIC (124.64 µM), leaving the most room for improvement.

2. **mammuthusin-3** could not be improved by either method. At only 10 residues, it is the shortest seed peptide. The MUTANG++ pipeline may generate fewer viable single-position mutations for such a short sequence, and the latent-space landscape around it may be locally flat with respect to the MIC objective.

3. **jurand-4** converged to the same best peptide (`KRFRRFRWLKFRWL`) with both methods, suggesting this is a strong local optimum reachable from the seed in a small number of mutations.

4. Both methods found qualitatively similar peptides for middle-1 (`RKYWWKVRRMRWWL` vs `RWYWWKVRRMRWWL`), differing by a single residue, indicating they converge to the same high-quality region.

---

## 6. Episode and Step Structure

Both methods share the same episode structure:

```
for episode in 1..E:
    z_t = encode(start_peptide)      # or best_peptide if start_from_best
    for step in 1..T:
        candidates = MUTANG++(z_t)   # generate & encode candidates
        a_t = agent.select(z_t, candidates)
        z_{t+1} = candidates[a_t]
        r_t = max(0, best_so_far - score(z_{t+1}))
        agent.learn(z_t, a_t, r_t, z_{t+1})  # DQN: per-step TD | A2C: store
        z_t = z_{t+1}
    # A2C only: batch update at end of episode
```

With 100 episodes × 20 steps = **2,000 candidate evaluations** per seed peptide. Both methods use caching to avoid recomputing candidates or scores for previously visited peptides.

---

## 7. Implementation Notes

- Both scripts use **`fire`** for CLI argument parsing, enabling direct command-line invocation with keyword arguments.
- Results are saved as JSON (full results including trajectories) and CSV (per-episode log with scores and hyperparameter schedules).
- The DQN script defines all shared utilities (`build_encoder_decoder`, `build_apex_predictor`, `score_peptides`, `compute_jacobian_svd`, `generate_candidates`); the A2C script imports these directly.
- The `start_from_best` option (available in both methods) makes each new episode start from the best peptide found so far, encouraging continued frontier exploration rather than re-traversing from the same seed.
- DQN uses `per_episode_epsilon` to reset $\varepsilon$ at the beginning of each episode via a linear schedule, preventing premature convergence to the first local optimum.
