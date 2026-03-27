"""
RL Actor-Critic Peptide Optimizer (A2C) using MUTANG++ in the HydrAMP latent space.

This approach keeps MUTANG++ as the core candidate generator while using a continuous
actor network to guide selection. The actor outputs a query direction d in 64-D latent
space, and candidates are scored by their alignment with this direction.

Key differences from DQN:
- Actor outputs continuous 64-D direction (L2-normalised), not Q-values
- Critic estimates state value V(s), not action-value Q(s,a)
- Monte Carlo returns with advantage estimation (A2C, no TD bootstrapping)
- Entropy bonus for exploration, temperature-based softmax over candidate scores

Usage:
    python rl_actor_critic_optimizer.py run_a2c_optimization --start_peptide FLPKKVIPLL
    python rl_actor_critic_optimizer.py run_a2c_all_peptides --peptide_name middle-1
    python rl_actor_critic_optimizer.py test_components
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import shared utilities from DQN optimizer
sys.path.insert(0, os.path.dirname(__file__))
from rl_peptide_optimizer import (
    LATENT_DIM,
    SEED_PEPTIDES,
    MAX_PEPTIDE_LEN,
    build_encoder_decoder,
    build_apex_predictor,
    score_peptides,
    generate_candidates,
    discounted_sum,
    resolve_start_peptides,
)


# ──────────────────────────────────────────────────────────────────────────────
# Actor Network
# ──────────────────────────────────────────────────────────────────────────────


class ActorNetwork(nn.Module):
    """Actor network that outputs a direction vector in latent space.

    Given a state z (64-D latent vector), outputs a unit direction d ∈ ℝ⁶⁴.
    Candidates are scored by dot(z_i - z_t, d) — how well they align with
    the actor's preferred exploration direction.

    Architecture: 64 → 256 → 256 → 64, with LayerNorm + ReLU, L2 normalisation.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # output raw direction (64-D)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised direction vector(s).

        Parameters
        ----------
        state : ``(..., LATENT_DIM)``

        Returns
        -------
        d : ``(..., LATENT_DIM)`` unit direction vector(s).
        """
        raw = self.net(state)
        # L2 normalise to unit sphere
        d = F.normalize(raw, p=2, dim=-1)
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Critic Network
# ──────────────────────────────────────────────────────────────────────────────


class CriticNetwork(nn.Module):
    """State-value critic V(s) for advantage estimation.

    Architecture: 64 → 256 → 128 → 1 with ReLU activations.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value V(s).

        Parameters
        ----------
        state : ``(..., LATENT_DIM)``

        Returns
        -------
        v : ``(..., 1)`` or ``(...)`` scalar state value(s).
        """
        return self.net(state).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# A2C Agent
# ──────────────────────────────────────────────────────────────────────────────


class A2CAgent:
    """Actor-Critic agent using continuous direction guidance over MUTANG++ candidates.

    The actor outputs a direction d = actor(z_t) in latent space.
    Candidates z_i are scored by: score_i = dot(z_i - z_t, d).
    Softmax over scores (with temperature τ) gives action probabilities.

    Training uses Monte Carlo returns (no bootstrapping):
    - Advantages: A_t = G_t - V(z_t)
    - Actor loss: -Σ A_t * log π(a_t | z_t, candidates)
    - Critic loss: MSE(V(z_t), G_t)
    - Entropy bonus: -β * H(π) for exploration

    Parameters
    ----------
    latent_dim : Dimensionality of the latent space (default 64).
    lr : Learning rate for the Adam optimiser.
    gamma : Discount factor γ.
    device : PyTorch device string.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.device = device

        self.actor = ActorNetwork(latent_dim).to(device)
        self.critic = CriticNetwork(latent_dim).to(device)

        # Joint optimiser for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # Per-episode trajectory storage
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []  # chosen candidate index
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []

    def reset_trajectory(self) -> None:
        """Clear trajectory buffers for a new episode."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.entropies.clear()

    def select_action(
        self,
        current_z: np.ndarray,
        candidate_zs: np.ndarray,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> int:
        """Select a candidate via softmax over direction-alignment scores.

        Parameters
        ----------
        current_z : Current latent state, shape ``(LATENT_DIM,)``.
        candidate_zs : Encoded candidates, shape ``(n, LATENT_DIM)``.
        temperature : Softmax temperature (higher = more exploration).
        deterministic : If True, use argmax instead of sampling.

        Returns
        -------
        int : Index of the chosen candidate.
        """
        n = len(candidate_zs)
        if n == 0:
            return 0

        self.actor.eval()
        with torch.no_grad():
            state_t = torch.tensor(
                current_z, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 64)
            direction = self.actor(state_t).squeeze(0)  # (64,)

        # Compute candidate displacement vectors
        candidates_t = torch.tensor(
            candidate_zs, dtype=torch.float32, device=self.device
        )  # (n, 64)
        state_t_expanded = state_t.expand(n, -1)  # (n, 64)
        displacements = candidates_t - state_t_expanded  # (n, 64)

        # Score = projection onto direction (dot product)
        scores = torch.matmul(displacements, direction)  # (n,)

        # Apply temperature and softmax
        probs = F.softmax(scores / temperature, dim=0)  # (n,)

        if deterministic:
            action_idx = int(probs.argmax().item())
        else:
            # Sample from categorical distribution
            action_idx = int(torch.multinomial(probs, num_samples=1).item())

        # Store log_prob and entropy for training
        log_prob = torch.log(probs[action_idx] + 1e-10)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

        return action_idx

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        """Store a transition for later batch update."""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)

    def update(self, entropy_beta: float = 0.1) -> tuple[float, float, float]:
        """Perform A2C update using the stored trajectory.

        Uses Monte Carlo returns (no bootstrapping) for computing advantages.

        Parameters
        ----------
        entropy_beta : Coefficient for entropy bonus in the loss.

        Returns
        -------
        actor_loss, critic_loss, entropy : Individual loss components.
        """
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0

        self.actor.train()
        self.critic.train()

        # Convert trajectory to tensors
        states = torch.tensor(
            np.stack(self.states), dtype=torch.float32, device=self.device
        )  # (T, 64)
        rewards = self.rewards  # list of floats

        # Compute Monte Carlo returns G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}
        T = len(rewards)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Compute state values V(s)
        values = self.critic(states)  # (T,)

        # Compute advantages A_t = G_t - V(z_t)
        advantages = returns - values.detach()

        # Stack log_probs and entropies
        log_probs = torch.stack(self.log_probs)  # (T,)
        entropies = torch.stack(self.entropies)  # (T,)

        # Actor loss: -Σ A_t * log π(a_t | z_t)
        actor_loss = -(advantages * log_probs).sum()

        # Critic loss: MSE(V(z_t), G_t)
        critic_loss = F.mse_loss(values, returns, reduction="sum")

        # Entropy bonus (negative because we want to maximise entropy)
        entropy_bonus = entropies.sum()

        # Total loss: actor + 0.5 * critic - entropy_bonus
        total_loss = actor_loss + 0.5 * critic_loss - entropy_beta * entropy_bonus

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()

        return (
            actor_loss.item(),
            critic_loss.item(),
            entropy_bonus.item(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main optimisation loop
# ──────────────────────────────────────────────────────────────────────────────


def run_a2c_optimization(
    start_peptide: str = "FLPKKVIPLL",
    start_peptides: str = "",
    start_peptides_file: str = "",
    start_selection: str = "cycle",
    n_episodes: int = 100,
    max_steps: int = 20,
    max_candidates: int = 40,
    device: str = "cpu",
    verbose: bool = True,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_start: float = 0.1,
    entropy_end: float = 0.01,
    temp_start: float = 2.0,
    temp_end: float = 0.2,
    output_dir: str = "results",
    start_from_best: bool = False,
    potential_type: str = "similarity",
    mutation_direction_significance_threshold: float = 1e-3,
    mutation_min_number_of_directions: int = 5,
    mutation_token_threshold: float = 0.1,
    similarity_horizontal_threshold: float = 1e-3,
    similarity_max_positions: int = 5,
    similarity_max_mutations_per_position: int = 6,
    similarity_max_combinations: int = 40_000,
    similarity_sample_combinations: int = 8_000,
    similarity_include_parent_residue: bool = False,
    run_name: str = "",
) -> None:
    """Run A2C-based peptide optimisation in the HydrAMP latent space.

    The actor learns a direction in latent space that guides selection among
    MUTANG++-generated candidates. This keeps MUTANG++ as the core candidate
    generator while enabling continuous policy learning.

    Parameters
    ----------
    start_peptide : Fallback single seed peptide.
    start_peptides : Inline peptide dataset (comma/space/newline separated).
    start_peptides_file : Path to peptide dataset file.
    start_selection : Episode start selector: ``cycle`` or ``random``.
    n_episodes : Number of training episodes.
    max_steps : Maximum transitions per episode.
    max_candidates : Maximum number of mutant candidates per step.
    device : PyTorch device ("cpu" or "cuda").
    verbose : Print per-episode summaries when True.
    lr : Learning rate for actor and critic.
    gamma : Discount factor.
    entropy_start : Initial entropy bonus coefficient.
    entropy_end : Final entropy bonus coefficient.
    temp_start : Initial softmax temperature.
    temp_end : Final softmax temperature.
    output_dir : Directory to save results JSON and CSV.
    start_from_best : If True, each episode starts from the best peptide found
                      so far instead of sampled dataset starts.
    potential_type : Candidate potential type (``similarity`` or ``decoder_logprob``).
    run_name : Optional tag prepended to the output file names.
    """
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        DecoderLogProbPotential,
    )

    selection = start_selection.strip().lower()
    if selection not in {"cycle", "random"}:
        raise ValueError("start_selection must be one of: 'cycle', 'random'")

    # ── build models ──────────────────────────────────────────────────────────
    if verbose:
        print("Loading models …")
    encoder_decoder = build_encoder_decoder(device)
    apex = build_apex_predictor(device)

    mutation_enumerator = MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=mutation_direction_significance_threshold,
        min_number_of_directions=mutation_min_number_of_directions,
        token_threshold=mutation_token_threshold,
    )
    log_prob_potential = DecoderLogProbPotential(encoder_decoder=encoder_decoder)

    agent = A2CAgent(
        latent_dim=LATENT_DIM,
        lr=lr,
        gamma=gamma,
        device=device,
    )

    # ── resolve and encode start peptide dataset ──────────────────────────────
    start_pool = resolve_start_peptides(
        start_peptide=start_peptide,
        start_peptides=start_peptides,
        start_peptides_file=start_peptides_file,
    )
    if verbose:
        print(f"Encoding {len(start_pool)} start peptide(s) …")
    with torch.no_grad():
        start_z_tensor: torch.Tensor = encoder_decoder.encode_peptides(start_pool)
    start_zs: np.ndarray = start_z_tensor.detach().cpu().numpy()
    start_scores_arr = score_peptides(apex, start_pool)

    run_start_peptide = start_pool[0]
    run_start_score = float(start_scores_arr[0])
    best_init_idx = int(np.argmin(start_scores_arr))
    best_peptide: str = start_pool[best_init_idx]
    best_score: float = float(start_scores_arr[best_init_idx])
    all_start_scores: list[float] = [float(x) for x in start_scores_arr.tolist()]

    _latent_cache: dict[str, np.ndarray] = {
        pep: start_zs[i].copy() for i, pep in enumerate(start_pool)
    }
    _score_cache: dict[str, float] = {
        pep: float(start_scores_arr[i]) for i, pep in enumerate(start_pool)
    }

    def _get_latent(peptide: str) -> np.ndarray:
        if peptide not in _latent_cache:
            with torch.no_grad():
                _z = encoder_decoder.encode_peptides([peptide])
            _latent_cache[peptide] = _z.detach().cpu().numpy()[0]
        return _latent_cache[peptide]

    def _get_score(peptide: str) -> float:
        if peptide not in _score_cache:
            _score_cache[peptide] = float(score_peptides(apex, [peptide])[0])
        return _score_cache[peptide]

    if verbose:
        print(
            "Start dataset loaded: "
            f"count={len(start_pool)}  first={run_start_peptide!r} ({run_start_score:.4f})  "
            f"best={best_peptide!r} ({best_score:.4f})"
        )

    # ── candidate cache (keyed by peptide string) ────────────────────────────
    _cand_cache: dict[str, tuple] = {}

    def _get_candidates(peptide: str, z: np.ndarray) -> tuple:
        if peptide not in _cand_cache:
            _cand_cache[peptide] = generate_candidates(
                peptide=peptide,
                z_np=z,
                encoder_decoder=encoder_decoder,
                mutation_enumerator=mutation_enumerator,
                potential_type=potential_type,
                similarity_horizontal_threshold=similarity_horizontal_threshold,
                similarity_max_positions=similarity_max_positions,
                similarity_max_mutations_per_position=similarity_max_mutations_per_position,
                similarity_max_combinations=similarity_max_combinations,
                similarity_sample_combinations=similarity_sample_combinations,
                similarity_include_parent_residue=similarity_include_parent_residue,
                log_prob_potential=log_prob_potential,
                max_candidates=max_candidates,
            )
        return _cand_cache[peptide]

    # ── output dir + run id ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{run_name}_" if run_name else ""
    run_id = f"{prefix}a2c_{int(time.time())}"

    # ── CSV log (per-episode) ─────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{run_id}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode",
        "run_start_mode", "run_start_count",
        "run_start_peptide", "run_start_log2mic",
        "ep_start_peptide", "ep_start_log2mic",
        "ep_best_peptide", "ep_best_log2mic",
        "ep_return_log2mic",
        "ep_discounted_return_log2mic",
        "global_best_peptide", "global_best_log2mic",
        "temperature", "entropy_beta",
    ])
    csv_file.flush()

    # ── tracking containers ───────────────────────────────────────────────────
    episode_rewards: list[float] = []
    episode_discounted_returns: list[float] = []
    all_best_scores: list[float] = []
    all_best_peptides: list[str] = []
    trajectories: list[list[str]] = []

    # ── episode loop ──────────────────────────────────────────────────────────
    for ep in range(n_episodes):
        # ── compute per-episode schedules ─────────────────────────────────────
        frac = ep / max(n_episodes - 1, 1)
        temperature = temp_start + frac * (temp_end - temp_start)
        entropy_beta = entropy_start + frac * (entropy_end - entropy_start)

        # Reset trajectory for the new episode
        agent.reset_trajectory()

        # ── episode starting point ────────────────────────────────────────────
        if start_from_best and ep > 0:
            current_peptide = best_peptide
            current_z = _get_latent(current_peptide).copy()
            current_score = _get_score(current_peptide)
        else:
            if selection == "random":
                current_peptide = str(np.random.choice(start_pool))
            else:
                current_peptide = start_pool[ep % len(start_pool)]
            current_z = _get_latent(current_peptide).copy()
            current_score = _get_score(current_peptide)

        ep_reward = 0.0
        step_rewards: list[float] = []
        ep_start_score = current_score
        ep_start_peptide = current_peptide
        ep_best_score = current_score
        ep_best_peptide = current_peptide
        trajectory: list[str] = [current_peptide]

        # Pre-generate candidates (cached)
        cand_seqs, cand_probs, cand_zs = _get_candidates(current_peptide, current_z)

        for step in range(max_steps):
            # ── select action ─────────────────────────────────────────────────
            action_idx: int = agent.select_action(
                current_z, cand_zs, temperature=temperature
            )
            chosen_seq: str = cand_seqs[action_idx]
            chosen_z: np.ndarray = cand_zs[action_idx].copy()  # (64,)

            # ── evaluate chosen candidate (cached) ────────────────────────────
            next_score: float = _get_score(chosen_seq)

            # Reward = consecutive score delta in log2(MIC).
            reward: float = current_score - next_score
            if not np.isfinite(reward):
                reward = 0.0
            step_rewards.append(reward)
            ep_reward += reward

            # ── store transition ──────────────────────────────────────────────
            agent.store_transition(
                state=current_z,
                action=action_idx,
                reward=reward,
            )

            # ── bookkeeping ───────────────────────────────────────────────────
            if next_score < ep_best_score:
                ep_best_score = next_score
                ep_best_peptide = chosen_seq
            if next_score < best_score:
                best_score = next_score
                best_peptide = chosen_seq

            trajectory.append(chosen_seq)

            # ── generate next-step candidates ─────────────────────────────────
            if step < max_steps - 1:
                next_cand_seqs, next_cand_probs, next_cand_zs = _get_candidates(
                    chosen_seq, chosen_z
                )
            else:
                next_cand_seqs = []
                next_cand_probs = np.empty(0)
                next_cand_zs = np.empty((0, LATENT_DIM))

            # ── advance state ─────────────────────────────────────────────────
            current_z = chosen_z
            current_score = next_score
            current_peptide = chosen_seq
            _latent_cache[chosen_seq] = chosen_z.copy()

            # Reuse next candidates as current candidates for the next step
            if step < max_steps - 1:
                cand_seqs = next_cand_seqs
                cand_probs = next_cand_probs
                cand_zs = next_cand_zs

                if len(cand_seqs) == 0:
                    cand_seqs, cand_probs, cand_zs = _get_candidates(
                        current_peptide, current_z
                    )

        # ── end of episode: A2C update ────────────────────────────────────────
        actor_loss, critic_loss, entropy_val = agent.update(entropy_beta=entropy_beta)

        # ── episode summary ───────────────────────────────────────────────────
        ep_improvement = ep_start_score - ep_best_score
        ep_discounted_return = discounted_sum(step_rewards, gamma)
        episode_rewards.append(ep_reward)
        episode_discounted_returns.append(ep_discounted_return)
        all_best_scores.append(ep_best_score)
        all_best_peptides.append(ep_best_peptide)
        trajectories.append(trajectory)

        # write CSV row
        csv_writer.writerow([
            ep + 1,
            selection, len(start_pool),
            run_start_peptide, f"{run_start_score:.6f}",
            ep_start_peptide, f"{ep_start_score:.6f}",
            ep_best_peptide, f"{ep_best_score:.6f}",
            f"{ep_reward:.6f}",
            f"{ep_discounted_return:.6f}",
            best_peptide, f"{best_score:.6f}",
            f"{temperature:.4f}", f"{entropy_beta:.4f}",
        ])
        csv_file.flush()

        if verbose:
            print(
                f"Ep {ep + 1:>3}/{n_episodes}  "
                f"return={ep_reward:+.4f}  "
                f"disc_return={ep_discounted_return:+.4f}  "
                f"improvement={ep_improvement:+.4f}  "
                f"ep_best={ep_best_score:.4f} log2 ({ep_best_peptide!r})  "
                f"global_best={best_score:.4f} log2  "
                f"τ={temperature:.2f}  β={entropy_beta:.3f}"
            )

    csv_file.close()

    if verbose:
        print(f"\nOptimisation complete.")
        print(f"  Start : {run_start_peptide!r}  {run_start_score:.4f} log2 MIC  ({2**run_start_score:.1f} µM)")
        print(f"  Best  : {best_peptide!r}  {best_score:.4f} log2 MIC  ({2**best_score:.1f} µM)")
        print(f"  Improvement: {run_start_score - best_score:+.4f} log2 MIC  ({2**run_start_score / 2**best_score:.1f}x fold)")
        print(f"  Log saved  : {csv_path}")

    results = {
        "model_type": "A2C",
        "run_name": run_name or run_id,
        "best_peptide": best_peptide,
        "best_log2mic": best_score,
        "best_mic_uM": 2 ** best_score,
        "start_peptide": run_start_peptide,
        "start_log2mic": run_start_score,
        "start_mic_uM": 2 ** run_start_score,
        "start_peptides": start_pool,
        "start_scores_log2mic": all_start_scores,
        "start_selection_mode": selection,
        "improvement_log2mic": run_start_score - best_score,
        "fold_improvement": (2 ** run_start_score) / (2 ** best_score),
        "reward_formula": "log2MIC_t - log2MIC_t+1",
        "discount_gamma": gamma,
        "potential_type": potential_type,
        "episode_rewards": episode_rewards,
        "episode_discounted_returns": episode_discounted_returns,
        "all_best_scores": all_best_scores,
        "all_best_peptides": all_best_peptides,
        "trajectories": trajectories,
        # legacy aliases kept for report compatibility
        "best_score": best_score,
        "start_score": run_start_score,
        "improvement": run_start_score - best_score,
    }

    # ── persist results to JSON ───────────────────────────────────────────────
    results["trajectories"] = [list(t) for t in results["trajectories"]]
    out_path = os.path.join(output_dir, f"{run_id}_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if verbose:
        print(f"  JSON saved : {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-peptide benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


def run_a2c_all_peptides(
    start_peptides: str = "",
    start_peptides_file: str = "",
    start_selection: str = "cycle",
    n_episodes: int = 100,
    max_steps: int = 20,
    max_candidates: int = 40,
    device: str = "cpu",
    output_dir: str = "results",
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_start: float = 0.1,
    entropy_end: float = 0.01,
    temp_start: float = 2.0,
    temp_end: float = 0.2,
    start_from_best: bool = False,
    potential_type: str = "similarity",
    mutation_direction_significance_threshold: float = 1e-3,
    mutation_min_number_of_directions: int = 5,
    mutation_token_threshold: float = 0.1,
    similarity_horizontal_threshold: float = 1e-3,
    similarity_max_positions: int = 5,
    similarity_max_mutations_per_position: int = 6,
    similarity_max_combinations: int = 40_000,
    similarity_sample_combinations: int = 8_000,
    similarity_include_parent_residue: bool = False,
    peptide_name: str = "",
) -> None:
    """Run A2C optimisation for all six benchmark seed peptides.

    Iterates over ``SEED_PEPTIDES``, calling :func:`run_a2c_optimization` for each
    one with a ``run_name`` equal to the peptide's key (e.g. "middle-1").
    Results are saved as separate JSON + CSV files under *output_dir*, one pair
    per seed peptide.

    If *peptide_name* is provided (must be a key of ``SEED_PEPTIDES``), only that
    single peptide is processed — convenient for SLURM array jobs.

    Parameters
    ----------
    All parameters are forwarded verbatim to :func:`run_a2c_optimization`.
    peptide_name : Optional key into ``SEED_PEPTIDES``.  Empty string = run all.
    """
    os.makedirs(output_dir, exist_ok=True)

    targets = (
        {peptide_name: SEED_PEPTIDES[peptide_name]}
        if peptide_name
        else SEED_PEPTIDES
    )

    for name, seq in targets.items():
        print("\n" + "=" * 70)
        print(f"  SEED: {name!r}  →  {seq!r}")
        print("=" * 70)
        run_a2c_optimization(
            start_peptide=seq,
            start_peptides=start_peptides,
            start_peptides_file=start_peptides_file,
            start_selection=start_selection,
            n_episodes=n_episodes,
            max_steps=max_steps,
            max_candidates=max_candidates,
            device=device,
            output_dir=output_dir,
            lr=lr,
            gamma=gamma,
            entropy_start=entropy_start,
            entropy_end=entropy_end,
            temp_start=temp_start,
            temp_end=temp_end,
            start_from_best=start_from_best,
            potential_type=potential_type,
            mutation_direction_significance_threshold=mutation_direction_significance_threshold,
            mutation_min_number_of_directions=mutation_min_number_of_directions,
            mutation_token_threshold=mutation_token_threshold,
            similarity_horizontal_threshold=similarity_horizontal_threshold,
            similarity_max_positions=similarity_max_positions,
            similarity_max_mutations_per_position=similarity_max_mutations_per_position,
            similarity_max_combinations=similarity_max_combinations,
            similarity_sample_combinations=similarity_sample_combinations,
            similarity_include_parent_residue=similarity_include_parent_residue,
            run_name=name,
            verbose=True,
        )
        print(f"  Done: {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


def test_components(peptide: str = "FLPKKVIPLL", device: str = "cpu") -> None:
    """Smoke-test A2C components: build models, run 2 episodes of 3 steps.

    Parameters
    ----------
    peptide : Amino-acid sequence used as the test input.
    device : PyTorch device string.
    """
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        DecoderLogProbPotential,
    )

    print("=" * 60)
    print("Smoke-testing A2C peptide optimiser components")
    print(f"  Peptide : {peptide!r}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # ── 1. Build models ───────────────────────────────────────────────────────
    print("\n[1] Building encoder/decoder and APEX …")
    enc_dec = build_encoder_decoder(device)
    apex = build_apex_predictor(device)

    mutation_enumerator = MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=1e-3,
        min_number_of_directions=5,
        token_threshold=0.1,
    )
    log_prob_potential = DecoderLogProbPotential(encoder_decoder=enc_dec)
    print("  ✓")

    # ── 2. Actor and Critic Networks ──────────────────────────────────────────
    print("\n[2] ActorNetwork + CriticNetwork …")
    actor = ActorNetwork(LATENT_DIM).to(device)
    critic = CriticNetwork(LATENT_DIM).to(device)

    dummy_state = torch.zeros(1, LATENT_DIM, device=device)
    direction = actor(dummy_state)
    assert direction.shape == (1, LATENT_DIM), f"Actor output shape: {direction.shape}"
    assert abs(torch.norm(direction).item() - 1.0) < 1e-5, "Direction not normalised"

    value = critic(dummy_state)
    assert value.shape == (1,), f"Critic output shape: {value.shape}"
    print(f"  Actor direction shape: {direction.shape}, norm: {torch.norm(direction).item():.4f}")
    print(f"  Critic value shape: {value.shape}  ✓")

    # ── 3. A2C Agent ──────────────────────────────────────────────────────────
    print("\n[3] A2CAgent …")
    agent = A2CAgent(latent_dim=LATENT_DIM, lr=3e-4, gamma=0.99, device=device)

    # Encode peptide
    z_tensor = enc_dec.encode_peptides([peptide])
    z_np = z_tensor.detach().cpu().numpy()[0]

    # Generate candidates
    cand_seqs, cand_probs, cand_zs = generate_candidates(
        peptide=peptide,
        z_np=z_np,
        encoder_decoder=enc_dec,
        mutation_enumerator=mutation_enumerator,
        potential_type="decoder_logprob",
        log_prob_potential=log_prob_potential,
        max_candidates=10,
    )
    print(f"  Generated {len(cand_seqs)} candidates  ✓")

    # ── 4. Run 2 mini-episodes of 3 steps each ────────────────────────────────
    print("\n[4] Running 2 episodes × 3 steps …")

    for ep in range(2):
        agent.reset_trajectory()
        current_z = z_np.copy()
        current_peptide = peptide
        current_cand_seqs = cand_seqs
        current_cand_zs = cand_zs

        for step in range(3):
            # Select action
            action_idx = agent.select_action(
                current_z, current_cand_zs, temperature=1.0
            )
            chosen_seq = current_cand_seqs[action_idx]
            chosen_z = current_cand_zs[action_idx]

            # Compute reward (dummy: just use step improvement metric)
            next_score = float(score_peptides(apex, [chosen_seq])[0])
            reward = max(0.0, 0.1 - 0.01 * step)  # dummy reward

            agent.store_transition(state=current_z, action=action_idx, reward=reward)

            # Advance
            current_z = chosen_z
            current_peptide = chosen_seq
            current_cand_seqs, _, current_cand_zs = generate_candidates(
                peptide=current_peptide,
                z_np=current_z,
                encoder_decoder=enc_dec,
                mutation_enumerator=mutation_enumerator,
                potential_type="decoder_logprob",
                log_prob_potential=log_prob_potential,
                max_candidates=10,
            )

        # Update
        actor_loss, critic_loss, entropy = agent.update(entropy_beta=0.1)
        print(
            f"  Episode {ep + 1}: actor_loss={actor_loss:.4f}, "
            f"critic_loss={critic_loss:.4f}, entropy={entropy:.4f}"
        )

    print("\n" + "=" * 60)
    print("All A2C component tests passed ✓")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "run_a2c_optimization": run_a2c_optimization,
            "run_a2c_all_peptides": run_a2c_all_peptides,
            "test_components": test_components,
        }
    )
