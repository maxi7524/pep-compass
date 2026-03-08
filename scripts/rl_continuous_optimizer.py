"""
TD3 (Twin Delayed DDPG) Continuous Peptide Optimizer in the HydrAMP latent space.

Optimizes antimicrobial peptides by navigating directly in the 64-dimensional
HydrAMP latent space using a continuous actor-critic algorithm (TD3).
Unlike the DQN version, this does not use MUTANG++ for candidate enumeration.

The agent outputs a continuous action (delta vector) which is added to the
current latent vector to produce the next state.

Usage:
    python rl_continuous_optimizer.py run_td3_optimization --start_peptide FLPKKVIPLL --n_episodes 50
    python rl_continuous_optimizer.py run_td3_all_peptides --peptide_name middle-1 --device cuda
    python rl_continuous_optimizer.py test_components --peptide FLPKKVIPLL
"""

from __future__ import annotations

import os
import sys

# Enable imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import shared components from the DQN optimizer
from rl_peptide_optimizer import (
    LATENT_DIM,
    ECOLI_INDICES,
    SEED_PEPTIDES,
    build_encoder_decoder,
    build_apex_predictor,
    score_peptides,
)

# ──────────────────────────────────────────────────────────────────────────────
# Actor Network
# ──────────────────────────────────────────────────────────────────────────────


class ActorNetwork(nn.Module):
    """Actor network that outputs continuous actions (delta in latent space).

    Architecture: Linear(64→256)→ReLU→Linear(256→256)→ReLU→Linear(256→64)→Tanh
    Output is scaled by action_scale.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Produce action (delta) for the given state.

        Parameters
        ----------
        state : ``(..., LATENT_DIM)``

        Returns
        -------
        action : ``(..., LATENT_DIM)`` scaled by action_scale
        """
        return self.net(state) * self.action_scale


# ──────────────────────────────────────────────────────────────────────────────
# Critic Network (Twin Q-functions)
# ──────────────────────────────────────────────────────────────────────────────


class CriticNetwork(nn.Module):
    """Twin Critic network that outputs two Q-value estimates.

    Each critic: Linear(64+64→256)→ReLU→Linear(256→256)→ReLU→Linear(256→1)
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        input_dim = latent_dim * 2  # state + action

        # First critic (Q1)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Second critic (Q2)
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q1 and Q2 values for (state, action) pairs.

        Parameters
        ----------
        state  : ``(..., LATENT_DIM)``
        action : ``(..., LATENT_DIM)``

        Returns
        -------
        q1, q2 : ``(..., 1)`` or ``(...,)`` scalar Q-values.
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 value (used for actor update)."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TD3Transition:
    """Single experience tuple stored in the replay buffer.

    Attributes
    ----------
    state : np.ndarray
        Latent vector before the transition, shape ``(LATENT_DIM,)``.
    action : np.ndarray
        Delta vector applied to the state, shape ``(LATENT_DIM,)``.
    reward : float
        Reward received for the transition.
    next_state : np.ndarray
        Latent vector after the transition, shape ``(LATENT_DIM,)``.
    done : bool
        Whether this transition ends the episode.
    """

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class TD3ReplayBuffer:
    """Fixed-capacity circular replay buffer for TD3 training."""

    def __init__(self, capacity: int = 50_000) -> None:
        self._buffer: deque[TD3Transition] = deque(maxlen=capacity)

    def push(self, transition: TD3Transition) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[TD3Transition]:
        """Sample *batch_size* transitions uniformly at random."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ──────────────────────────────────────────────────────────────────────────────
# TD3 Agent
# ──────────────────────────────────────────────────────────────────────────────


class TD3Agent:
    """TD3 (Twin Delayed DDPG) agent for continuous peptide optimisation.

    Parameters
    ----------
    latent_dim         : Dimensionality of the latent space (default 64).
    action_scale       : Maximum magnitude of action delta (default 0.5).
    lr_actor           : Learning rate for the actor network.
    lr_critic          : Learning rate for the critic networks.
    gamma              : Discount factor γ.
    tau                : Soft update coefficient for target networks.
    policy_delay       : Actor update frequency (update every N critic updates).
    exploration_noise  : Std dev of Gaussian exploration noise.
    target_noise       : Std dev of target policy smoothing noise.
    target_noise_clip  : Clipping bound for target noise.
    batch_size         : Mini-batch size for updates.
    buffer_capacity    : Maximum replay buffer size.
    device             : PyTorch device string.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_scale: float = 0.5,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        exploration_noise: float = 0.3,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = latent_dim
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.batch_size = batch_size
        self.device = device

        # Actor networks
        self.actor = ActorNetwork(latent_dim, action_scale).to(device)
        self.actor_target = ActorNetwork(latent_dim, action_scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # Critic networks (twin)
        self.critic = CriticNetwork(latent_dim).to(device)
        self.critic_target = CriticNetwork(latent_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = TD3ReplayBuffer(buffer_capacity)

        # Update counter
        self._update_steps: int = 0

    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Select an action (delta vector) for the given state.

        Parameters
        ----------
        state     : Current latent state, shape ``(LATENT_DIM,)``.
        add_noise : Whether to add exploration noise.

        Returns
        -------
        action : np.ndarray, shape ``(LATENT_DIM,)``
            The delta vector to apply to the current state.
        """
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            # Clip to action bounds
            action = np.clip(action, -self.action_scale, self.action_scale)

        return action

    def push(self, transition: TD3Transition) -> None:
        """Add a transition to the replay buffer."""
        self.replay_buffer.push(transition)

    def update(self) -> dict | None:
        """Perform one TD3 update step.

        Returns
        -------
        dict or None
            Dictionary with 'critic_loss' and optionally 'actor_loss',
            or None if buffer has too few samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # Assemble tensors
        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.stack([t.action for t in batch]), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        # ── Target Q-values ──
        with torch.no_grad():
            # Target policy smoothing
            noise = (
                torch.randn_like(actions) * self.target_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)

            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.action_scale, self.action_scale
            )

            # Compute target Q-values (take minimum of twin critics)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * target_q * (1.0 - dones)

        # ── Update critics ──
        self.critic.train()
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(
            current_q2, target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        result = {"critic_loss": critic_loss.item()}

        # ── Delayed actor update ──
        self._update_steps += 1
        if self._update_steps % self.policy_delay == 0:
            self.actor.train()

            # Actor loss: maximize Q1(s, actor(s))
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            result["actor_loss"] = actor_loss.item()

        return result

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def set_exploration_noise(self, noise: float) -> None:
        """Set the exploration noise standard deviation."""
        self.exploration_noise = noise


# ──────────────────────────────────────────────────────────────────────────────
# Main optimisation loop
# ──────────────────────────────────────────────────────────────────────────────


def run_td3_optimization(
    start_peptide: str = "FLPKKVIPLL",
    n_episodes: int = 100,
    max_steps: int = 20,
    action_scale: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    exploration_noise_start: float = 0.3,
    exploration_noise_end: float = 0.05,
    target_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    output_dir: str = "results",
    run_name: str = "",
) -> None:
    """Run TD3-based peptide optimisation directly in the HydrAMP latent space.

    Starting from *start_peptide*, the agent navigates the latent space by
    outputting continuous delta vectors for *n_episodes* episodes of *max_steps*
    steps each, guided by APEX MIC predictions.

    The objective is to minimise mean log2(MIC) over the three E. coli strains.

    Parameters
    ----------
    start_peptide         : Amino-acid sequence of the seed peptide.
    n_episodes            : Number of training episodes.
    max_steps             : Maximum steps per episode.
    action_scale          : Maximum magnitude of delta vectors.
    device                : PyTorch device ("cpu" or "cuda").
    verbose               : Print per-episode summaries when True.
    lr_actor              : Actor learning rate.
    lr_critic             : Critic learning rate.
    gamma                 : Discount factor.
    tau                   : Soft update coefficient.
    policy_delay          : Actor update delay (every N critic updates).
    exploration_noise_start : Initial exploration noise std dev.
    exploration_noise_end   : Final exploration noise std dev.
    target_noise          : Target policy smoothing noise std dev.
    target_noise_clip     : Target noise clipping bound.
    batch_size            : Replay buffer mini-batch size.
    buffer_capacity       : Maximum replay buffer capacity.
    output_dir            : Directory to save results JSON and CSV.
    run_name              : Optional tag prepended to output file names.
    """
    import csv
    import json
    import time

    # ── build models ──────────────────────────────────────────────────────────
    if verbose:
        print("Loading models …")
    encoder_decoder = build_encoder_decoder(device)
    apex = build_apex_predictor(device)

    agent = TD3Agent(
        latent_dim=LATENT_DIM,
        action_scale=action_scale,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        policy_delay=policy_delay,
        exploration_noise=exploration_noise_start,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        device=device,
    )

    # ── encode start peptide and compute baseline score ───────────────────────
    if verbose:
        print(f"Encoding start peptide: {start_peptide!r}")
    with torch.no_grad():
        start_z_tensor: torch.Tensor = encoder_decoder.encode_peptides([start_peptide])
    start_z: np.ndarray = start_z_tensor.detach().cpu().numpy()[0]  # (64,)
    start_score: float = float(score_peptides(apex, [start_peptide])[0])

    if verbose:
        print(f"Start score (mean log2 MIC over E.coli): {start_score:.4f}")

    # ── score cache (keyed by peptide string) ────────────────────────────────
    _score_cache: dict[str, float] = {start_peptide: start_score}

    def _get_score(peptide: str) -> float:
        if peptide not in _score_cache:
            _score_cache[peptide] = float(score_peptides(apex, [peptide])[0])
        return _score_cache[peptide]

    # ── output dir + run id ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"td3_{run_name}_" if run_name else "td3_"
    run_id = f"{prefix}rl_{int(time.time())}"

    # ── CSV log (per-episode) ─────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{run_id}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode",
        "run_start_peptide", "run_start_log2mic",
        "ep_start_peptide", "ep_start_log2mic",
        "ep_best_peptide", "ep_best_log2mic",
        "ep_reward_log2mic",
        "global_best_peptide", "global_best_log2mic",
        "exploration_noise",
    ])
    csv_file.flush()

    # ── tracking containers ───────────────────────────────────────────────────
    best_peptide: str = start_peptide
    best_score: float = start_score
    episode_rewards: list[float] = []
    all_best_scores: list[float] = []
    all_best_peptides: list[str] = []
    trajectories: list[list[str]] = []

    # ── episode loop ──────────────────────────────────────────────────────────
    for ep in range(n_episodes):
        # ── per-episode noise decay: linearly decay from start → end
        frac = ep / max(n_episodes - 1, 1)
        current_noise = exploration_noise_start + frac * (
            exploration_noise_end - exploration_noise_start
        )
        agent.set_exploration_noise(current_noise)

        # ── episode starting point ────────────────────────────────────────────
        current_z = start_z.copy()
        current_peptide = start_peptide
        current_score = start_score

        ep_reward = 0.0
        ep_start_score = current_score
        ep_start_peptide = current_peptide
        ep_best_score = current_score
        ep_best_peptide = current_peptide
        trajectory: list[str] = [current_peptide]

        for step in range(max_steps):
            # ── select action (delta) ─────────────────────────────────────────
            action = agent.select_action(current_z, add_noise=True)

            # ── compute next latent state ─────────────────────────────────────
            next_z = current_z + action
            next_z_clamped = np.clip(
                next_z,
                current_z - action_scale,
                current_z + action_scale
            )

            # ── decode to peptide ─────────────────────────────────────────────
            with torch.no_grad():
                next_z_tensor = torch.tensor(
                    next_z_clamped[np.newaxis, :],
                    dtype=torch.float32,
                    device=device
                )
                decoded = encoder_decoder.decode_peptides(next_z_tensor)
            next_peptide = decoded[0].strip()

            # ── evaluate the decoded peptide ──────────────────────────────────
            next_score = _get_score(next_peptide)

            # Reward = improvement beyond the running episode-best score
            reward = max(0.0, ep_best_score - next_score)
            ep_reward += reward
            done = step == max_steps - 1

            # ── store transition ──────────────────────────────────────────────
            agent.push(
                TD3Transition(
                    state=current_z.copy(),
                    action=action.copy(),
                    reward=reward,
                    next_state=next_z_clamped.copy(),
                    done=done,
                )
            )

            # ── learn ─────────────────────────────────────────────────────────
            agent.update()

            # ── bookkeeping ───────────────────────────────────────────────────
            if next_score < ep_best_score:
                ep_best_score = next_score
                ep_best_peptide = next_peptide
            if next_score < best_score:
                best_score = next_score
                best_peptide = next_peptide

            trajectory.append(next_peptide)

            # ── advance state ─────────────────────────────────────────────────
            current_z = next_z_clamped
            current_score = next_score
            current_peptide = next_peptide

        # ── episode summary ───────────────────────────────────────────────────
        ep_improvement = ep_start_score - ep_best_score
        episode_rewards.append(ep_reward)
        all_best_scores.append(ep_best_score)
        all_best_peptides.append(ep_best_peptide)
        trajectories.append(trajectory)

        # write CSV row
        csv_writer.writerow([
            ep + 1,
            start_peptide, f"{start_score:.6f}",
            ep_start_peptide, f"{ep_start_score:.6f}",
            ep_best_peptide, f"{ep_best_score:.6f}",
            f"{ep_improvement:.6f}",
            best_peptide, f"{best_score:.6f}",
            f"{current_noise:.4f}",
        ])
        csv_file.flush()

        if verbose:
            print(
                f"Ep {ep + 1:>3}/{n_episodes}  "
                f"reward={ep_improvement:+.4f} log2  "
                f"ep_best={ep_best_score:.4f} log2 ({ep_best_peptide!r})  "
                f"global_best={best_score:.4f} log2  "
                f"noise={current_noise:.3f}"
            )

    csv_file.close()

    if verbose:
        print(f"\nOptimisation complete.")
        print(f"  Start : {start_peptide!r}  {start_score:.4f} log2 MIC  ({2**start_score:.1f} µM)")
        print(f"  Best  : {best_peptide!r}  {best_score:.4f} log2 MIC  ({2**best_score:.1f} µM)")
        print(f"  Improvement: {start_score - best_score:+.4f} log2 MIC  ({2**start_score / 2**best_score:.1f}x fold)")
        print(f"  Log saved  : {csv_path}")

    results = {
        "run_name": f"td3_{run_name}" if run_name else run_id,
        "algorithm": "TD3",
        "best_peptide": best_peptide,
        "best_log2mic": best_score,
        "best_mic_uM": 2 ** best_score,
        "start_peptide": start_peptide,
        "start_log2mic": start_score,
        "start_mic_uM": 2 ** start_score,
        "improvement_log2mic": start_score - best_score,
        "fold_improvement": (2 ** start_score) / (2 ** best_score),
        "episode_rewards": episode_rewards,
        "all_best_scores": all_best_scores,
        "all_best_peptides": all_best_peptides,
        "trajectories": [list(t) for t in trajectories],
        # legacy aliases kept for compatibility
        "best_score": best_score,
        "start_score": start_score,
        "improvement": start_score - best_score,
    }

    # ── persist results to JSON ───────────────────────────────────────────────
    out_path = os.path.join(output_dir, f"{run_id}_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    if verbose:
        print(f"  JSON saved : {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


def test_components(peptide: str = "FLPKKVIPLL", device: str = "cpu") -> None:
    """Smoke-test all TD3 pipeline components end-to-end.

    Verifies that all API calls succeed and return the expected shapes/types.
    Intended to be called before a full training run.

    Parameters
    ----------
    peptide : Amino-acid sequence used as the test input.
    device  : PyTorch device string.
    """
    print("=" * 60)
    print("Smoke-testing TD3 continuous peptide optimiser components")
    print(f"  Peptide : {peptide!r}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # ── 1. Encoder / Decoder ──────────────────────────────────────────────────
    print("\n[1] HydrAMPEncoderDecoder …")
    enc_dec = build_encoder_decoder(device)
    z_tensor = enc_dec.encode_peptides([peptide])  # (1, 64)
    assert z_tensor.shape == (1, LATENT_DIM), f"Expected (1,64), got {z_tensor.shape}"
    decoded = enc_dec.decode_peptides(z_tensor)
    assert len(decoded) == 1, "decode_peptides should return list of length 1"
    print(f"    encode → {z_tensor.shape}  decode → {decoded!r}  ✓")

    # ── 2. APEX predictor ─────────────────────────────────────────────────────
    print("\n[2] PredictorAPEX …")
    apex = build_apex_predictor(device)
    scores = score_peptides(apex, [peptide])
    assert scores.shape == (1,), f"Expected (1,), got {scores.shape}"
    print(f"    score for {peptide!r}: {scores[0]:.4f} (mean log2 MIC over E.coli)  ✓")

    # ── 3. ActorNetwork ───────────────────────────────────────────────────────
    print("\n[3] ActorNetwork …")
    action_scale = 0.5
    actor = ActorNetwork(LATENT_DIM, action_scale).to(device)
    dummy_state = torch.zeros(1, LATENT_DIM, device=device)
    action_out = actor(dummy_state)
    assert action_out.shape == (1, LATENT_DIM), f"Expected (1,64), got {action_out.shape}"
    assert action_out.abs().max().item() <= action_scale + 1e-6, "Action out of bounds"
    print(f"    action shape: {action_out.shape}  max_abs: {action_out.abs().max().item():.4f}  ✓")

    # ── 4. CriticNetwork ──────────────────────────────────────────────────────
    print("\n[4] CriticNetwork (twin) …")
    critic = CriticNetwork(LATENT_DIM).to(device)
    dummy_action = torch.zeros(1, LATENT_DIM, device=device)
    q1, q2 = critic(dummy_state, dummy_action)
    assert q1.shape == (1,), f"Expected (1,), got {q1.shape}"
    assert q2.shape == (1,), f"Expected (1,), got {q2.shape}"
    print(f"    Q1 shape: {q1.shape}  Q2 shape: {q2.shape}  ✓")

    # ── 5. TD3Agent ───────────────────────────────────────────────────────────
    print("\n[5] TD3Agent.select_action …")
    agent = TD3Agent(device=device, action_scale=action_scale)
    z_np = z_tensor.detach().cpu().numpy()[0]
    action = agent.select_action(z_np, add_noise=True)
    assert action.shape == (LATENT_DIM,), f"Expected ({LATENT_DIM},), got {action.shape}"
    print(f"    action shape: {action.shape}  ✓")

    # ── 6. Latent space navigation ────────────────────────────────────────────
    print("\n[6] Latent space navigation …")
    next_z = z_np + action
    with torch.no_grad():
        next_z_tensor = torch.tensor(next_z[np.newaxis, :], dtype=torch.float32, device=device)
        decoded_next = enc_dec.decode_peptides(next_z_tensor)
    print(f"    next_z norm: {np.linalg.norm(next_z):.4f}  decoded: {decoded_next[0]!r}  ✓")

    print("\n" + "=" * 60)
    print("All TD3 component tests passed ✓")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-peptide benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


def run_td3_all_peptides(
    n_episodes: int = 100,
    max_steps: int = 20,
    action_scale: float = 0.5,
    device: str = "cpu",
    output_dir: str = "results",
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    exploration_noise_start: float = 0.3,
    exploration_noise_end: float = 0.05,
    target_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    peptide_name: str = "",
) -> None:
    """Run TD3 optimisation for all six benchmark seed peptides.

    Iterates over ``SEED_PEPTIDES``, calling :func:`run_td3_optimization` for each
    one with a ``run_name`` equal to the peptide's key (e.g. "middle-1").
    Results are saved as separate JSON + CSV files under *output_dir*, one pair
    per seed peptide.

    If *peptide_name* is provided (must be a key of ``SEED_PEPTIDES``), only that
    single peptide is processed — convenient for SLURM array jobs.

    Parameters
    ----------
    All parameters are forwarded verbatim to :func:`run_td3_optimization`.
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
        print(f"  TD3 SEED: {name!r}  →  {seq!r}")
        print("=" * 70)
        run_td3_optimization(
            start_peptide=seq,
            n_episodes=n_episodes,
            max_steps=max_steps,
            action_scale=action_scale,
            device=device,
            output_dir=output_dir,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            exploration_noise_start=exploration_noise_start,
            exploration_noise_end=exploration_noise_end,
            target_noise=target_noise,
            target_noise_clip=target_noise_clip,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            run_name=name,
            verbose=True,
        )
        print(f"  Done: {name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "run_td3_optimization": run_td3_optimization,
            "run_td3_all_peptides": run_td3_all_peptides,
            "test_components": test_components,
        }
    )
