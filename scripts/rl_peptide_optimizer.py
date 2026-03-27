"""
RL Peptide Optimizer using DQN in the HydrAMP latent space.

Optimizes antimicrobial peptides by navigating the HydrAMP latent space
using a Deep Q-Network. Candidates are generated via tangent-space mutation
enumeration (MUTANG++) and scored by the APEX MIC predictor.

Usage:
    python rl_peptide_optimizer.py --start_peptide FLPKKVIPLL --n_episodes 50
    python rl_peptide_optimizer.py test_components --peptide FLPKKVIPLL
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────────

LATENT_DIM: int = 64
AMBIENT_DIM: int = 525  # 25 positions × 21 tokens (flattened)
MAX_PEPTIDE_LEN: int = 25
ALPHABET: list[str] = list(" ACDEFGHIKLMNPQRSTVWY")  # 21 tokens, index 0 = space

# APEX pathogen indices for the three E. coli strains used as the optimisation target
ECOLI_INDICES: list[int] = [1, 2, 3]

# The six benchmark seed peptides shared with the LEBO/LPBeBo optimization scripts
SEED_PEPTIDES: dict[str, str] = {
    "middle-1":      "FLYKWWIRIGRLKL",
    "jurand-4":      "KYCRRFRWLTFRWL",
    "jurand-2":      "KFRNRHRWKFKLIFRN",
    "jurand-7":      "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2":  "RMARNLVRYVQGLKKKKVI",
}

# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_encoder_decoder(device: str = "cpu"):
    """Instantiate and return a HydrAMPEncoderDecoder."""
    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
        HydrAMPEncoderDecoder,
    )

    return HydrAMPEncoderDecoder(
        jacobian_mode="approx",
        device=device,
        default_condition=torch.tensor([1.0, 1.0]),
        temp=1.0,
        jacobian_eps=0.05,
        field_eps=0.05,
    )


def build_apex_predictor(device: str = "cpu"):
    """Instantiate and return a PredictorAPEX."""
    from pep_compass.models.apex.APEX_predictor import PredictorAPEX

    return PredictorAPEX(device=device)


def discounted_sum(rewards: list[float], gamma: float) -> float:
    """Return discounted trajectory sum ``Σ_t gamma^t * rewards[t]``."""
    total = 0.0
    coeff = 1.0
    for reward in rewards:
        total += coeff * reward
        coeff *= gamma
    return float(total)


def _iter_text_chunks(value: object) -> list[str]:
    """Flatten CLI value into string chunks."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="ignore")]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        chunks: list[str] = []
        for item in value:
            chunks.extend(_iter_text_chunks(item))
        return chunks
    return [str(value)]


def _tokenize_peptide_text(text: str) -> list[str]:
    """Split free-form text/csv input into peptide tokens."""
    peptides: list[str] = []
    valid_aas = set(ALPHABET[1:])
    skip_tokens = {"peptide", "sequence", "seq"}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.replace(",", " ").split():
            peptide = token.strip().upper()
            if not peptide:
                continue
            if peptide.lower() in skip_tokens:
                continue
            if not set(peptide).issubset(valid_aas):
                continue
            if peptide:
                peptides.append(peptide)
    return peptides


def resolve_start_peptides(
    start_peptide: str,
    start_peptides: object = "",
    start_peptides_file: object = "",
) -> list[str]:
    """Resolve run starting peptides from CLI values and optional dataset file."""
    resolved: list[str] = []

    for chunk in _iter_text_chunks(start_peptides):
        if chunk:
            resolved.extend(_tokenize_peptide_text(chunk))

    for path_chunk in _iter_text_chunks(start_peptides_file):
        if not path_chunk:
            continue
        file_path = Path(path_chunk)
        file_text = file_path.read_text(encoding="utf-8")
        resolved.extend(_tokenize_peptide_text(file_text))

    if not resolved and start_peptide:
        resolved.append(start_peptide.strip().upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for peptide in resolved:
        if peptide and peptide not in seen:
            deduped.append(peptide)
            seen.add(peptide)

    if not deduped:
        raise ValueError("No valid start peptides were provided.")
    return deduped


def _normalise_mutations(
    mutations: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Return deterministic, deduplicated mutation lists per position."""
    normalised: dict[int, list[int]] = {}
    for pos, aa_indices in mutations.items():
        unique = sorted({int(idx) for idx in aa_indices})
        if unique:
            normalised[int(pos)] = unique
    return normalised


def _prune_mutation_space(
    mutations: dict[int, list[int]],
    max_positions: int,
    max_mutations_per_position: int,
) -> dict[int, list[int]]:
    """Prune mutation space for performance while keeping deterministic ordering."""
    items = [(pos, indices) for pos, indices in mutations.items() if indices]
    if not items:
        return {}

    # Keep positions with the richest mutation options first.
    items.sort(key=lambda item: (-len(item[1]), item[0]))
    if max_positions > 0:
        items = items[:max_positions]

    pruned: dict[int, list[int]] = {}
    for pos, indices in items:
        kept = indices[:max_mutations_per_position] if max_mutations_per_position > 0 else indices
        if kept:
            pruned[pos] = kept

    return dict(sorted(pruned.items()))


def _cartesian_size(mutations: dict[int, list[int]]) -> int:
    """Return ``k = ∏_pos |mutations[pos]|``."""
    if not mutations:
        return 0
    size = 1
    for aa_indices in mutations.values():
        size *= max(1, len(aa_indices))
    return int(size)


def _shrink_mutation_space_to_budget(
    mutations: dict[int, list[int]],
    max_combinations: int,
) -> dict[int, list[int]]:
    """Shrink per-position mutation lists until cartesian size is within budget."""
    if max_combinations <= 0:
        return mutations

    current = {pos: list(indices) for pos, indices in sorted(mutations.items()) if indices}
    if not current:
        return {}

    while _cartesian_size(current) > max_combinations:
        candidates = [pos for pos, indices in current.items() if len(indices) > 1]
        if candidates:
            pos = max(candidates, key=lambda p: (len(current[p]), -p))
            current[pos] = current[pos][:-1]
            continue
        if len(current) <= 1:
            break
        # If all positions already have one option, drop a position to reduce search.
        drop_pos = max(current.keys())
        current.pop(drop_pos)

    return {pos: indices for pos, indices in current.items() if indices}


# ──────────────────────────────────────────────────────────────────────────────
# Jacobian / SVD utilities
# ──────────────────────────────────────────────────────────────────────────────


def compute_jacobian_svd(
    encoder_decoder,
    z_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the SVD of the decoder Jacobian at latent vector *z_tensor*.

    Parameters
    ----------
    encoder_decoder:
        A HydrAMPEncoderDecoder instance.
    z_tensor:
        1-D tensor of shape ``(LATENT_DIM,)`` representing the current point
        in latent space.

    Returns
    -------
    S_np : np.ndarray
        Singular values, shape ``(LATENT_DIM,)`` == ``(64,)``.
    U_np : np.ndarray
        Left singular vectors (ambient tangent directions), shape
        ``(AMBIENT_DIM, LATENT_DIM)`` == ``(525, 64)``.
    """
    S_np, U_np, _V_np = compute_jacobian_full_svd(encoder_decoder, z_tensor)
    return S_np, U_np


def compute_jacobian_full_svd(
    encoder_decoder,
    z_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full SVD of decoder Jacobian returning ``(S, U, V)``."""
    z_2d = z_tensor.unsqueeze(0) if z_tensor.ndim == 1 else z_tensor  # ensure (1, 64)
    J: torch.Tensor = encoder_decoder.decoder_jacobian(z_2d)  # (1, 525, 64) or (525, 64)
    if J.ndim == 3:
        J = J.squeeze(0)  # → (525, 64)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    V = Vh.transpose(0, 1)
    return (
        S.detach().cpu().numpy(),
        U.detach().cpu().numpy(),
        V.detach().cpu().numpy(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_candidates(
    peptide: str,
    z_np: np.ndarray,
    encoder_decoder,
    mutation_enumerator,
    potential_type: str = "similarity",
    similarity_horizontal_threshold: float = 1e-3,
    similarity_max_positions: int = 5,
    similarity_max_mutations_per_position: int = 6,
    similarity_max_combinations: int = 40_000,
    similarity_sample_combinations: int = 8_000,
    similarity_include_parent_residue: bool = False,
    log_prob_potential=None,
    max_candidates: int = 40,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate candidate mutant peptides in the local tangent space.

    Steps:
    1. Compute the decoder Jacobian and its SVD at the current latent point.
    2. Enumerate per-position amino-acid substitutions via
       MutationEnumerationInTangentSpace.
    3. Rank candidates with selected potential via compose_mutant_distribution.
    4. Encode the top-k candidates with the encoder to obtain latent vectors.

    Parameters
    ----------
    peptide:
        Current amino-acid sequence (the "parent" peptide).
    z_np:
        Current latent vector as a numpy array of shape ``(LATENT_DIM,)``.
    encoder_decoder:
        HydrAMPEncoderDecoder instance.
    mutation_enumerator:
        MutationEnumerationInTangentSpace instance.
    potential_type:
        ``"similarity"`` (default) or ``"decoder_logprob"``.
    similarity_horizontal_threshold:
        Horizontal threshold used in ``SubRiemannianTangentSpace`` projection.
    similarity_max_positions:
        Maximum mutable positions retained for similarity scoring.
    similarity_max_mutations_per_position:
        Maximum amino-acid options retained per mutable position.
    similarity_max_combinations:
        Upper bound on cartesian combinations after deterministic pruning.
    similarity_sample_combinations:
        Maximum number of cartesian combinations sampled for similarity potentials.
    similarity_include_parent_residue:
        Whether to include parent amino acid per mutable position for similarity potential.
    log_prob_potential:
        Optional ``DecoderLogProbPotential`` used when ``potential_type="decoder_logprob"``.
    max_candidates:
        Maximum number of candidate sequences to return.

    Returns
    -------
    seqs : list[str]
        Candidate peptide sequences (length ≤ max_candidates).
    softmax_probs : np.ndarray
        Normalised probabilities derived from log-potentials, shape ``(n,)``.
    candidate_zs : np.ndarray
        Encoded latent vectors for each candidate, shape ``(n, LATENT_DIM)``.
    """
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        DecoderLogProbPotential,
        ProjectedDirectionPairwiseSimilarityPotential,
        compose_mutant_distribution,
    )
    from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace

    z_tensor = torch.tensor(z_np, dtype=torch.float32, device=encoder_decoder.device)

    # ── 1. Jacobian SVD ──────────────────────────────────────────────────────
    S_np, U_np, V_np = compute_jacobian_full_svd(encoder_decoder, z_tensor)
    # Keep similarity projection on CPU for stability with current
    # SubRiemannianTangentSpace implementation.
    proj_device = "cpu"
    S_t = torch.tensor(S_np, dtype=torch.float32, device=proj_device)
    U_t = torch.tensor(U_np, dtype=torch.float32, device=proj_device)
    V_t = torch.tensor(V_np, dtype=torch.float32, device=proj_device)

    # ── 2. Enumerate mutations in tangent space ──────────────────────────────
    raw_mutations: dict[int, list[int]] = mutation_enumerator.get_mutations_from_s_u(
        S_np, U_np
    )
    mutations = _normalise_mutations(raw_mutations)

    # Edge case: no mutations discovered → fall back to current peptide only
    if not mutations:
        candidate_zs = z_np[np.newaxis, :]  # (1, 64)
        return [peptide], np.array([1.0]), candidate_zs

    # ── 3. Rank with selected potential ─────────────────────────────────────
    potential_key = potential_type.strip().lower()
    if potential_key == "similarity":
        tangent_space = SubRiemannianTangentSpace(
            U_t,
            S_t,
            V_t,
            horizontal_threshold=similarity_horizontal_threshold,
            device=proj_device,
        )
        potential = ProjectedDirectionPairwiseSimilarityPotential(
            tangent_space=tangent_space,
            alphabet=ALPHABET,
        )

        pruned_mutations = _prune_mutation_space(
            mutations,
            max_positions=similarity_max_positions,
            max_mutations_per_position=similarity_max_mutations_per_position,
        )
        effective_budget = similarity_max_combinations
        if similarity_sample_combinations > 0:
            effective_budget = min(effective_budget, similarity_sample_combinations)
        budgeted_mutations = _shrink_mutation_space_to_budget(
            pruned_mutations,
            max_combinations=effective_budget,
        )
        if budgeted_mutations:
            mutations = budgeted_mutations
    elif potential_key in {"decoder_logprob", "logprob", "decoder"}:
        potential = log_prob_potential or DecoderLogProbPotential(
            encoder_decoder=encoder_decoder
        )
    else:
        raise ValueError(f"Unsupported potential_type={potential_type!r}")

    mutant_dist = compose_mutant_distribution(
        parent_peptide=peptide,
        mutations=mutations,
        potential=potential,
        alphabet=ALPHABET,
        max_len=MAX_PEPTIDE_LEN,
        include_parent_residue=(
            similarity_include_parent_residue if potential_key == "similarity" else False
        ),
        top_k=max_candidates,
    )

    # Edge case: compose_mutant_distribution returned empty distribution
    if not mutant_dist.sequences:
        candidate_zs = z_np[np.newaxis, :]
        return [peptide], np.array([1.0]), candidate_zs

    seqs: list[str] = mutant_dist.sequences
    log_pots: np.ndarray = mutant_dist.log_potentials  # sorted descending

    # ── 4. Remove parent sequence — the decoder assigns it the highest
    #       log-prob so it dominates the softmax unless explicitly excluded.
    parent_stripped = peptide.strip()
    mask_not_parent = np.array([s.strip() != parent_stripped for s in seqs])
    if mask_not_parent.sum() == 0:
        # All candidates decoded back to parent — return parent as fallback
        candidate_zs = z_np[np.newaxis, :]
        return [peptide], np.array([1.0]), candidate_zs
    seqs = [s for s, m in zip(seqs, mask_not_parent) if m]
    log_pots = log_pots[mask_not_parent]

    # ── 5. Softmax over log-potentials ──────────────────────────────────────
    shifted = log_pots - log_pots.max()
    exp_pots = np.exp(shifted)
    softmax_probs: np.ndarray = exp_pots / exp_pots.sum()

    # ── 6. Filter to candidates above the uniform threshold 1/k ─────────────
    #  Keep only mutations whose softmax probability exceeds the uniform baseline.
    #  This sharpens the action space to above-average candidates only.
    k = _cartesian_size(mutations)
    if k <= 0:
        k = len(seqs)
    keep = softmax_probs > (1.0 / k)
    if keep.sum() == 0:
        keep[np.argmax(softmax_probs)] = True  # always keep the best one
    seqs = [s for s, m in zip(seqs, keep) if m]
    softmax_probs = softmax_probs[keep]
    softmax_probs = softmax_probs / softmax_probs.sum()  # renormalise

    # ── 7. Encode surviving candidates ──────────────────────────────────────
    with torch.no_grad():
        z_tensor_batch: torch.Tensor = encoder_decoder.encode_peptides(seqs)
    candidate_zs: np.ndarray = z_tensor_batch.detach().cpu().numpy()  # (n, 64)

    return seqs, softmax_probs, candidate_zs


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────


def score_peptides(
    apex_predictor,
    peptides: list[str],
    ecoli_indices: list[int] = ECOLI_INDICES,
) -> np.ndarray:
    """Score a list of peptides as mean log2(MIC) over selected E. coli strains.

    Lower score = more potent (lower MIC).

    Parameters
    ----------
    apex_predictor:
        PredictorAPEX instance.
    peptides:
        List of amino-acid sequences.
    ecoli_indices:
        Column indices in APEX output corresponding to E. coli strains.

    Returns
    -------
    scores : np.ndarray
        Shape ``(n,)``. Mean log2(MIC) over the selected pathogens.
    """
    mic: np.ndarray = apex_predictor.predict(peptides)  # (n, n_pathogens), μM
    mic_ecoli = mic[:, ecoli_indices]  # (n, len(ecoli_indices))
    # Guard against non-positive MIC values before log2
    mic_ecoli = np.clip(mic_ecoli, a_min=1e-6, a_max=None)
    log2_mic = np.log2(mic_ecoli)  # (n, len(ecoli_indices))
    return log2_mic.mean(axis=1)  # (n,)


# ──────────────────────────────────────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Deep Q-Network that estimates Q(state, action) for peptide optimisation.

    Input  : concatenation of state latent vector and action latent vector
             → 128-dimensional vector.
    Output : scalar Q-value estimate.

    Architecture: 128 → 256 → 128 → 64 → 1
    Each hidden layer uses LayerNorm followed by ReLU activation.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        input_dim = latent_dim * 2  # state ∥ action
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-value(s) for (state, action) pair(s).

        Parameters
        ----------
        state  : ``(..., LATENT_DIM)``
        action : ``(..., LATENT_DIM)``

        Returns
        -------
        q : ``(..., 1)`` or ``(...,)`` scalar Q-values.
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Transition:
    """Single experience tuple stored in the replay buffer.

    Attributes
    ----------
    state : np.ndarray
        Latent vector before the transition, shape ``(LATENT_DIM,)``.
    action : np.ndarray
        Latent vector of the chosen candidate (the "action"), shape
        ``(LATENT_DIM,)``.
    reward : float
        Reward received for the transition.
    next_state : np.ndarray
        Latent vector after the transition (same as *action*), shape
        ``(LATENT_DIM,)``.
    done : bool
        Whether this transition ends the episode.
    next_candidates : np.ndarray
        Encoded latent vectors of all candidates available at *next_state*,
        shape ``(n_candidates, LATENT_DIM)``.  Empty array when ``done=True``.
    """

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    next_candidates: np.ndarray = field(default_factory=lambda: np.empty((0, LATENT_DIM)))


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for DQN training."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample *batch_size* transitions uniformly at random."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────


class DQNAgent:
    """Double-network DQN agent for peptide optimisation in latent space.

    Uses epsilon-greedy exploration where the random fallback samples from the
    softmax distribution over log-potentials (guided exploration) rather than
    pure uniform random.

    Parameters
    ----------
    latent_dim      : Dimensionality of the latent space (default 64).
    lr              : Learning rate for the Adam optimiser.
    gamma           : Discount factor γ.
    epsilon_start   : Initial exploration rate.
    epsilon_end     : Minimum exploration rate.
    epsilon_decay   : Multiplicative decay applied per update step.
    batch_size      : Mini-batch size for TD updates.
    buffer_capacity : Maximum replay buffer size.
    target_update_freq : Number of update steps between hard target-network syncs.
    device          : PyTorch device string.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_capacity: int = 10_000,
        target_update_freq: int = 50,
        device: str = "cpu",
    ) -> None:
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        self.q_net = QNetwork(latent_dim).to(device)
        self.target_net = QNetwork(latent_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self._update_steps: int = 0

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(
        self,
        current_z: np.ndarray,
        candidate_zs: np.ndarray,
        softmax_probs: np.ndarray,
    ) -> int:
        """Select an action (candidate index) via epsilon-greedy policy.

        * With probability ε  → guided exploration: sample an index
          proportional to *softmax_probs*.
        * With probability 1-ε → exploitation: choose the candidate with the
          highest Q-value under the online network.

        Parameters
        ----------
        current_z     : Current latent state, shape ``(LATENT_DIM,)``.
        candidate_zs  : Encoded candidates, shape ``(n, LATENT_DIM)``.
        softmax_probs : Normalised softmax of log-potentials, shape ``(n,)``.

        Returns
        -------
        int
            Index of the chosen candidate.
        """
        n = len(candidate_zs)
        if n == 0:
            return 0

        if random.random() < self.epsilon:
            # Guided random: sample from mutant distribution probabilities
            probs = softmax_probs if len(softmax_probs) == n else np.ones(n) / n
            return int(np.random.choice(n, p=probs))

        # Exploitation: argmax Q(s, a) over all candidates
        self.q_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(
                np.tile(current_z, (n, 1)), dtype=torch.float32, device=self.device
            )  # (n, 64)
            actions_t = torch.tensor(
                candidate_zs, dtype=torch.float32, device=self.device
            )  # (n, 64)
            q_values = self.q_net(state_t, actions_t)  # (n,)
        return int(q_values.argmax().item())

    # ── buffer interaction ────────────────────────────────────────────────────

    def push(self, transition: Transition) -> None:
        """Add a transition to the replay buffer."""
        self.replay_buffer.push(transition)

    # ── learning update ───────────────────────────────────────────────────────

    def update(self) -> float | None:
        """Perform one TD-learning update step.

        Returns
        -------
        float or None
            The training loss, or ``None`` if the buffer has too few samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.q_net.train()
        batch: list[Transition] = self.replay_buffer.sample(self.batch_size)

        # ── assemble tensors ─────────────────────────────────────────────────
        states = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device
        )  # (B, 64)
        actions = torch.tensor(
            np.stack([t.action for t in batch]), dtype=torch.float32, device=self.device
        )  # (B, 64)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )  # (B,)
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )  # (B,)

        # ── current Q-values ─────────────────────────────────────────────────
        q_values = self.q_net(states, actions)  # (B,)

        # ── target Q-values: r + γ * max_a' Q_target(s', a') ────────────────
        with torch.no_grad():
            next_q_max = torch.zeros(len(batch), device=self.device)
            for i, transition in enumerate(batch):
                if transition.done or len(transition.next_candidates) == 0:
                    next_q_max[i] = 0.0
                    continue
                ns_np = transition.next_candidates  # (k, 64)
                k = len(ns_np)
                next_state_t = torch.tensor(
                    np.tile(transition.next_state, (k, 1)),
                    dtype=torch.float32,
                    device=self.device,
                )
                next_actions_t = torch.tensor(
                    ns_np, dtype=torch.float32, device=self.device
                )
                q_next = self.target_net(next_state_t, next_actions_t)  # (k,)
                next_q_max[i] = q_next.max()

        targets = rewards + self.gamma * next_q_max * (1.0 - dones)  # (B,)

        # ── TD loss with gradient clipping ───────────────────────────────────
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── hard target-network sync ─────────────────────────────────────────
        self._update_steps += 1
        if self._update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # ── epsilon decay ────────────────────────────────────────────────────
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Main optimisation loop
# ──────────────────────────────────────────────────────────────────────────────


def run_rl_optimization(
    start_peptide: str = "FLPKKVIPLL",
    start_peptides: str = "",
    start_peptides_file: str = "",
    start_selection: str = "cycle",
    n_episodes: int = 50,
    max_steps: int = 20,
    max_candidates: int = 40,
    device: str = "cpu",
    verbose: bool = True,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 32,
    buffer_capacity: int = 10_000,
    target_update_freq: int = 50,
    output_dir: str = "results",
    start_from_best: bool = False,
    per_episode_epsilon: bool = True,
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
    """Run DQN-based peptide optimisation in the HydrAMP latent space.

    Supports a dataset of starting peptides. If ``start_peptides`` and
    ``start_peptides_file`` are empty, falls back to ``start_peptide``.
    Per-step reward is the consecutive score delta:
    ``reward_t = log2MIC_t - log2MIC_{t+1}``, and discounted return uses
    ``gamma``.

    Parameters
    ----------
    start_peptide     : Fallback single amino-acid seed sequence.
    start_peptides    : Inline peptide dataset (comma/space/newline separated).
    start_peptides_file: Path to peptide dataset file.
    start_selection   : Episode start selector: ``cycle`` or ``random``.
    n_episodes        : Number of training episodes.
    max_steps         : Maximum transitions per episode.
    max_candidates    : Maximum number of mutant candidates per step.
    device            : PyTorch device ("cpu" or "cuda").
    verbose           : Print per-episode summaries when True.
    lr                : Learning rate.
    gamma             : Discount factor.
    epsilon_start     : Initial ε for ε-greedy exploration.
    epsilon_end       : Minimum ε.
    epsilon_decay     : Per-update multiplicative ε decay factor.
    batch_size        : Replay-buffer mini-batch size.
    buffer_capacity   : Maximum replay-buffer capacity.
    target_update_freq: Hard target-network update frequency (update steps).
    output_dir        : Directory to save results JSON and CSV.
    start_from_best   : If True, each episode starts from the best peptide found
                        so far instead of sampled dataset starts.
    per_episode_epsilon: If True, reset ε at the start of every episode using a
                        linearly decaying schedule (epsilon_start → epsilon_end
                        over n_episodes).
    potential_type    : Candidate potential type: ``similarity`` or ``decoder_logprob``.
    mutation_direction_significance_threshold: MUTANG++ direction threshold.
    mutation_min_number_of_directions: MUTANG++ minimum kept directions.
    mutation_token_threshold: MUTANG++ token threshold.
    similarity_horizontal_threshold: Horizontal threshold for similarity projection.
    similarity_max_positions: Max positions for similarity potential scoring.
    similarity_max_mutations_per_position: Max residues per position for similarity scoring.
    similarity_max_combinations: Combination budget before deterministic shrinking.
    similarity_sample_combinations: Sampling cap for similarity combinations.
    similarity_include_parent_residue: Include parent AA as option in similarity scoring.
    run_name          : Optional tag prepended to the output file names.
    """
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        DecoderLogProbPotential,
    )

    import csv
    import json
    import os
    import time

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

    agent = DQNAgent(
        latent_dim=LATENT_DIM,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
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

    # ── latent + score caches keyed by peptide string ─────────────────────────
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
    run_id = f"{prefix}rl_{int(time.time())}"

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
        "epsilon",
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
        # ── per-episode epsilon: linearly decay from epsilon_start → epsilon_end
        if per_episode_epsilon:
            frac = ep / max(n_episodes - 1, 1)
            agent.epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        # ── episode starting point ────────────────────────────────────────────
        if start_from_best and ep > 0:
            current_peptide = best_peptide
            current_z = _get_latent(current_peptide).copy()
            current_score = _get_score(current_peptide)
        else:
            if selection == "random":
                current_peptide = random.choice(start_pool)
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
            action_idx: int = agent.select_action(current_z, cand_zs, cand_probs)
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
            done: bool = step == max_steps - 1

            # ── generate next-step candidates (cached) ────────────────────────
            if not done:
                next_cand_seqs, next_cand_probs, next_cand_zs = _get_candidates(
                    chosen_seq, chosen_z
                )
            else:
                next_cand_seqs = []
                next_cand_probs = np.empty(0)
                next_cand_zs = np.empty((0, LATENT_DIM))

            # ── store transition ──────────────────────────────────────────────
            agent.push(
                Transition(
                    state=current_z,
                    action=chosen_z,
                    reward=reward,
                    next_state=chosen_z,
                    done=done,
                    next_candidates=next_cand_zs,
                )
            )

            # ── learn ─────────────────────────────────────────────────────────
            agent.update()

            # ── bookkeeping ───────────────────────────────────────────────────
            if next_score < ep_best_score:
                ep_best_score = next_score
                ep_best_peptide = chosen_seq
            if next_score < best_score:
                best_score = next_score
                best_peptide = chosen_seq

            trajectory.append(chosen_seq)

            # ── advance state ─────────────────────────────────────────────────
            current_z = chosen_z
            current_score = next_score
            current_peptide = chosen_seq
            _latent_cache[chosen_seq] = chosen_z.copy()

            # Reuse next candidates as current candidates for the next step
            if not done:
                cand_seqs = next_cand_seqs
                cand_probs = next_cand_probs
                cand_zs = next_cand_zs

                if len(cand_seqs) == 0:
                    cand_seqs, cand_probs, cand_zs = _get_candidates(
                        current_peptide, current_z
                    )

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
            f"{agent.epsilon:.4f}",
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
                f"ε={agent.epsilon:.3f}"
            )

    csv_file.close()

    if verbose:
        print(f"\nOptimisation complete.")
        print(f"  Start : {run_start_peptide!r}  {run_start_score:.4f} log2 MIC  ({2**run_start_score:.1f} µM)")
        print(f"  Best  : {best_peptide!r}  {best_score:.4f} log2 MIC  ({2**best_score:.1f} µM)")
        print(f"  Improvement: {run_start_score - best_score:+.4f} log2 MIC  ({2**run_start_score / 2**best_score:.1f}x fold)")
        print(f"  Log saved  : {csv_path}")

    results = {
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
        # legacy aliases kept for notebook compatibility
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
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────


def test_components(peptide: str = "FLPKKVIPLL", device: str = "cpu") -> None:
    """Smoke-test all pipeline components end-to-end.

    Verifies that all API calls succeed and return the expected shapes /
    types.  Intended to be called before a full training run.

    Parameters
    ----------
    peptide : Amino-acid sequence used as the test input.
    device  : PyTorch device string.
    """
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        DecoderLogProbPotential,
        compose_mutant_distribution,
    )

    print("=" * 60)
    print("Smoke-testing RL peptide optimiser components")
    print(f"  Peptide : {peptide!r}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # ── 1. Encoder / Decoder ──────────────────────────────────────────────────
    print("\n[1] HydrAMPEncoderDecoder …")
    enc_dec = build_encoder_decoder(device)
    assert enc_dec.latent_dim == LATENT_DIM, "latent_dim mismatch"
    assert enc_dec.ambient_dim == AMBIENT_DIM, "ambient_dim mismatch"

    z_tensor = enc_dec.encode_peptides([peptide])  # (1, 64)
    assert z_tensor.shape == (1, LATENT_DIM), f"Expected (1,64), got {z_tensor.shape}"

    decoded = enc_dec.decode_peptides(z_tensor)
    assert len(decoded) == 1, "decode_peptides should return list of length 1"
    print(f"    encode → {z_tensor.shape}  decode → {decoded!r}  ✓")

    # ── 2. Jacobian + SVD ────────────────────────────────────────────────────
    print("\n[2] decoder_jacobian + SVD …")
    z_1d = z_tensor[0]  # (64,)
    S_np, U_np = compute_jacobian_svd(enc_dec, z_1d)
    assert S_np.shape == (LATENT_DIM,), f"S shape expected ({LATENT_DIM},), got {S_np.shape}"
    assert U_np.shape == (AMBIENT_DIM, LATENT_DIM), (
        f"U shape expected ({AMBIENT_DIM},{LATENT_DIM}), got {U_np.shape}"
    )
    print(f"    S: {S_np.shape}  U: {U_np.shape}  ✓")

    # ── 3. Mutation enumerator ────────────────────────────────────────────────
    print("\n[3] MutationEnumerationInTangentSpace …")
    mut_enum = MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=1e-3,
        min_number_of_directions=5,
        token_threshold=0.1,
    )
    mutations = mut_enum.get_mutations_from_s_u(S_np, U_np)
    print(f"    mutations dict: {len(mutations)} positions  ✓")

    # ── 4. DecoderLogProbPotential ────────────────────────────────────────────
    print("\n[4] DecoderLogProbPotential …")
    pot = DecoderLogProbPotential(encoder_decoder=enc_dec)
    if mutations:
        potentials = pot.compute(peptide, mutations)
        assert isinstance(potentials, dict), "potential.compute should return dict"
        print(f"    potentials: {len(potentials)} positions  ✓")
    else:
        print("    (no mutations – skipping potential.compute)  ✓")

    # ── 5. compose_mutant_distribution ───────────────────────────────────────
    print("\n[5] compose_mutant_distribution …")
    if mutations:
        mutant_dist = compose_mutant_distribution(
            parent_peptide=peptide,
            mutations=mutations,
            potential=pot,
            alphabet=ALPHABET,
            max_len=MAX_PEPTIDE_LEN,
            include_parent_residue=False,
            top_k=10,
        )
        print(
            f"    sequences: {len(mutant_dist.sequences)}  "
            f"log_potentials shape: {mutant_dist.log_potentials.shape}  ✓"
        )
    else:
        print("    (no mutations – skipping)  ✓")

    # ── 6. generate_candidates ────────────────────────────────────────────────
    print("\n[6] generate_candidates …")
    z_np = z_1d.detach().cpu().numpy()
    seqs, probs, cand_zs = generate_candidates(
        peptide=peptide,
        z_np=z_np,
        encoder_decoder=enc_dec,
        mutation_enumerator=mut_enum,
        potential_type="decoder_logprob",
        log_prob_potential=pot,
        max_candidates=10,
    )
    assert cand_zs.ndim == 2 and cand_zs.shape[1] == LATENT_DIM, (
        f"candidate_zs shape error: {cand_zs.shape}"
    )
    assert len(seqs) == len(probs) == len(cand_zs), "Mismatched candidate lengths"
    assert abs(probs.sum() - 1.0) < 1e-5, "Probabilities must sum to 1"
    print(f"    {len(seqs)} candidates, zs: {cand_zs.shape}  ✓")

    # ── 7. APEX predictor ─────────────────────────────────────────────────────
    print("\n[7] PredictorAPEX …")
    apex = build_apex_predictor(device)
    scores = score_peptides(apex, [peptide])
    assert scores.shape == (1,), f"Expected (1,), got {scores.shape}"
    print(f"    score for {peptide!r}: {scores[0]:.4f} (mean log2 MIC over E.coli)  ✓")

    # ── 8. QNetwork ───────────────────────────────────────────────────────────
    print("\n[8] QNetwork …")
    qnet = QNetwork(LATENT_DIM)
    dummy_s = torch.zeros(1, LATENT_DIM)
    dummy_a = torch.zeros(1, LATENT_DIM)
    q_out = qnet(dummy_s, dummy_a)
    assert q_out.shape == (1,), f"Expected (1,), got {q_out.shape}"
    print(f"    Q-value shape: {q_out.shape}  ✓")

    # ── 9. DQNAgent (select_action) ───────────────────────────────────────────
    print("\n[9] DQNAgent.select_action …")
    agent = DQNAgent(device=device)
    idx = agent.select_action(z_np, cand_zs, probs)
    assert 0 <= idx < len(seqs), f"action index {idx} out of range [0, {len(seqs)})"
    print(f"    selected action idx: {idx}  ✓")

    print("\n" + "=" * 60)
    print("All component tests passed ✓")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-peptide benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


def run_all_peptides(
    start_peptides: str = "",
    start_peptides_file: str = "",
    start_selection: str = "cycle",
    n_episodes: int = 100,
    max_steps: int = 20,
    max_candidates: int = 40,
    device: str = "cpu",
    output_dir: str = "results",
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 32,
    buffer_capacity: int = 10_000,
    target_update_freq: int = 50,
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
    """Run DQN optimisation for all six benchmark seed peptides.

    Iterates over ``SEED_PEPTIDES``, calling :func:`run_rl_optimization` for each
    one with a ``run_name`` equal to the peptide's key (e.g. "middle-1").
    Results are saved as separate JSON + CSV files under *output_dir*, one pair
    per seed peptide.

    If *peptide_name* is provided (must be a key of ``SEED_PEPTIDES``), only that
    single peptide is processed — convenient for SLURM array jobs.

    Parameters
    ----------
    All parameters are forwarded verbatim to :func:`run_rl_optimization`.
    peptide_name : Optional key into ``SEED_PEPTIDES``.  Empty string = run all.
    """
    import os

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
        run_rl_optimization(
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
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
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
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "run_rl_optimization": run_rl_optimization,
            "run_all_peptides": run_all_peptides,
            "test_components": test_components,
        }
    )
