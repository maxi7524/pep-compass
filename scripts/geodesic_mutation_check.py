"""
Geodesic Mutation Verification for MUTANG++
============================================

For each seed peptide and each MUTANG++ mutation candidate above the softmax
threshold, this script checks whether following a geodesic (or Euclidean line)
in the mutation direction actually leads to the mutated peptide.

Pipeline (following potentials_analysis.ipynb):
  1. Encode peptide → z  (HydrAMPEncoderDecoder)
  2. Compute Jacobian of the *softmax* decoder output at z
  3. SVD → U, S, V
  4. get_mutations_from_s_u_standard (filter to positions within peptide length)
  5. Score with DecoderLogProbPotential, remove identity, softmax, threshold 1/n_mut
  6. For each surviving mutation compute a latent direction via J⁺ @ e_mutation
     (using SubRiemannianTangentSpace horizontal projection) and also the direct
     encoding direction z_mutant − z_parent
  7. Follow both Euclidean straight lines and Riemannian geodesics (frozen Γ at
     z_parent) and decode the endpoint, comparing to the expected mutant

Methods tested per mutation:
  1. Euclidean line with projected direction    (z + v_proj)
  2. Euclidean line with encoded direction       (z + v_enc)
  3. Geodesic with projected direction           exp_z(v_proj)
  4. Geodesic with encoded direction             exp_z(v_enc)

Usage:
    python scripts/geodesic_mutation_check.py [--device cpu|cuda]
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from rl_peptide_optimizer import (
    LATENT_DIM,
    AMBIENT_DIM,
    SEED_PEPTIDES,
    ALPHABET,
    MAX_PEPTIDE_LEN,
    build_encoder_decoder,
)

from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.local_enumeration.mutation.utils import get_mutations_from_s_u_standard
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    DecoderLogProbPotential,
)
from pep_compass.geometry.utils import (
    approx_dg_from_jac,
    christoffel_from_jac_and_dg,
    exponential_map,
    integrate_geodesic_rk4,
)


# ═══════════════════════════════════════════════════════════════════════════
#  SVD + mutations  (following potentials_analysis.ipynb)
# ═══════════════════════════════════════════════════════════════════════════

JACOBIAN_EPS = 1e-6          # ε for finite-difference Jacobian (matches notebook)
DIRECTION_SIG_THRESH = 1e-6  # SVD singular-value threshold
TOKEN_THRESH = 1e-4          # per-token weight threshold
HORIZONTAL_THRESH = 1e-4     # sub-Riemannian horizontal threshold
MIN_DIRECTIONS = 5


def compute_svd_and_mutations(
    encoder_decoder,
    peptide: str,
    z: torch.Tensor,
) -> tuple:
    """Compute Jacobian SVD, mutations, and tangent space — notebook style.

    Returns (jac, U, S, V, mutations, tangent_space).
    """
    z_2d = z.unsqueeze(0) if z.ndim == 1 else z  # (1, 64)
    pep_len = len(peptide)

    # Jacobian of the *softmax* decoder (matches potentials_analysis.ipynb)
    jac = decoder_jacobian(
        lambda x: encoder_decoder.decoder_forward(x, softmax=True, flatten=True),
        z_2d,
        jacobian_fn_mode="approx",
        jacobian_fn_kwargs={"jacobian_eps": JACOBIAN_EPS},
    )  # (1, 525, 64)

    U, S, V = torch.linalg.svd(jac, full_matrices=False)
    # U: (1, 525, 64), S: (1, 64), V: (1, 64, 64)

    # Enumerate mutations (only within peptide length)
    mutations = get_mutations_from_s_u_standard(
        s=S[0].detach().cpu().numpy(),
        u=U[0].detach().cpu().numpy(),
        max_len=MAX_PEPTIDE_LEN,
        alphabet_size=len(ALPHABET),
        direction_significance_threshold=DIRECTION_SIG_THRESH,
        min_number_of_directions=MIN_DIRECTIONS,
        token_threshold=TOKEN_THRESH,
    )
    mutations = {pos: muts for pos, muts in mutations.items() if pos < pep_len}

    # Sub-Riemannian tangent space (for horizontal projection)
    tangent_space = SubRiemannianTangentSpace(
        U=U[0], S=S[0], V=V[0],
        horizontal_threshold=HORIZONTAL_THRESH,
        device=str(z.device),
    )

    return jac, U, S, V, mutations, tangent_space


# ═══════════════════════════════════════════════════════════════════════════
#  Gamma (Christoffel symbols) computation
# ═══════════════════════════════════════════════════════════════════════════

GAMMA_EPS = 0.05  # perturbation ε for Christoffel finite differences
METRIC_REG = 1.0  # regularization λ added to metric diagonal before inversion
# The raw pullback metric g = J^T J has near-zero and even slightly negative
# eigenvalues (from finite-difference noise), making it effectively singular.
# λ = 1.0 brings Gamma_max to ~O(10^1), yielding stable RK4 integration.


def _christoffel_regularised(jac, dg, reg=None):
    """Christoffel symbols with regularised metric: g_reg = J^T J + λ I.

    The raw pullback metric is nearly singular (only a few large singular
    values), so torch.linalg.inv(g) explodes.  Adding a small diagonal
    stabilises the inverse and keeps Γ bounded.
    """
    if reg is None:
        reg = METRIC_REG
    from einops import rearrange, einsum

    g = einsum(jac, jac, "b m i, b m j -> b i j")
    g_reg = g + reg * torch.eye(g.shape[-1], device=g.device).unsqueeze(0)
    g_inv = torch.linalg.inv(g_reg)

    term1 = rearrange(dg, "b l i j -> b l i j")   # ∂_j g_{li}
    term2 = rearrange(dg, "b l j i -> b l i j")   # ∂_i g_{lj}
    term3 = rearrange(dg, "b i j l -> b l i j")   # ∂_l g_{ij}
    T = term1 + term2 - term3

    Gamma = 0.5 * einsum(g_inv, T, "b k l, b l i j -> b k i j")
    return Gamma


def build_frozen_gamma(
    encoder_decoder,
    z: torch.Tensor,
    eps: float = GAMMA_EPS,
) -> tuple:
    """Compute Christoffel symbols at *z* (frozen for geodesic integration).

    Uses the **raw** (non-softmax) decoder Jacobian with ε = 0.05 for
    numerical stability, and regularises the pullback metric before
    inversion to tame the ill-conditioned metric.

    Returns (gamma_fn, Gamma).
    """
    z_2d = z.unsqueeze(0) if z.ndim == 1 else z  # (1, 64)
    device = z.device

    # Base Jacobian at z — raw logits, eps=0.05 (default encoder_decoder setting)
    with torch.no_grad():
        J = encoder_decoder.decoder_jacobian(z_2d)  # (525, 64) or (1, 525, 64)
    if J.ndim == 2:
        J = J.unsqueeze(0)  # → (1, 525, 64)

    # Perturbed Jacobians: J(z + eps * e_k) for k = 0 .. 63
    J_pert_list = []
    for k in range(LATENT_DIM):
        e_k = torch.zeros(1, LATENT_DIM, device=device)
        e_k[0, k] = eps
        with torch.no_grad():
            J_k = encoder_decoder.decoder_jacobian(z_2d + e_k)
        if J_k.ndim == 2:
            J_k = J_k.unsqueeze(0)
        J_pert_list.append(J_k.squeeze(0))  # (525, 64)

    # Stack: (1, 64, 525, 64)  — needed by approx_dg_from_jac
    J_pert = torch.stack(J_pert_list, dim=0).unsqueeze(0)

    # Metric derivatives and regularised Christoffel symbols
    dg = approx_dg_from_jac(J, J_pert, eps)           # (1, 64, 64, 64)
    Gamma = _christoffel_regularised(J, dg)            # (1, 64, 64, 64)

    def gamma_fn(_x: torch.Tensor) -> torch.Tensor:
        """Return frozen Gamma (ignores the query point)."""
        return Gamma

    return gamma_fn, Gamma


# ═══════════════════════════════════════════════════════════════════════════
#  Direction helpers
# ═══════════════════════════════════════════════════════════════════════════

def mutation_direction_projected(
    tangent_space: SubRiemannianTangentSpace,
    pos: int,
    aa_idx: int,
) -> torch.Tensor:
    """Compute latent direction for a mutation via horizontal projection.

    Projects the one-hot ambient vector e_{pos*21+aa} through J_horizontal^+.
    """
    device = tangent_space.device
    e = torch.zeros(AMBIENT_DIM, device=device)
    e[pos * len(ALPHABET) + aa_idx] = 1.0
    return tangent_space.project_ambient_vector_to_horizontal_space(e)  # (64,)


def scale_direction(v: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Scale *v* to have the given L2 norm (no-op if v ≈ 0)."""
    n = torch.norm(v).item()
    if n < 1e-12:
        return v
    return v * (target_norm / n)


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-step methods for projected direction
# ═══════════════════════════════════════════════════════════════════════════

def _quick_tangent_space(encoder_decoder, z: torch.Tensor):
    """Compute softmax-decoder SVD and build SubRiemannianTangentSpace at z.

    Returns (tangent_space, jac) — lighter than compute_svd_and_mutations
    since we skip mutation enumeration.
    """
    z_2d = z.unsqueeze(0) if z.ndim == 1 else z
    jac = decoder_jacobian(
        lambda x: encoder_decoder.decoder_forward(x, softmax=True, flatten=True),
        z_2d,
        jacobian_fn_mode="approx",
        jacobian_fn_kwargs={"jacobian_eps": JACOBIAN_EPS},
    )
    U, S, V = torch.linalg.svd(jac, full_matrices=False)
    ts = SubRiemannianTangentSpace(
        U=U[0], S=S[0], V=V[0],
        horizontal_threshold=HORIZONTAL_THRESH,
        device=str(z.device),
    )
    return ts, jac


def euler_reprojection(
    encoder_decoder,
    z_start: torch.Tensor,
    pos: int,
    aa_idx: int,
    total_dist: float,
    n_steps: int = 10,
) -> tuple[torch.Tensor, list[float]]:
    """Follow the re-projection flow dz/dt = J_H^+(z(t)) @ e_mut.

    At each step recompute the softmax Jacobian SVD and tangent space,
    then take an Euler step in the freshly-projected direction.

    Returns (z_end, cosines) where cosines[i] is the cosine similarity
    between the projected direction and the *remaining* displacement to
    z_end (for diagnostics).
    """
    step_size = total_dist / n_steps
    z = z_start.clone()
    cosines: list[float] = []

    for t in range(n_steps):
        ts, _ = _quick_tangent_space(encoder_decoder, z)
        v_proj = mutation_direction_projected(ts, pos, aa_idx)
        v_norm = torch.norm(v_proj).item()
        if v_norm < 1e-12:
            break
        v_unit = v_proj / v_norm
        z = z + step_size * v_unit
        cosines.append(v_norm)

    return z, cosines


def geodesic_live_gamma(
    encoder_decoder,
    z_start: torch.Tensor,
    v_init: torch.Tensor,
    n_major_steps: int = 5,
    rk4_substeps: int = 4,
) -> torch.Tensor:
    """Geodesic with Gamma recomputed at each major step.

    Splits the unit-time geodesic into n_major_steps segments.
    At each segment, recomputes Christoffel symbols at the current point
    and integrates a short geodesic with rk4_substeps sub-steps.
    """
    z = z_start.unsqueeze(0) if z_start.ndim == 1 else z_start
    v = v_init.unsqueeze(0) if v_init.ndim == 1 else v_init

    dt = 1.0 / n_major_steps

    for step in range(n_major_steps):
        _, Gamma_here = build_frozen_gamma(encoder_decoder, z.squeeze(0))

        def gamma_fn_local(_x, _G=Gamma_here):
            return _G

        z_new, v_new = integrate_geodesic_rk4(
            z, v, gamma_fn_local, t1=dt, n_steps=rk4_substeps,
        )
        if torch.isnan(z_new).any() or torch.isinf(z_new).any():
            break
        z, v = z_new, v_new

    return z.squeeze(0)


def geodesic_reprojection(
    encoder_decoder,
    z_start: torch.Tensor,
    pos: int,
    aa_idx: int,
    total_dist: float,
    n_major_steps: int = 5,
    rk4_substeps: int = 4,
) -> torch.Tensor:
    """Geodesic with both Gamma AND direction recomputed at each step.

    At each major step:
      1. Recompute SVD at current z  → fresh projected direction
      2. Recompute Christoffel symbols at current z
      3. Take a short geodesic step in the fresh direction
    """
    step_dist = total_dist / n_major_steps
    z = z_start.clone()

    for step in range(n_major_steps):
        # Fresh direction from tangent space at current z
        ts, _ = _quick_tangent_space(encoder_decoder, z)
        v_proj = mutation_direction_projected(ts, pos, aa_idx)
        v_proj = scale_direction(v_proj, step_dist)

        # Fresh Christoffel symbols at current z
        _, Gamma_here = build_frozen_gamma(encoder_decoder, z)

        def gamma_fn_local(_x, _G=Gamma_here):
            return _G

        z_2d = z.unsqueeze(0) if z.ndim == 1 else z
        v_2d = v_proj.unsqueeze(0) if v_proj.ndim == 1 else v_proj

        z_new, _ = integrate_geodesic_rk4(
            z_2d, v_2d, gamma_fn_local, t1=1.0, n_steps=rk4_substeps,
        )
        z_new = z_new.squeeze(0)

        if torch.isnan(z_new).any() or torch.isinf(z_new).any():
            break
        z = z_new

    return z


# ═══════════════════════════════════════════════════════════════════════════
#  Decoder gradient methods (∂ log p / ∂z)
# ═══════════════════════════════════════════════════════════════════════════

GRAD_LR = 0.01       # learning rate for gradient ascent
GRAD_STEPS = 150     # gradient ascent steps
GRAD_ALPHA = 0.1     # preservation weight: α in the multi-objective loss


def decoder_logprob_gradient(
    encoder_decoder,
    z: torch.Tensor,
    pos: int,
    aa_idx: int,
) -> torch.Tensor:
    """∂ log p(aa=aa_idx | z, pos) / ∂z  via autograd."""
    z_var = z.clone().detach().requires_grad_(True)
    logits = encoder_decoder.decoder_forward(
        z_var.unsqueeze(0), softmax=False, flatten=False
    )  # (1, 25, 21)
    log_probs = torch.nn.functional.log_softmax(logits[0, pos, :], dim=-1)
    log_probs[aa_idx].backward()
    return z_var.grad.detach()


def multi_obj_gradient(
    encoder_decoder,
    z: torch.Tensor,
    pos: int,
    aa_idx: int,
    peptide_padded: str,
    alpha: float = GRAD_ALPHA,
) -> torch.Tensor:
    """Gradient of: log p(aa_mut | z, pos) + α · Σ_{p≠pos} log p(parent_aa | z, p).

    Maximises the mutation amino acid at the target position while preserving
    the parent amino acids at all other positions.
    """
    z_var = z.clone().detach().requires_grad_(True)
    logits = encoder_decoder.decoder_forward(
        z_var.unsqueeze(0), softmax=False, flatten=False
    )  # (1, 25, 21)

    # Mutation term
    lp_mut = torch.nn.functional.log_softmax(logits[0, pos, :], dim=-1)
    obj = lp_mut[aa_idx]

    # Preservation term (all other positions)
    for p in range(len(peptide_padded.rstrip())):
        if p == pos:
            continue
        parent_aa = ALPHABET.index(peptide_padded[p])
        lp_p = torch.nn.functional.log_softmax(logits[0, p, :], dim=-1)
        obj = obj + alpha * lp_p[parent_aa]

    obj.backward()
    return z_var.grad.detach()


def gradient_ascent_mutation(
    encoder_decoder,
    z_parent: torch.Tensor,
    pos: int,
    aa_idx: int,
    peptide_padded: str,
    lr: float = GRAD_LR,
    n_steps: int = GRAD_STEPS,
    alpha: float = GRAD_ALPHA,
) -> torch.Tensor:
    """Multi-objective gradient ascent to reach a specific mutation.

    Returns z_end after n_steps of gradient ascent.
    """
    z = z_parent.clone()
    for _ in range(n_steps):
        g = multi_obj_gradient(encoder_decoder, z, pos, aa_idx, peptide_padded, alpha)
        z = z + lr * g
    return z


# ═══════════════════════════════════════════════════════════════════════════
#  Core check logic
# ═══════════════════════════════════════════════════════════════════════════

def check_single_mutation(
    encoder_decoder,
    z_parent: torch.Tensor,
    z_mutant: torch.Tensor,
    mutant_peptide: str,
    v_proj: torch.Tensor,
    v_enc: torch.Tensor,
    gamma_fn,
    target_pos: int,
    target_aa_idx: int,
    n_steps: int = 16,
) -> dict:
    """Run all 4 test configurations and return a result dict."""
    device = z_parent.device
    z_p = z_parent.unsqueeze(0)  # (1, 64)

    results: dict = {}

    for dir_name, v in [("proj", v_proj), ("enc", v_enc)]:
        for method, use_geo in [("euc", False), ("geo", True)]:
            key = f"{method}_{dir_name}"
            if use_geo:
                # Adaptive step count: more steps for larger ||v||
                v_norm = torch.norm(v).item()
                adaptive_steps = max(n_steps, int(v_norm * 100))
                adaptive_steps = min(adaptive_steps, 256)  # cap
                z_end = exponential_map(
                    z_p, v.unsqueeze(0), gamma_fn, t1=1.0,
                    n_steps=adaptive_steps,
                )
                # Fall back to Euclidean if geodesic diverges
                if torch.isnan(z_end).any() or torch.isinf(z_end).any():
                    z_end = z_p + v.unsqueeze(0)
                    results[f"geo_fallback_{dir_name}"] = True
            else:
                z_end = z_p + v.unsqueeze(0)

            with torch.no_grad():
                decoded = encoder_decoder.decode_peptides(z_end)[0]

            full_match = decoded.strip() == mutant_peptide.strip()
            pos_match = (
                decoded[target_pos] == ALPHABET[target_aa_idx]
                if target_pos < len(decoded)
                else False
            )

            # Hamming distance
            d_strip = decoded.strip()
            m_strip = mutant_peptide.strip()
            hamming = sum(
                a != b for a, b in zip(d_strip, m_strip)
            ) + abs(len(d_strip) - len(m_strip))

            results[f"decoded_{key}"] = decoded.strip()
            results[f"match_{key}"] = full_match
            results[f"pos_match_{key}"] = pos_match
            results[f"hamming_{key}"] = hamming

    return results


def _eval_endpoint(
    encoder_decoder, z_end: torch.Tensor, mutant_peptide: str,
    target_pos: int, target_aa_idx: int,
) -> dict:
    """Decode z_end and compare to the expected mutant peptide."""
    z_2d = z_end.unsqueeze(0) if z_end.ndim == 1 else z_end
    with torch.no_grad():
        decoded = encoder_decoder.decode_peptides(z_2d)[0]
    d_strip = decoded.strip()
    m_strip = mutant_peptide.strip()
    full_match = d_strip == m_strip
    pos_match = (
        d_strip[target_pos] == ALPHABET[target_aa_idx]
        if target_pos < len(d_strip)
        else False
    )
    hamming = sum(a != b for a, b in zip(d_strip, m_strip)) + abs(len(d_strip) - len(m_strip))
    return {
        "decoded": d_strip,
        "match": full_match,
        "pos_match": pos_match,
        "hamming": hamming,
    }


def check_multistep_mutation(
    encoder_decoder,
    z_parent: torch.Tensor,
    z_mutant: torch.Tensor,
    mutant_peptide: str,
    target_pos: int,
    target_aa_idx: int,
    v_proj_raw_norm: float,
    n_euler_steps: int = 10,
    n_major_steps: int = 5,
) -> dict:
    """Run the three multi-step methods on a single mutation.

    1. euler_reproj: Euler re-projection flow (dz/dt = J_H^+(z) @ e_mut)
    2. geo_live: Geodesic with live Gamma (same initial direction, recomputed Γ)
    3. geo_reproj: Geodesic with BOTH live Gamma AND re-projected direction
    """
    total_dist = torch.norm(z_mutant - z_parent).item()
    results: dict = {}

    # 1. Euler re-projection flow
    z_euler, cosines = euler_reprojection(
        encoder_decoder, z_parent, target_pos, target_aa_idx,
        total_dist=total_dist, n_steps=n_euler_steps,
    )
    ev = _eval_endpoint(encoder_decoder, z_euler, mutant_peptide, target_pos, target_aa_idx)
    for k, val in ev.items():
        results[f"{k}_euler_reproj"] = val

    # 2. Geodesic with live Gamma (initial projected direction, recomputed Γ)
    ts_init, _ = _quick_tangent_space(encoder_decoder, z_parent)
    v_proj_init = mutation_direction_projected(ts_init, target_pos, target_aa_idx)
    v_proj_init = scale_direction(v_proj_init, total_dist)

    z_geo_live = geodesic_live_gamma(
        encoder_decoder, z_parent, v_proj_init,
        n_major_steps=n_major_steps, rk4_substeps=4,
    )
    ev = _eval_endpoint(encoder_decoder, z_geo_live, mutant_peptide, target_pos, target_aa_idx)
    for k, val in ev.items():
        results[f"{k}_geo_live"] = val

    # 3. Geodesic with live Gamma + re-projected direction
    z_geo_reproj = geodesic_reprojection(
        encoder_decoder, z_parent, target_pos, target_aa_idx,
        total_dist=total_dist,
        n_major_steps=n_major_steps, rk4_substeps=4,
    )
    ev = _eval_endpoint(encoder_decoder, z_geo_reproj, mutant_peptide, target_pos, target_aa_idx)
    for k, val in ev.items():
        results[f"{k}_geo_reproj"] = val

    # Cosine between initial projected direction and v_enc (for reference)
    v_enc = z_mutant - z_parent
    cos_init = (
        torch.dot(v_proj_init, v_enc)
        / (torch.norm(v_proj_init) * torch.norm(v_enc) + 1e-12)
    ).item()
    results["cos_proj_enc"] = cos_init

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main(device: str = "cpu") -> dict:
    print("=" * 70)
    print("GEODESIC MUTATION VERIFICATION FOR MUTANG++")
    print("=" * 70)

    t0 = time.time()

    # ------ build models -------------------------------------------------
    print("\n[1/3] Loading encoder-decoder …")
    encoder_decoder = build_encoder_decoder(device)
    log_prob_potential = DecoderLogProbPotential(encoder_decoder=encoder_decoder)

    all_results: dict = {}

    for pep_idx, (pep_name, peptide) in enumerate(SEED_PEPTIDES.items(), 1):
        print(f"\n{'─' * 70}")
        print(f"[{pep_idx}/{len(SEED_PEPTIDES)}] Peptide: {pep_name} = {peptide}")
        print(f"{'─' * 70}")

        padded = peptide.ljust(MAX_PEPTIDE_LEN)

        # ── 1. Encode parent ────────────────────────────────────────────
        with torch.no_grad():
            z_parent = encoder_decoder.encode_peptides([peptide])[0]  # (64,)

        # ── 2. SVD + mutations (notebook style) ─────────────────────────
        jac, U, S, V, mutations, tangent_space = compute_svd_and_mutations(
            encoder_decoder, peptide, z_parent
        )

        if not mutations:
            print("  ⚠ No mutations found — skipping")
            continue

        print(
            f"  SVD done — {len(mutations)} positions mutated, "
            f"horizontal_dim={tangent_space.horizontal_dim}"
        )

        # ── 3. Compute per-position log-prob potentials ─────────────────
        potentials = log_prob_potential.compute(peptide, mutations)

        # ── 4. Flatten, remove identity, softmax, threshold ─────────────
        flat_mutations: list[tuple[int, int, float]] = []  # (pos, aa_idx, log_pot)

        for pos, aa_dict in potentials.items():
            parent_aa_idx = ALPHABET.index(padded[pos])
            for aa_idx, lp in aa_dict.items():
                if aa_idx != parent_aa_idx:
                    flat_mutations.append((pos, aa_idx, lp))

        if not flat_mutations:
            print("  ⚠ No non-identity mutations — skipping")
            continue

        log_pots = np.array([m[2] for m in flat_mutations])
        shifted = log_pots - log_pots.max()
        softmax_probs = np.exp(shifted) / np.exp(shifted).sum()

        n_mut = len(flat_mutations)
        threshold = 1.0 / n_mut

        above = [
            (m, p)
            for m, p in zip(flat_mutations, softmax_probs)
            if p > threshold
        ]

        print(
            f"  Mutations total: {n_mut}  |  "
            f"above 1/{n_mut} = {threshold:.4f} threshold: {len(above)}"
        )

        # ── 5. Build frozen Christoffel symbols at z_parent ─────────────
        print("  Computing Christoffel symbols (frozen at z_parent) …")
        gamma_fn, Gamma = build_frozen_gamma(encoder_decoder, z_parent)
        print(
            f"  Done.  Γ norm={torch.norm(Gamma).item():.2e}  "
            f"max={torch.max(torch.abs(Gamma)).item():.2e}"
        )

        # ── 6. Test each surviving mutation ─────────────────────────────
        pep_results: list[dict] = []

        for idx, ((pos, aa_idx, log_pot), softmax_p) in enumerate(above):
            # Build mutant peptide
            mutant_list = list(peptide)
            mutant_list[pos] = ALPHABET[aa_idx]
            mutant_peptide = "".join(mutant_list)

            # Encode mutant
            with torch.no_grad():
                z_mutant = encoder_decoder.encode_peptides([mutant_peptide])[0]

            # Direction from horizontal projection (SubRiemannianTangentSpace)
            v_proj_raw = mutation_direction_projected(
                tangent_space, pos, aa_idx
            )
            # Scale to match Euclidean distance z_mutant - z_parent
            eucl_dist = torch.norm(z_mutant - z_parent).item()
            v_proj = scale_direction(v_proj_raw, eucl_dist)

            # Direction from encoding
            v_enc = z_mutant - z_parent

            # Run the 4 checks
            checks = check_single_mutation(
                encoder_decoder=encoder_decoder,
                z_parent=z_parent,
                z_mutant=z_mutant,
                mutant_peptide=mutant_peptide,
                v_proj=v_proj,
                v_enc=v_enc,
                gamma_fn=gamma_fn,
                target_pos=pos,
                target_aa_idx=aa_idx,
            )

            entry = {
                "pos": pos,
                "parent_aa": padded[pos],
                "mutant_aa": ALPHABET[aa_idx],
                "softmax_prob": float(softmax_p),
                "log_potential": float(log_pot),
                "mutant_peptide": mutant_peptide,
                "euclidean_dist": eucl_dist,
                "v_proj_raw_norm": torch.norm(v_proj_raw).item(),
                **checks,
            }
            pep_results.append(entry)

            # Pretty print
            geo_ok = "✓" if checks["match_geo_proj"] else "✗"
            euc_ok = "✓" if checks["match_euc_enc"] else "✗"
            print(
                f"  [{idx+1:>2}/{len(above)}] {geo_ok} pos={pos:>2} "
                f"{padded[pos]}→{ALPHABET[aa_idx]}  p={softmax_p:.4f}  "
                f"geo_proj: h={checks['hamming_geo_proj']}  "
                f"euc_enc: h={checks['hamming_euc_enc']}  "
                f"geo_enc: h={checks['hamming_geo_enc']}"
            )

        # ── Per-peptide summary ─────────────────────────────────────────
        n = len(pep_results)
        summary: dict = {}
        if n > 0:
            for key in ["euc_proj", "euc_enc", "geo_proj", "geo_enc"]:
                full = sum(1 for r in pep_results if r[f"match_{key}"])
                pos = sum(1 for r in pep_results if r[f"pos_match_{key}"])
                avg_h = np.mean([r[f"hamming_{key}"] for r in pep_results])
                summary[key] = {
                    "full_match": full,
                    "pos_match": pos,
                    "avg_hamming": float(avg_h),
                }

            print(f"\n  ── Summary for {pep_name} ({n} mutations tested) ──")
            for key, s in summary.items():
                print(
                    f"    {key:<10}: full {s['full_match']:>2}/{n}  "
                    f"pos {s['pos_match']:>2}/{n}  "
                    f"avg_hamming {s['avg_hamming']:.2f}"
                )

        all_results[pep_name] = {
            "peptide": peptide,
            "n_mutations_total": n_mut,
            "n_above_threshold": len(above),
            "gamma_norm": float(torch.norm(Gamma).item()),
            "gamma_max": float(torch.max(torch.abs(Gamma)).item()),
            "summary": summary,
            "mutations": pep_results,
        }

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "geodesic_mutation_check_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Overall summary ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'═' * 70}")

    total = 0
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for data in all_results.values():
        for r in data["mutations"]:
            total += 1
            for key in ["euc_proj", "euc_enc", "geo_proj", "geo_enc"]:
                if r[f"match_{key}"]:
                    totals[key]["full"] += 1
                if r[f"pos_match_{key}"]:
                    totals[key]["pos"] += 1

    print(f"Total mutations tested: {total}")
    if total > 0:
        for key in ["euc_proj", "euc_enc", "geo_proj", "geo_enc"]:
            f = totals[key]["full"]
            p = totals[key]["pos"]
            print(
                f"  {key:<10}: full {f:>3}/{total} ({100*f/total:5.1f}%)  "
                f"pos {p:>3}/{total} ({100*p/total:5.1f}%)"
            )
    print(f"\nElapsed: {elapsed:.1f}s")

    return all_results


def main_multistep(
    device: str = "cpu",
    n_euler_steps: int = 10,
    n_major_steps: int = 5,
) -> dict:
    """Run the multi-step projected-direction experiments.

    Three additional methods:
      - euler_reproj:  Euler flow with re-projected direction at each step
      - geo_live:      Geodesic with live Gamma (frozen initial direction)
      - geo_reproj:    Geodesic with live Gamma AND re-projected direction
    """
    print("=" * 70)
    print("MULTI-STEP PROJECTED DIRECTION EXPERIMENT")
    print(f"  euler_steps={n_euler_steps}  geo_major_steps={n_major_steps}")
    print("=" * 70)

    t0 = time.time()
    encoder_decoder = build_encoder_decoder(device)
    log_prob_potential = DecoderLogProbPotential(encoder_decoder=encoder_decoder)

    all_results: dict = {}
    methods = ["euler_reproj", "geo_live", "geo_reproj"]

    for pep_idx, (pep_name, peptide) in enumerate(SEED_PEPTIDES.items(), 1):
        print(f"\n{'─' * 70}")
        print(f"[{pep_idx}/{len(SEED_PEPTIDES)}] Peptide: {pep_name} = {peptide}")
        print(f"{'─' * 70}")

        padded = peptide.ljust(MAX_PEPTIDE_LEN)
        with torch.no_grad():
            z_parent = encoder_decoder.encode_peptides([peptide])[0]

        jac, U, S, V, mutations, tangent_space = compute_svd_and_mutations(
            encoder_decoder, peptide, z_parent
        )
        if not mutations:
            print("  ⚠ No mutations — skipping")
            continue

        potentials = log_prob_potential.compute(peptide, mutations)
        flat_mutations = []
        for pos, aa_dict in potentials.items():
            parent_aa_idx = ALPHABET.index(padded[pos])
            for aa_idx, lp in aa_dict.items():
                if aa_idx != parent_aa_idx:
                    flat_mutations.append((pos, aa_idx, lp))
        if not flat_mutations:
            continue

        log_pots = np.array([m[2] for m in flat_mutations])
        shifted = log_pots - log_pots.max()
        softmax_probs = np.exp(shifted) / np.exp(shifted).sum()
        n_mut = len(flat_mutations)
        above = [
            (m, p) for m, p in zip(flat_mutations, softmax_probs)
            if p > 1.0 / n_mut
        ]
        print(f"  {len(above)} mutations above threshold")

        pep_results = []
        for idx, ((pos, aa_idx, log_pot), softmax_p) in enumerate(above):
            mutant_list = list(peptide)
            mutant_list[pos] = ALPHABET[aa_idx]
            mutant_peptide = "".join(mutant_list)
            with torch.no_grad():
                z_mutant = encoder_decoder.encode_peptides([mutant_peptide])[0]

            v_proj_raw = mutation_direction_projected(tangent_space, pos, aa_idx)

            print(f"  [{idx+1}/{len(above)}] {padded[pos]}→{ALPHABET[aa_idx]} pos={pos} ...", end=" ", flush=True)
            ms_t0 = time.time()

            ms_checks = check_multistep_mutation(
                encoder_decoder=encoder_decoder,
                z_parent=z_parent,
                z_mutant=z_mutant,
                mutant_peptide=mutant_peptide,
                target_pos=pos,
                target_aa_idx=aa_idx,
                v_proj_raw_norm=torch.norm(v_proj_raw).item(),
                n_euler_steps=n_euler_steps,
                n_major_steps=n_major_steps,
            )

            elapsed_m = time.time() - ms_t0
            hamming_str = "  ".join(
                f"{m}: h={ms_checks.get(f'hamming_{m}', '?')}"
                for m in methods
            )
            print(f"({elapsed_m:.1f}s)  {hamming_str}")

            entry = {
                "pos": pos,
                "parent_aa": padded[pos],
                "mutant_aa": ALPHABET[aa_idx],
                "softmax_prob": float(softmax_p),
                "mutant_peptide": mutant_peptide,
                "euclidean_dist": torch.norm(z_mutant - z_parent).item(),
                "cos_proj_enc": ms_checks.get("cos_proj_enc", None),
                **ms_checks,
            }
            pep_results.append(entry)

        # Per-peptide summary
        n = len(pep_results)
        summary = {}
        if n > 0:
            for m in methods:
                full = sum(1 for r in pep_results if r.get(f"match_{m}", False))
                pos_m = sum(1 for r in pep_results if r.get(f"pos_match_{m}", False))
                avg_h = np.mean([r.get(f"hamming_{m}", 99) for r in pep_results])
                summary[m] = {"full_match": full, "pos_match": pos_m, "avg_hamming": float(avg_h)}

            print(f"\n  ── Summary for {pep_name} ({n} mutations) ──")
            for m, s in summary.items():
                print(f"    {m:<14}: full {s['full_match']:>2}/{n}  pos {s['pos_match']:>2}/{n}  avg_h {s['avg_hamming']:.2f}")

        all_results[pep_name] = {
            "peptide": peptide,
            "n_above_threshold": len(above),
            "summary": summary,
            "mutations": pep_results,
        }

    # Overall summary
    print(f"\n{'═' * 70}")
    print("OVERALL MULTI-STEP SUMMARY")
    print(f"{'═' * 70}")
    total = sum(len(d["mutations"]) for d in all_results.values())
    if total > 0:
        for m in methods:
            full = sum(1 for d in all_results.values() for r in d["mutations"] if r.get(f"match_{m}", False))
            pos_m = sum(1 for d in all_results.values() for r in d["mutations"] if r.get(f"pos_match_{m}", False))
            print(f"  {m:<14}: full {full:>3}/{total} ({100*full/total:5.1f}%)  pos {pos_m:>3}/{total} ({100*pos_m/total:5.1f}%)")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out_path = os.path.join("results", "geodesic_multistep_results.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


def main_gradient(
    device: str = "cpu",
    lr: float = GRAD_LR,
    n_steps: int = GRAD_STEPS,
    alpha: float = GRAD_ALPHA,
) -> dict:
    """Run gradient-ascent projected-direction experiments on all peptides.

    Method:  z_{t+1} = z_t + lr · ∇_z [ log p(aa_mut | z, pos) + α · Σ log p(parent | z, p') ]

    This optimises the decoder output directly (no pseudoinverse, no SVD projection).
    Comparison: also computes single-step gradient direction vs Euclidean/geodesic.
    """
    print("=" * 70)
    print("GRADIENT-BASED MUTATION VERIFICATION")
    print(f"  lr={lr}  steps={n_steps}  α_preserve={alpha}")
    print("=" * 70)

    t0 = time.time()
    encoder_decoder = build_encoder_decoder(device)
    log_prob_potential = DecoderLogProbPotential(encoder_decoder=encoder_decoder)

    all_results: dict = {}
    methods = ["grad_ascent", "grad_1step", "euc_enc", "euc_proj"]

    for pep_idx, (pep_name, peptide) in enumerate(SEED_PEPTIDES.items(), 1):
        print(f"\n{'─' * 70}")
        print(f"[{pep_idx}/{len(SEED_PEPTIDES)}] Peptide: {pep_name} = {peptide}")
        print(f"{'─' * 70}")

        padded = peptide.ljust(MAX_PEPTIDE_LEN)
        with torch.no_grad():
            z_parent = encoder_decoder.encode_peptides([peptide])[0]

        jac, U, S, V, mutations, tangent_space = compute_svd_and_mutations(
            encoder_decoder, peptide, z_parent
        )
        if not mutations:
            print("  ⚠ No mutations — skipping")
            continue

        potentials = log_prob_potential.compute(peptide, mutations)
        flat_mutations = []
        for pos, aa_dict in potentials.items():
            parent_aa_idx = ALPHABET.index(padded[pos])
            for aa_idx, lp in aa_dict.items():
                if aa_idx != parent_aa_idx:
                    flat_mutations.append((pos, aa_idx, lp))
        if not flat_mutations:
            continue

        log_pots = np.array([m[2] for m in flat_mutations])
        shifted = log_pots - log_pots.max()
        softmax_probs = np.exp(shifted) / np.exp(shifted).sum()
        n_mut = len(flat_mutations)
        above = [
            (m, p) for m, p in zip(flat_mutations, softmax_probs)
            if p > 1.0 / n_mut
        ]
        print(f"  {len(above)} mutations above threshold")

        pep_results = []
        for idx, ((pos, aa_idx, log_pot), softmax_p) in enumerate(above):
            mutant_list = list(peptide)
            mutant_list[pos] = ALPHABET[aa_idx]
            mutant_peptide = "".join(mutant_list)
            with torch.no_grad():
                z_mutant = encoder_decoder.encode_peptides([mutant_peptide])[0]

            v_enc = z_mutant - z_parent
            eucl_dist = torch.norm(v_enc).item()

            print(f"  [{idx+1}/{len(above)}] {padded[pos]}→{ALPHABET[aa_idx]} pos={pos} ...", end=" ", flush=True)
            mt0 = time.time()

            entry: dict = {
                "pos": pos,
                "parent_aa": padded[pos],
                "mutant_aa": ALPHABET[aa_idx],
                "softmax_prob": float(softmax_p),
                "mutant_peptide": mutant_peptide,
                "euclidean_dist": eucl_dist,
            }

            # ── Method 1: Multi-objective gradient ascent ──
            z_grad = gradient_ascent_mutation(
                encoder_decoder, z_parent, pos, aa_idx, padded,
                lr=lr, n_steps=n_steps, alpha=alpha,
            )
            ev = _eval_endpoint(encoder_decoder, z_grad, mutant_peptide, pos, aa_idx)
            d_to_mut = torch.norm(z_grad - z_mutant).item()
            d_from_par = torch.norm(z_grad - z_parent).item()
            for k, val in ev.items():
                entry[f"{k}_grad_ascent"] = val
            entry["dist_to_mut_grad"] = d_to_mut
            entry["dist_from_parent_grad"] = d_from_par

            # ── Method 2: Single step in gradient direction (scaled to eucl_dist) ──
            grad_at_parent = decoder_logprob_gradient(encoder_decoder, z_parent, pos, aa_idx)
            v_grad_1step = scale_direction(grad_at_parent, eucl_dist)
            ev1 = _eval_endpoint(
                encoder_decoder, z_parent + v_grad_1step,
                mutant_peptide, pos, aa_idx,
            )
            for k, val in ev1.items():
                entry[f"{k}_grad_1step"] = val

            # ── Method 3: Euclidean with encoder direction (baseline) ──
            ev_enc = _eval_endpoint(
                encoder_decoder, z_parent + v_enc,
                mutant_peptide, pos, aa_idx,
            )
            for k, val in ev_enc.items():
                entry[f"{k}_euc_enc"] = val

            # ── Method 4: Euclidean with projected direction (baseline) ──
            v_proj_raw = mutation_direction_projected(tangent_space, pos, aa_idx)
            v_proj = scale_direction(v_proj_raw, eucl_dist)
            ev_proj = _eval_endpoint(
                encoder_decoder, z_parent + v_proj,
                mutant_peptide, pos, aa_idx,
            )
            for k, val in ev_proj.items():
                entry[f"{k}_euc_proj"] = val
            entry["v_proj_raw_norm"] = torch.norm(v_proj_raw).item()

            # ── Cosine diagnostics ──
            cos_grad_enc = (
                torch.dot(grad_at_parent, v_enc)
                / (torch.norm(grad_at_parent) * torch.norm(v_enc) + 1e-12)
            ).item()
            cos_proj_enc = (
                torch.dot(v_proj_raw, v_enc)
                / (torch.norm(v_proj_raw) * torch.norm(v_enc) + 1e-12)
            ).item()
            cos_grad_proj = (
                torch.dot(grad_at_parent, v_proj_raw)
                / (torch.norm(grad_at_parent) * torch.norm(v_proj_raw) + 1e-12)
            ).item()
            entry["cos_grad_enc"] = cos_grad_enc
            entry["cos_proj_enc"] = cos_proj_enc
            entry["cos_grad_proj"] = cos_grad_proj

            elapsed_m = time.time() - mt0
            hstr = "  ".join(
                f"{m}: h={entry.get(f'hamming_{m}', '?')}"
                for m in methods
            )
            print(f"({elapsed_m:.1f}s)  {hstr}  cos(∇,enc)={cos_grad_enc:+.3f}")

            pep_results.append(entry)

        # Per-peptide summary
        n = len(pep_results)
        summary = {}
        if n > 0:
            for m in methods:
                full = sum(1 for r in pep_results if r.get(f"match_{m}", False))
                pos_m = sum(1 for r in pep_results if r.get(f"pos_match_{m}", False))
                avg_h = np.mean([r.get(f"hamming_{m}", 99) for r in pep_results])
                summary[m] = {"full_match": full, "pos_match": pos_m, "avg_hamming": float(avg_h)}

            print(f"\n  ── Summary for {pep_name} ({n} mutations) ──")
            for m, s in summary.items():
                pct = 100 * s["full_match"] / n
                print(f"    {m:<14}: full {s['full_match']:>2}/{n} ({pct:5.1f}%)  pos {s['pos_match']:>2}/{n}  avg_h {s['avg_hamming']:.2f}")

        all_results[pep_name] = {
            "peptide": peptide,
            "n_above_threshold": len(above),
            "summary": summary,
            "mutations": pep_results,
        }

    # Overall summary
    print(f"\n{'═' * 70}")
    print("OVERALL GRADIENT SUMMARY")
    print(f"{'═' * 70}")
    total = sum(len(d["mutations"]) for d in all_results.values())
    if total > 0:
        for m in methods:
            full = sum(1 for d in all_results.values() for r in d["mutations"] if r.get(f"match_{m}", False))
            pos_m = sum(1 for d in all_results.values() for r in d["mutations"] if r.get(f"pos_match_{m}", False))
            avg_cos = np.mean([
                r.get("cos_grad_enc", 0)
                for d in all_results.values()
                for r in d["mutations"]
            ]) if m == "grad_ascent" else None
            extra = f"  avg_cos(∇,enc)={avg_cos:.3f}" if avg_cos is not None else ""
            print(f"  {m:<14}: full {full:>3}/{total} ({100*full/total:5.1f}%)  pos {pos_m:>3}/{total} ({100*pos_m/total:5.1f}%){extra}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out_path = os.path.join("results", "geodesic_gradient_results.json")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Geodesic mutation verification for MUTANG++"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device (cpu or cuda)"
    )
    parser.add_argument(
        "--multistep", action="store_true",
        help="Run multi-step projected-direction experiments"
    )
    parser.add_argument(
        "--gradient", action="store_true",
        help="Run gradient-ascent mutation experiments"
    )
    parser.add_argument(
        "--euler-steps", type=int, default=10,
        help="Number of Euler re-projection steps"
    )
    parser.add_argument(
        "--major-steps", type=int, default=5,
        help="Number of major steps for geodesic methods"
    )
    parser.add_argument(
        "--grad-lr", type=float, default=GRAD_LR,
        help="Learning rate for gradient ascent"
    )
    parser.add_argument(
        "--grad-steps", type=int, default=GRAD_STEPS,
        help="Number of gradient ascent steps"
    )
    parser.add_argument(
        "--grad-alpha", type=float, default=GRAD_ALPHA,
        help="Preservation weight α for gradient ascent"
    )
    args = parser.parse_args()

    if args.gradient:
        main_gradient(
            device=args.device,
            lr=args.grad_lr,
            n_steps=args.grad_steps,
            alpha=args.grad_alpha,
        )
    elif args.multistep:
        main_multistep(
            device=args.device,
            n_euler_steps=args.euler_steps,
            n_major_steps=args.major_steps,
        )
    else:
        main(device=args.device)
