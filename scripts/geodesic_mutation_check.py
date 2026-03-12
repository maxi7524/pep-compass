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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Geodesic mutation verification for MUTANG++"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device (cpu or cuda)"
    )
    args = parser.parse_args()
    main(device=args.device)
