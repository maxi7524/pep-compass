"""Helpers for the MUTANG++ geodesic sanity-check notebook.

For each parent peptide we:
  1. Encode → z, compute softmax-decoder Jacobian SVD at z.
  2. Enumerate MUTANG candidates {position: [aa_indices]}.
  3. Score the full Cartesian product with ``ProjectedDirectionPairwiseSimilarityPotential``
     and take the top fraction.
  4. Build frozen Christoffel symbols Γ at z (raw-decoder Jacobian + regularised metric).
  5. For each top mutant, project the ambient direction Σ_p e_{p, new_aa_p} into the
     horizontal subspace, integrate a geodesic with RK4 for unit time, decode the
     endpoint and record whether it matches the predicted mutant.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from einops import einsum, rearrange

from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace
from pep_compass.local_enumeration.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    ProjectedDirectionPairwiseSimilarityPotential,
    compose_mutant_distribution,
)
from pep_compass.geometry.utils import (
    approx_dg_from_jac,
    exponential_map,
)


# ---- Constants (match scripts/geodesic_mutation_check.py + mutang++.ipynb) ----
LATENT_DIM = 64
ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
ALPHABET_SIZE = len(ALPHABET)
MAX_PEPTIDE_LEN = 25
AMBIENT_DIM = MAX_PEPTIDE_LEN * ALPHABET_SIZE

# Softmax SVD (used by MUTANG / projection) — tight finite-difference step
JACOBIAN_EPS_SOFTMAX = 1e-6
DIRECTION_SIG_THRESH = 1e-6
TOKEN_THRESH = 1e-4
HORIZONTAL_THRESH = 1e-4
MIN_DIRECTIONS = 5

# Christoffel (raw decoder Jacobian) — larger step for numerical stability
GAMMA_EPS = 0.05
METRIC_REG = 1.0

# Bound the MUTANG++ Cartesian product so the test stays tractable AND each
# tested mutant differs from the parent at a modest number of positions
# (matching MUTANG++'s typical low-edit-distance regime).
MAX_CANDIDATES_PER_PARENT = 10_000
MAX_POSITIONS_PER_PARENT = 6     # keep at most this many MUTANG-identified positions
MAX_AAS_PER_POSITION = 8         # per-position candidate cap


def cap_mutations(
    mutations: dict[int, list[int]],
    max_total: int = MAX_CANDIDATES_PER_PARENT,
    max_positions: int = MAX_POSITIONS_PER_PARENT,
    max_aa_per_pos: int = MAX_AAS_PER_POSITION,
    rng: random.Random | None = None,
) -> tuple[dict[int, list[int]], int, int, int]:
    """Cap the Cartesian product size of MUTANG candidates.

    Strategy:
      1. ``mutations`` is iteration-ordered by descending singular-value
         significance (``get_mutations_from_s_u`` walks directions in S-order
         and inserts the chosen position the first time it appears).
      2. Keep only the first ``max_positions`` positions in insertion order.
      3. Cap each position to ``max_aa_per_pos`` AAs (subsampling).
      4. If the product still exceeds ``max_total``, drop the trailing
         (least-significant) position until it fits.

    Returns ``(capped_mutations, raw_total, capped_total, dropped_positions)``.
    """
    rng = rng or random.Random()
    positions = list(mutations.keys())  # insertion order = significance order
    if not positions:
        return mutations, 0, 0, 0

    raw_total = math.prod(len(mutations[pos]) for pos in positions)
    if raw_total <= max_total and len(positions) <= max_positions and all(
        len(mutations[p]) <= max_aa_per_pos for p in positions
    ):
        return mutations, raw_total, raw_total, 0

    dropped_positions = max(0, len(positions) - max_positions)
    kept_positions = positions[:max_positions]

    capped: dict[int, list[int]] = {}
    for pos in kept_positions:
        aas = list(mutations[pos])
        if len(aas) > max_aa_per_pos:
            aas = rng.sample(aas, max_aa_per_pos)
        capped[pos] = aas

    def product_size(muts: dict[int, list[int]]) -> int:
        return math.prod(len(v) for v in muts.values()) if muts else 0

    while product_size(capped) > max_total and len(capped) > 1:
        # Drop the trailing (least-significant) kept position.
        drop_pos = list(capped.keys())[-1]
        capped.pop(drop_pos)
        dropped_positions += 1

    return capped, raw_total, product_size(capped), dropped_positions


def _christoffel_regularised(
    jac: torch.Tensor, dg: torch.Tensor, reg: float = METRIC_REG
) -> torch.Tensor:
    """Christoffel symbols with regularised metric g_reg = J^T J + reg * I.

    The raw pullback metric is near-singular (few large singular values), so
    ``torch.linalg.inv(g)`` explodes. The diagonal regularisation keeps Γ bounded
    so RK4 integration stays stable.
    """
    g = einsum(jac, jac, "b m i, b m j -> b i j")
    g_reg = g + reg * torch.eye(g.shape[-1], device=g.device).unsqueeze(0)
    g_inv = torch.linalg.inv(g_reg)

    term1 = rearrange(dg, "b l i j -> b l i j")  # ∂_j g_{l i}
    term2 = rearrange(dg, "b l j i -> b l i j")  # ∂_i g_{l j}
    term3 = rearrange(dg, "b i j l -> b l i j")  # ∂_l g_{i j}
    T = term1 + term2 - term3

    Gamma = 0.5 * einsum(g_inv, T, "b k l, b l i j -> b k i j")
    return Gamma


def build_frozen_gamma(encoder_decoder, z: torch.Tensor, eps: float = GAMMA_EPS):
    """Compute frozen Christoffel symbols at z. Returns (gamma_fn, Gamma).

    Uses the raw (non-softmax) decoder Jacobian via the encoder_decoder's own
    ``decoder_jacobian`` method (eps set at construction). The finite-difference
    perturbation for ∂g is ``eps`` (default 0.05).
    """
    z_2d = z.unsqueeze(0) if z.ndim == 1 else z
    device = z.device

    with torch.no_grad():
        J = encoder_decoder.decoder_jacobian(z_2d)
    if J.ndim == 2:
        J = J.unsqueeze(0)

    J_pert_list = []
    for k in range(LATENT_DIM):
        e_k = torch.zeros(1, LATENT_DIM, device=device)
        e_k[0, k] = eps
        with torch.no_grad():
            J_k = encoder_decoder.decoder_jacobian(z_2d + e_k)
        if J_k.ndim == 2:
            J_k = J_k.unsqueeze(0)
        J_pert_list.append(J_k.squeeze(0))

    J_pert = torch.stack(J_pert_list, dim=0).unsqueeze(0)  # (1, 64, 525, 64)
    dg = approx_dg_from_jac(J, J_pert, eps)
    Gamma = _christoffel_regularised(J, dg)

    def gamma_fn(_x: torch.Tensor) -> torch.Tensor:
        return Gamma

    return gamma_fn, Gamma


def compute_svd_and_mutations(encoder_decoder, peptide: str, z: torch.Tensor):
    """Softmax-decoder Jacobian → SVD → MUTANG mutations → SubRiemannianTangentSpace."""
    z_2d = z.unsqueeze(0) if z.ndim == 1 else z
    pep_len = len(peptide)

    jac = decoder_jacobian(
        lambda x: encoder_decoder.decoder_forward(x, softmax=True, flatten=True),
        z_2d,
        jacobian_fn_mode="approx",
        jacobian_fn_kwargs={"jacobian_eps": JACOBIAN_EPS_SOFTMAX},
    )  # (1, 525, 64)

    U, S, V = torch.linalg.svd(jac, full_matrices=False)

    enumerator = MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=DIRECTION_SIG_THRESH,
        min_number_of_directions=MIN_DIRECTIONS,
        token_threshold=TOKEN_THRESH,
        alphabet=ALPHABET,
    )
    mutations = enumerator.get_mutations_from_s_u(
        s=S[0].detach().cpu().numpy(),
        u=U[0].detach().cpu().numpy(),
    )
    mutations = {pos: muts for pos, muts in mutations.items() if pos < pep_len}

    tangent_space = SubRiemannianTangentSpace(
        U=U[0],
        S=S[0],
        V=V[0],
        horizontal_threshold=HORIZONTAL_THRESH,
        device=str(z.device),
    )
    return jac, U, S, V, mutations, tangent_space


def get_top_mutants(
    encoder_decoder,
    peptide: str,
    z: torch.Tensor | None = None,
    top_frac: float = 0.2,
    max_mutants: int = 200,
    rng: random.Random | None = None,
):
    """Run MUTANG++ scoring, return ``(tangent_space, sequences, log_potentials, info)``
    for the top ``top_frac`` of mutants (capped at ``max_mutants``)."""
    if z is None:
        with torch.no_grad():
            z = encoder_decoder.encode_peptides([peptide])[0]

    _, U, S, V, mutations, ts = compute_svd_and_mutations(encoder_decoder, peptide, z)

    info = {
        "n_positions_raw": len(mutations),
        "raw_product_size": 0,
        "capped_product_size": 0,
        "dropped_positions": 0,
        "n_total_scored": 0,
    }
    if not mutations:
        return ts, [], np.array([]), info

    capped_mutations, raw_total, capped_total, dropped = cap_mutations(
        mutations, max_total=MAX_CANDIDATES_PER_PARENT, rng=rng,
    )
    info["raw_product_size"] = int(raw_total)
    info["capped_product_size"] = int(capped_total)
    info["dropped_positions"] = int(dropped)

    if not capped_mutations:
        return ts, [], np.array([]), info

    potential = ProjectedDirectionPairwiseSimilarityPotential(
        tangent_space=ts, alphabet=ALPHABET,
    )
    dist = compose_mutant_distribution(
        parent_peptide=peptide,
        mutations=capped_mutations,
        potential=potential,
        alphabet=ALPHABET,
        max_len=MAX_PEPTIDE_LEN,
        include_parent_residue=False,
        sample_combinations=None,
        top_k=None,
    )
    info["n_total_scored"] = int(len(dist.sequences))
    if len(dist.sequences) == 0:
        return ts, [], np.array([]), info

    n_total = len(dist.sequences)
    n_top = max(1, int(np.ceil(n_total * top_frac)))
    n_top = min(n_top, max_mutants)
    return ts, list(dist.sequences[:n_top]), np.array(dist.log_potentials[:n_top]), info


def aggregate_projected_direction(
    tangent_space: SubRiemannianTangentSpace, parent: str, mutant: str
):
    """Build ambient one-hot Σ_{p mutated} e_{p, new_aa_p} (525-dim), project via
    J_H⁺, return (v_latent, list_of_mutated_positions). Both strings are padded to
    MAX_PEPTIDE_LEN with spaces."""
    device = tangent_space.device
    parent_padded = parent.ljust(MAX_PEPTIDE_LEN)
    mutant_padded = mutant.ljust(MAX_PEPTIDE_LEN)

    ambient = torch.zeros(AMBIENT_DIM, device=device)
    mutated_positions: list[int] = []
    for p in range(MAX_PEPTIDE_LEN):
        if parent_padded[p] != mutant_padded[p]:
            aa = mutant_padded[p]
            if aa not in ALPHABET:
                continue
            aa_idx = ALPHABET.index(aa)
            ambient[p * ALPHABET_SIZE + aa_idx] = 1.0
            mutated_positions.append(p)

    v = tangent_space.project_ambient_vector_to_horizontal_space(ambient)
    return v, mutated_positions


def geodesic_check_mutant(
    encoder_decoder,
    z_parent: torch.Tensor,
    parent: str,
    mutant: str,
    tangent_space: SubRiemannianTangentSpace,
    gamma_fn,
    n_steps: int = 32,
    t1: float = 1.0,
) -> dict:
    """Project the (parent → mutant) ambient direction, geodesic-traverse, decode,
    return per-mutant match statistics."""
    v, mutated_positions = aggregate_projected_direction(tangent_space, parent, mutant)
    n_mutated = len(mutated_positions)
    if n_mutated == 0:
        return {
            "decoded": parent,
            "full_match": False,
            "pos_match_count": 0,
            "pos_match_frac": float("nan"),
            "n_mutated": 0,
            "hamming": 0,
            "direction_norm": 0.0,
            "geo_fallback": False,
        }

    z_p = z_parent.unsqueeze(0) if z_parent.ndim == 1 else z_parent
    v_2d = v.unsqueeze(0)
    v_norm = float(torch.norm(v).item())

    fallback = False
    # Use the same adaptive step pattern as scripts/geodesic_mutation_check.py:
    # more RK4 steps for larger ‖v‖, capped at 256.
    adaptive_steps = min(256, max(n_steps, int(v_norm * 100)))
    z_end = exponential_map(z_p, v_2d, gamma_fn, t1=t1, n_steps=adaptive_steps)
    if torch.isnan(z_end).any() or torch.isinf(z_end).any():
        z_end = z_p + v_2d  # Euclidean fallback when geodesic diverges
        fallback = True

    with torch.no_grad():
        decoded = encoder_decoder.decode_peptides(z_end)[0]
    decoded_strip = decoded.strip()
    mutant_strip = mutant.strip()
    full_match = decoded_strip == mutant_strip

    pos_match_count = 0
    decoded_padded = decoded_strip.ljust(MAX_PEPTIDE_LEN)
    mutant_padded = mutant.ljust(MAX_PEPTIDE_LEN)
    for p in mutated_positions:
        if decoded_padded[p] == mutant_padded[p]:
            pos_match_count += 1
    pos_match_frac = pos_match_count / max(1, n_mutated)

    hamming = sum(a != b for a, b in zip(decoded_strip, mutant_strip)) + abs(
        len(decoded_strip) - len(mutant_strip)
    )

    return {
        "decoded": decoded_strip,
        "full_match": bool(full_match),
        "pos_match_count": int(pos_match_count),
        "pos_match_frac": float(pos_match_frac),
        "n_mutated": int(n_mutated),
        "hamming": int(hamming),
        "direction_norm": float(v_norm),
        "geo_fallback": bool(fallback),
    }


def run_one_parent(
    encoder_decoder,
    peptide: str,
    top_frac: float = 0.2,
    max_mutants: int = 200,
    n_steps: int = 32,
    t1: float = 1.0,
    rng: random.Random | None = None,
) -> list[dict]:
    """Full per-parent pipeline. Returns one row per tested mutant
    (empty list if peptide is too long or has no MUTANG++ candidates)."""
    if len(peptide) == 0 or len(peptide) > MAX_PEPTIDE_LEN:
        return []
    with torch.no_grad():
        z = encoder_decoder.encode_peptides([peptide])[0]

    ts, sequences, log_potentials, info = get_top_mutants(
        encoder_decoder, peptide, z,
        top_frac=top_frac, max_mutants=max_mutants, rng=rng,
    )
    if not sequences:
        return []

    gamma_fn, _ = build_frozen_gamma(encoder_decoder, z)

    rows = []
    for rank, (mutant, lp) in enumerate(zip(sequences, log_potentials)):
        r = geodesic_check_mutant(
            encoder_decoder, z, peptide, mutant, ts, gamma_fn,
            n_steps=n_steps, t1=t1,
        )
        r.update({
            "parent": peptide,
            "parent_len": len(peptide),
            "mutant": mutant,
            "rank": rank,
            "log_potential": float(lp),
            "n_top": len(sequences),
            "raw_product_size": info["raw_product_size"],
            "capped_product_size": info["capped_product_size"],
            "dropped_positions": info["dropped_positions"],
        })
        rows.append(r)
    return rows


def summarize_results(rows: list[dict]):
    """Aggregate rows into a JSON-friendly summary + the full DataFrame."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return {"n_rows": 0}, df

    by_n_mut = {}
    for k, sub in df.groupby("n_mutated"):
        by_n_mut[int(k)] = {
            "n": int(len(sub)),
            "full_match_rate": float(sub["full_match"].mean()),
            "pos_match_mean": float(sub["pos_match_frac"].mean()),
            "mean_hamming": float(sub["hamming"].mean()),
        }

    rank_deciles = {}
    try:
        df_ranked = df.copy()
        df_ranked["rank_decile"] = (
            df_ranked.groupby("parent")["rank"]
            .transform(lambda r: pd.qcut(r, q=min(10, max(1, r.nunique())), labels=False, duplicates="drop"))
        )
        for k, sub in df_ranked.groupby("rank_decile"):
            rank_deciles[int(k)] = {
                "n": int(len(sub)),
                "full_match_rate": float(sub["full_match"].mean()),
                "pos_match_mean": float(sub["pos_match_frac"].mean()),
            }
    except Exception:
        pass

    summary = {
        "n_parents": int(df["parent"].nunique()),
        "n_rows": int(len(df)),
        "full_match_rate": float(df["full_match"].mean()),
        "pos_match_mean": float(df["pos_match_frac"].mean()),
        "mean_hamming": float(df["hamming"].mean()),
        "mean_n_mutated": float(df["n_mutated"].mean()),
        "geo_fallback_rate": float(df["geo_fallback"].mean()),
        "by_n_mutated": by_n_mut,
        "by_rank_decile": rank_deciles,
    }
    return summary, df
