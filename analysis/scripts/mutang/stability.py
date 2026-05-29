"""Functions for the Mutang stability study (RQ3 a/b/c).

Mutang = ``MutationEnumerationInTangentSpace``: encode a peptide with HydrAMP,
take the decoder Jacobian, SVD it, and turn the top singular directions into
candidate point mutations. This module provides reusable helpers to probe its
three structural weaknesses:

    (a) Cartesian-product combinatorial explosion of joint mutations.
    (b) Instability of the mutation set under threshold selection across peptides.
    (c) Sensitivity to Jacobian approximation quality (strict autograd vs.
        finite-difference ``approx`` with step ``jacobian_eps``).

Everything here is enumeration-level (no APEX scoring) and depends only on
``pep_compass`` + numpy/torch/pandas, so it is importable in isolation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from pep_compass.local_enumeration.mutation.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)
from pep_compass.models.encoder_decoder.utils import (
    decoder_jacobian_approx,
    decoder_jacobian_strict,
)

# The 6 SAASBO optimization targets from scripts/run_optimization.py. Mutang's
# weaknesses matter most exactly on the peptides the project actually optimizes.
BENCHMARK_PEPTIDES: dict[str, str] = {
    "middle-1": "FLYKWWIRIGRLKL",
    "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN",
    "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}

ALPHABET: list[str] = list(" ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA: set[str] = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN: int = 25


# --------------------------------------------------------------------------- #
# Setup / shared helpers
# --------------------------------------------------------------------------- #
def load_model(
    jacobian_mode: str = "approx",
    jacobian_eps: float = 1e-6,
    field_eps: float = 1e-6,
    device: str | torch.device = "cpu",
) -> HydrAMPEncoderDecoder:
    """Load HydrAMP. ``jacobian_mode`` only affects ``model.decoder_jacobian``;
    the helpers below call the strict/approx functions directly, so a single
    loaded model can serve both modes."""
    model = HydrAMPEncoderDecoder(
        jacobian_mode=jacobian_mode,
        jacobian_eps=jacobian_eps,
        field_eps=field_eps,
        device=torch.device(device),
    )
    model.eval()
    return model


def sample_peptides(
    csv_path: str | Path,
    n: int,
    max_len: int = MAX_LEN,
    seed: int = 0,
    seq_col: str = "Sequence",
) -> list[str]:
    """Reproducibly sample ``n`` peptides from a CSV.

    Drops sequences longer than ``max_len`` (HydrAMP truncates at 25, which
    would corrupt the comparison) and sequences containing non-standard amino
    acids (``to_one_hot`` only knows the 20 canonical residues + space).
    """
    df = pd.read_csv(csv_path)
    seqs = df[seq_col].dropna().astype(str).str.strip()
    valid = seqs[
        (seqs.str.len() <= max_len)
        & (seqs.str.len() > 0)
        & seqs.apply(lambda s: set(s) <= STANDARD_AA)
    ].drop_duplicates()
    take = min(n, len(valid))
    return valid.sample(n=take, random_state=seed).tolist()


def encode(model: HydrAMPEncoderDecoder, peptide: str) -> torch.Tensor:
    """Encode a peptide to its latent mean, shape ``[1, latent_dim]``."""
    return model.encode_peptides([peptide]).detach()


def jacobian_strict_np(
    model: HydrAMPEncoderDecoder, z: torch.Tensor, vectorize: bool = True
) -> np.ndarray:
    """Exact decoder Jacobian via autograd, shape ``[ambient_dim, latent_dim]``.

    ``vectorize=True`` batches the 525 vector-Jacobian products (≈5x faster on
    CPU, numerically identical to the library ``decoder_jacobian_strict`` up to
    float noise). Set ``vectorize=False`` to use the library function verbatim.
    """
    if not vectorize:
        return decoder_jacobian_strict(model.decoder_forward, z)[0].detach().cpu().numpy()
    jac = torch.autograd.functional.jacobian(
        model.decoder_forward, z, vectorize=True
    )[0]
    return rearrange(jac, "a b d -> b a d")[0].detach().cpu().numpy()


def jacobian_approx_np(
    model: HydrAMPEncoderDecoder, z: torch.Tensor, eps: float
) -> np.ndarray:
    """Finite-difference decoder Jacobian with step ``eps``, shape
    ``[ambient_dim, latent_dim]``."""
    jac = decoder_jacobian_approx(model.decoder_forward, z, eps)
    return jac[0].detach().cpu().numpy()


def svd_of_jacobian(jac_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(U, S)`` from ``np.linalg.svd(full_matrices=False)``."""
    U, S, _ = np.linalg.svd(jac_np, full_matrices=False)
    return U, S


def make_mutang(
    direction_significance_threshold: float,
    token_threshold: float,
    max_len: int = MAX_LEN,
) -> MutationEnumerationInTangentSpace:
    return MutationEnumerationInTangentSpace(
        max_len=max_len,
        direction_significance_threshold=direction_significance_threshold,
        token_threshold=token_threshold,
    )


def compute_svd(
    model: HydrAMPEncoderDecoder,
    peptide: str,
    mode: str = "approx",
    eps: float = 1e-6,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode + Jacobian + SVD for one peptide, with optional memoization.

    ``cache`` (if given) is keyed by ``(peptide, mode, eps)`` so repeated
    threshold sweeps over the same peptide reuse one SVD.
    """
    key = (peptide, mode, eps if mode == "approx" else None)
    if cache is not None and key in cache:
        return cache[key]
    z = encode(model, peptide)
    if mode == "strict":
        jac = jacobian_strict_np(model, z)
    elif mode == "approx":
        jac = jacobian_approx_np(model, z, eps)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    result = svd_of_jacobian(jac)
    if cache is not None:
        cache[key] = result
    return result


# --------------------------------------------------------------------------- #
# (a) Combinatorial explosion
# --------------------------------------------------------------------------- #
def mutation_dict(
    S: np.ndarray, U: np.ndarray, mutang: MutationEnumerationInTangentSpace
) -> dict[int, list[int]]:
    """Thin wrapper over ``get_mutations_from_s_u`` -> {position: [aa indices]}."""
    return mutang.get_mutations_from_s_u(S, U)


def n_directions(
    S: np.ndarray, mutang: MutationEnumerationInTangentSpace
) -> int:
    """Number of significant SVD directions Mutang would use for these thresholds."""
    return int(
        max(
            (S > mutang.direction_significance_threshold).sum(),
            mutang.min_number_of_directions,
        )
    )


def single_position_count(mutations: dict[int, list[int]]) -> int:
    """Number of single-position substitution candidates (sum over positions)."""
    return sum(len(v) for v in mutations.values())


def joint_count_analytic(
    peptide: str,
    mutations: dict[int, list[int]],
    mutang: MutationEnumerationInTangentSpace,
) -> int:
    """Size of the Cartesian-product joint enumeration, computed analytically.

    Mirrors ``mutate_peptide``/``aux_mutate`` (each position branches over its
    candidate tokens plus the original residue) without materializing the
    (possibly millions-large) list. Equals ``len(mutang.mutate_peptide(...))``.
    """
    padded = (peptide + " " * mutang.max_len)[: mutang.max_len]
    product = 1
    for pos in range(mutang.max_len):
        tokens = set(mutations.get(pos, []))
        tokens.add(mutang.alphabet.index(padded[pos]))
        product *= len(tokens)
    return product


def explosion_scan(
    peptides: dict[str, str] | list[str],
    model: HydrAMPEncoderDecoder,
    thresholds: list[tuple[float, float]],
    mode: str = "approx",
    eps: float = 1e-6,
    cache: dict | None = None,
) -> pd.DataFrame:
    """For each peptide x (d, t) threshold pair, count single vs joint mutants."""
    named = peptides if isinstance(peptides, dict) else {p: p for p in peptides}
    if cache is None:
        cache = {}
    rows = []
    for name, seq in named.items():
        U, S = compute_svd(model, seq, mode=mode, eps=eps, cache=cache)
        for d_thresh, t_thresh in thresholds:
            mutang = make_mutang(d_thresh, t_thresh)
            mutations = mutation_dict(S, U, mutang)
            joint = joint_count_analytic(seq, mutations, mutang)
            rows.append(
                {
                    "peptide": name,
                    "sequence": seq,
                    "len": len(seq),
                    "d_thresh": d_thresh,
                    "t_thresh": t_thresh,
                    "n_directions": n_directions(S, mutang),
                    "n_positions": len(mutations),
                    "single_count": single_position_count(mutations),
                    "joint_count": joint,
                    "log10_joint": math.log10(joint) if joint > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (b) Threshold instability
# --------------------------------------------------------------------------- #
def enumerate_set(
    peptide: str,
    S: np.ndarray,
    U: np.ndarray,
    d_thresh: float,
    t_thresh: float,
) -> set[str]:
    """Deterministic set of single-position mutant strings for one (d, t) pair.

    Mirrors ``get_mutants_from_single_position_mutations`` and drops identities.
    """
    mutang = make_mutang(d_thresh, t_thresh)
    mutations = mutang.get_mutations_from_s_u(S, U)
    mutants: set[str] = set()
    for pos, muts in mutations.items():
        for m in muts:
            candidate = peptide[:pos] + mutang.alphabet[m] + peptide[pos + 1 :]
            if candidate != peptide:
                mutants.add(candidate)
    return mutants


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity; two empty sets are defined as identical (1.0)."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def threshold_grid_scan(
    peptides: dict[str, str] | list[str],
    model: HydrAMPEncoderDecoder,
    d_grid: list[float],
    t_grid: list[float],
    mode: str = "approx",
    eps: float = 1e-6,
    cache: dict | None = None,
) -> pd.DataFrame:
    """Per peptide x (d, t) on the full grid: n_positions and n_mutants.

    One SVD per peptide (cached); the threshold sweep is pure numpy, so this
    scales to thousands of peptides.
    """
    named = peptides if isinstance(peptides, dict) else {p: p for p in peptides}
    if cache is None:
        cache = {}
    rows = []
    for name, seq in named.items():
        U, S = compute_svd(model, seq, mode=mode, eps=eps, cache=cache)
        for d_thresh in d_grid:
            for t_thresh in t_grid:
                mutants = enumerate_set(seq, S, U, d_thresh, t_thresh)
                mutang = make_mutang(d_thresh, t_thresh)
                n_pos = len(mutang.get_mutations_from_s_u(S, U))
                rows.append(
                    {
                        "peptide": name,
                        "len": len(seq),
                        "d_thresh": d_thresh,
                        "t_thresh": t_thresh,
                        "n_positions": n_pos,
                        "n_mutants": len(mutants),
                    }
                )
    return pd.DataFrame(rows)


def cross_peptide_instability(scan_df: pd.DataFrame) -> pd.DataFrame:
    """For each (d, t): mean/std/CV of n_mutants across peptides.

    A high coefficient of variation (CV = std / mean) means a single threshold
    setting behaves very differently from peptide to peptide.
    """
    grouped = (
        scan_df.groupby(["d_thresh", "t_thresh"])["n_mutants"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    grouped["cv"] = grouped["std"] / grouped["mean"].replace(0, np.nan)
    return grouped


def set_sensitivity_curve(
    peptide: str,
    S: np.ndarray,
    U: np.ndarray,
    grid: list[tuple[float, float]],
) -> list[float]:
    """Jaccard overlap of the mutant set between consecutive grid points.

    Low values mean the enumerated set churns rapidly as the threshold moves
    (i.e. the result is unstable to small threshold changes).
    """
    sets = [enumerate_set(peptide, S, U, d, t) for d, t in grid]
    return [jaccard(sets[i], sets[i + 1]) for i in range(len(sets) - 1)]


# --------------------------------------------------------------------------- #
# (c) Jacobian approximation sensitivity
# --------------------------------------------------------------------------- #
def relative_frobenius_error(approx: np.ndarray, exact: np.ndarray) -> float:
    """||approx - exact||_F / ||exact||_F."""
    denom = np.linalg.norm(exact)
    return float(np.linalg.norm(approx - exact) / denom) if denom else float("nan")


def singular_value_rel_error(
    S_approx: np.ndarray, S_exact: np.ndarray, k: int
) -> float:
    """Relative L2 error of the top-k singular values."""
    a, e = S_approx[:k], S_exact[:k]
    denom = np.linalg.norm(e)
    return float(np.linalg.norm(a - e) / denom) if denom else float("nan")


def subspace_overlap(U1: np.ndarray, U2: np.ndarray, k: int) -> float:
    """Mean cosine of the principal angles between the top-k left-singular
    subspaces. 1.0 = identical subspace, 0.0 = orthogonal. Sign/order-robust."""
    k = min(k, U1.shape[1], U2.shape[1])
    m = U1[:, :k].T @ U2[:, :k]
    cosines = np.linalg.svd(m, compute_uv=False)
    return float(np.clip(cosines, 0.0, 1.0).mean())


def eps_sensitivity_scan(
    peptides: dict[str, str] | list[str],
    model: HydrAMPEncoderDecoder,
    eps_grid: list[float],
    d_thresh: float,
    t_thresh: float,
    k: int = 10,
) -> pd.DataFrame:
    """Compare approx (per eps) against strict (exact) Jacobian per peptide.

    Columns: peptide, eps, jac_rel_error, subspace_overlap, sv_rel_error,
    mut_jaccard (overlap of resulting single-position mutation sets).
    Expect a U-shaped error curve in eps (truncation vs. float cancellation).
    """
    named = peptides if isinstance(peptides, dict) else {p: p for p in peptides}
    rows = []
    for name, seq in named.items():
        z = encode(model, seq)
        jac_exact = jacobian_strict_np(model, z)
        U_e, S_e = svd_of_jacobian(jac_exact)
        exact_set = enumerate_set(seq, S_e, U_e, d_thresh, t_thresh)
        for eps in eps_grid:
            jac_a = jacobian_approx_np(model, z, eps)
            U_a, S_a = svd_of_jacobian(jac_a)
            approx_set = enumerate_set(seq, S_a, U_a, d_thresh, t_thresh)
            rows.append(
                {
                    "peptide": name,
                    "eps": eps,
                    "jac_rel_error": relative_frobenius_error(jac_a, jac_exact),
                    "subspace_overlap": subspace_overlap(U_a, U_e, k),
                    "sv_rel_error": singular_value_rel_error(S_a, S_e, k),
                    "mut_jaccard": jaccard(approx_set, exact_set),
                }
            )
    return pd.DataFrame(rows)
