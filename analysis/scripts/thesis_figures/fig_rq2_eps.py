"""RQ2(c) -- Jacobian finite-difference epsilon sensitivity (refresh).

Recreates the eps sweep (the old stability.py was removed) with: a finer grid in the
[1e-3, 1e-2] region, a vertical line at the default/production step (jacobian_eps = 1e-6),
and a zoom row over [1e-3, 1e-2]. For each peptide we compare the finite-difference Jacobian
at each step eps against the exact (autograd) Jacobian along the whole MUTANG pipeline:
relative Frobenius error, top-k singular-value error, top-k subspace overlap, and the
single-position mutation-set Jaccard.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, save

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)
from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.local_enumeration.mutation.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)

BENCHMARK_PEPTIDES = {
    "middle-1": "FLYKWWIRIGRLKL",
    "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN",
    "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}
ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
MAXLEN = 25
DEFAULT_EPS = 5e-2  # production / default jacobian_eps -> marked with a vertical line
D_C, T_C, KTOP = 1e-3, 1e-6, 10
N_SAMPLE = 60
# decade grid + a fine sweep inside the [1e-3, 1e-2] sweet-spot region
EPS_GRID = sorted(
    {
        1e-1,
        3e-2,
        1e-2,
        7e-3,
        5e-3,
        3e-3,
        2e-3,
        1e-3,
        3e-4,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
        1e-8,
    },
    reverse=True,
)
ZOOM = [e for e in EPS_GRID if 1e-3 <= e <= 1e-2]


def jac_strict(hy, z):
    dec = lambda x: hy.decoder_forward(x, softmax=True, flatten=True)
    J = torch.autograd.functional.jacobian(dec, z, vectorize=True)  # (1, M, 1, Z)
    return J[0, :, 0, :].detach().cpu().numpy()


def jac_approx(hy, z, eps):
    dec = lambda x: hy.decoder_forward(x, softmax=True, flatten=True)
    return (
        decoder_jacobian(dec, z, "approx", {"jacobian_eps": eps})[0]
        .detach()
        .cpu()
        .numpy()
    )


def single_pos_set(pep, S, U):
    mutang = MutationEnumerationInTangentSpace(
        max_len=MAXLEN, direction_significance_threshold=D_C, token_threshold=T_C
    )
    muts = mutang.get_mutations_from_s_u(S, U)
    out = set()
    for p, aas in muts.items():
        if p >= len(pep):
            continue
        for a in aas:
            cand = pep[:p] + ALPHABET[a] + pep[p + 1 :]
            if cand != pep:
                out.add(cand)
    return out


def subspace_overlap(Ua, Ue, k):
    k = min(k, Ua.shape[1], Ue.shape[1])
    cos = np.linalg.svd(Ua[:, :k].T @ Ue[:, :k], compute_uv=False)
    return float(np.clip(cos, 0, 1).mean())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def eps_scan(hy, peptides):
    rows = []
    for name, pep in peptides.items():
        z = hy.encode_peptides([pep])
        Je = jac_strict(hy, z)
        Ue, Se, _ = np.linalg.svd(Je, full_matrices=False)
        exact = single_pos_set(pep, Se, Ue)
        nrm_e = np.linalg.norm(Je)
        for eps in EPS_GRID:
            Ja = jac_approx(hy, z, eps)
            Ua, Sa, _ = np.linalg.svd(Ja, full_matrices=False)
            approx = single_pos_set(pep, Sa, Ua)
            rows.append(
                {
                    "peptide": name,
                    "eps": eps,
                    "jac_rel_error": float(np.linalg.norm(Ja - Je) / nrm_e),
                    "sv_rel_error": float(
                        np.linalg.norm(Sa[:KTOP] - Se[:KTOP])
                        / np.linalg.norm(Se[:KTOP])
                    ),
                    "subspace_overlap": subspace_overlap(Ua, Ue, KTOP),
                    "mut_jaccard": jaccard(approx, exact),
                }
            )
    return pd.DataFrame(rows)


def main():
    hy = HydrAMPEncoderDecoder(
        jacobian_mode="approx",
        jacobian_eps=1e-6,
        field_eps=1e-6,
        device=torch.device("cpu"),
    )
    hy.eval()
    print(f"HydrAMP loaded | latent {hy.latent_dim} | ambient {hy.ambient_dim}")
    pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")[
        "sequence"
    ]
    pool = pool.dropna().astype(str)
    pool = pool[
        (pool.str.len() <= MAXLEN)
        & (pool.str.len() > 3)
        & pool.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))
    ].drop_duplicates()
    sample = {
        f"s{i}": s
        for i, s in enumerate(pool.sample(n=N_SAMPLE, random_state=0).tolist())
    }

    samp_csv = CACHE / "_thesis_rq2_eps.csv"
    bench_csv = CACHE / "_thesis_rq2_eps_bench.csv"
    samp = pd.read_csv(samp_csv) if samp_csv.exists() else None
    bench = pd.read_csv(bench_csv) if bench_csv.exists() else None
    if samp is None:
        print("eps scan on sample ...", flush=True)
        samp = eps_scan(hy, sample)
        samp.to_csv(samp_csv, index=False)
    if bench is None:
        print("eps scan on benchmarks ...", flush=True)
        bench = eps_scan(hy, BENCHMARK_PEPTIDES)
        bench.to_csv(bench_csv, index=False)

    metrics = [
        ("jac_rel_error", "Jacobian rel. Frobenius error", True),
        ("sv_rel_error", "top-10 singular-value rel. error", True),
        ("subspace_overlap", "top-10 subspace overlap (1=identical)", False),
        ("mut_jaccard", "mutation-set Jaccard (approx vs exact)", False),
    ]
    metric_cols = [c for c, _, _ in metrics]
    agg = samp.groupby("eps")[metric_cols].agg(["mean", "std"])
    n = samp["peptide"].nunique()
    best = samp.groupby("eps")["mut_jaccard"].mean().idxmax()
    print(f"eps maximising mean mutation-set Jaccard: {best:g}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for j, (col, title, logy) in enumerate(metrics):
        # ---- top row: full eps range, with default-eps vertical line ----
        ax = axes[0, j]
        for _, g in bench.groupby("peptide"):
            ax.plot(g["eps"], g[col], color="0.78", lw=1, alpha=0.7)
        m, s = agg[(col, "mean")], agg[(col, "std")]
        ax.plot(
            m.index,
            m.values,
            color="C3",
            marker="o",
            lw=2,
            label=f"sample mean (n={n})",
        )
        ax.fill_between(m.index, (m - s).values, (m + s).values, color="C3", alpha=0.2)
        ax.axvline(
            DEFAULT_EPS,
            color="C0",
            lw=1.6,
            ls="--",
            label=r"default $\varepsilon=10^{-6}$",
        )
        ax.axvspan(1e-3, 1e-2, color="green", alpha=0.07)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"jacobian\_eps  $\varepsilon$")
        ax.set_title(title, fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, loc="upper center")
        # ---- bottom row: zoom into [1e-3, 1e-2] ----
        axz = axes[1, j]
        bz = bench[bench["eps"].between(1e-3, 1e-2)]
        for _, g in bz.groupby("peptide"):
            axz.plot(g["eps"], g[col], color="0.78", lw=1, alpha=0.7, marker=".")
        mz = m[(m.index >= 1e-3) & (m.index <= 1e-2)]
        sz = s[(s.index >= 1e-3) & (s.index <= 1e-2)]
        axz.plot(mz.index, mz.values, color="C2", marker="o", lw=2)
        axz.fill_between(
            mz.index, (mz - sz).values, (mz + sz).values, color="C2", alpha=0.2
        )
        axz.set_xscale("log")
        if logy:
            axz.set_yscale("log")
        axz.invert_xaxis()
        axz.set_xlabel(r"$\varepsilon$ (zoom $10^{-3}$--$10^{-2}$)")
        axz.set_title(f"{title} — zoom", fontsize=9)
    axes[0, 0].set_ylabel("full range", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel(r"zoom $[10^{-3},10^{-2}]$", fontsize=11, fontweight="bold")
    fig.suptitle(
        r"(c) Approx vs exact Jacobian across $\varepsilon$ "
        r"(grey = 6 benchmarks, red/green = sample mean$\pm$std; "
        r"blue dashed = default $\varepsilon=10^{-6}$; shaded = sweet-spot region)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "rq2_eps_sensitivity.pdf")

    print("\nmean metrics by eps:")
    print(samp.groupby("eps")[[c for c, _, _ in metrics]].mean().to_string())


if __name__ == "__main__":
    main()
