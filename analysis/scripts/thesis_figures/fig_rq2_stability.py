"""RQ2 -- structural weaknesses of MUTANG (a/b/c).

Uses analysis/scripts/mutang/stability.py and the bundled HydrAMP weights. Notation matches
chapters/models.tex: the direction-significance threshold is kappa (= the implementation's
direction_significance_threshold, the W_z^kappa threshold) and the token/sensitivity
threshold is theta_mut (= token_threshold). Writes vector PDFs:
    rq2_explosion.pdf          -- (a) joint Cartesian-product explosion
    rq2_threshold_heatmaps.pdf -- (b) per-benchmark #mutants over the kappa x theta_mut grid
    rq2_cv.pdf                 -- (b) cross-peptide CV of #mutants (892 and ~5000 peptides)
    rq2_jaccard_churn.pdf      -- (b) churn: theta_mut-sweep (flat) vs kappa-sweep (churns)
    rq2_eps_sensitivity.pdf    -- (c) approx-vs-exact Jacobian U-curve in eps

The ~5000-peptide instability sample is drawn broadly from the whole all_in collection
(peptides_apex.csv), well beyond the original 100/892 peptides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import PROJECT_ROOT, CACHE, save

sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.mutang import stability as st  # noqa: E402

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
APEX = PROJECT_ROOT / "results" / "data" / "all_in" / "peptides_apex.csv"
KAPPA_SYM = r"$\kappa$"
TMUT_SYM = r"$\theta_{\mathrm{mut}}$"

# PepCompass production hyperparameters (paper appendix, Table of LE-BO settings).
#   * PROD_KAPPA is the *singular-value* threshold used by the implementation
#     (direction_significance_threshold). The paper states kappa_MUTANG = 1e-6 as a bound on
#     the eigenvalues of the pullback metric G = J^T J, i.e. on the squared singular values;
#     the implementation thresholds the singular values directly, so the equivalent value is
#     sqrt(1e-6) = 1e-3.
#   * PROD_TMUT is the token / mutation sensitivity threshold theta_mut = 1e-6.
#   * PROD_EPS is the finite-difference step of the decoder-Jacobian approximation = 5e-2.
PROD_KAPPA = 1e-3
PROD_TMUT = 1e-6
PROD_EPS = 5e-2


def sample_from_parquet(parquet_path, n, seed=0, max_len=25):
    df = pd.read_parquet(parquet_path)
    seqs = df["sequence"].dropna().astype(str).str.strip()
    valid = seqs[(seqs.str.len() <= max_len) & (seqs.str.len() > 0)
                 & seqs.apply(lambda s: set(s) <= STANDARD_AA)].drop_duplicates()
    return valid.sample(n=min(n, len(valid)), random_state=seed).tolist()


def sample_from_apex(n, seed=0, max_len=25):
    """Draw n peptides broadly from the whole all_in collection."""
    df = pd.read_csv(APEX, usecols=["sequence"])
    seqs = df["sequence"].dropna().astype(str).str.strip()
    valid = seqs[(seqs.str.len() <= max_len) & (seqs.str.len() > 0)
                 & seqs.apply(lambda s: set(s) <= STANDARD_AA)].drop_duplicates()
    return valid.sample(n=min(n, len(valid)), random_state=seed).tolist()


def main():
    model = st.load_model(device="cpu")
    print(f"HydrAMP loaded | latent {model.latent_dim} | ambient {model.ambient_dim}")
    cache = {}
    benchmarks = st.BENCHMARK_PEPTIDES
    pos_parquet = CACHE / "parents_hydramp_veltri_positive.parquet"
    sample_full = sample_from_parquet(pos_parquet, 1000, seed=0)
    sample_big = sample_from_apex(5000, seed=0)
    print(f"benchmarks: {len(benchmarks)} | sample_full: {len(sample_full)} | "
          f"sample_big: {len(sample_big)}")

    # ---------------- (a) combinatorial explosion ----------------
    # production-centered sweep: theta_mut steps by x10 down to the production 1e-6, at the
    # production kappa=1e-3. The loosest point is the production setting itself.
    explosion_thresholds = [(PROD_KAPPA, 1e-2), (PROD_KAPPA, 1e-3), (PROD_KAPPA, 1e-4),
                            (PROD_KAPPA, 1e-5), (PROD_KAPPA, PROD_TMUT)]
    expl = st.explosion_scan(benchmarks, model, explosion_thresholds, mode="approx",
                             eps=PROD_EPS, cache=cache)
    expl.to_csv(CACHE / "_thesis_rq2_explosion.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, g in expl.groupby("peptide"):
        ax.plot(range(len(g)), g["joint_count"].values, marker="o", label=name)
    ax.set_yscale("log")
    ax.set_xticks(range(len(explosion_thresholds)))
    ax.set_xticklabels([f"$\\theta_{{mut}}$={t:g}" for _, t in explosion_thresholds], fontsize=8)
    ax.set_xlabel(r"token threshold $\theta_{mut}$ (at production $\kappa$=1e-3)")
    ax.set_ylabel("joint mutants (Cartesian product)")
    ax.set_title(r"(a) Joint enumeration explodes as $\theta_{mut}$ loosens toward production")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "rq2_explosion.pdf")
    prod = expl[(expl.d_thresh == PROD_KAPPA) & (expl.t_thresh == PROD_TMUT)]
    ratio = (prod["joint_count"] / prod["single_count"].clip(lower=1)).max()
    print("PRODUCTION explosion (kappa=1e-3, theta_mut=1e-6):")
    print(prod[["peptide", "len", "n_positions", "single_count", "joint_count"]].to_string(index=False))
    print(f"max joint/single ratio at production: {ratio:,.1f}")

    # ---------------- (b) threshold instability ----------------
    d_grid = [1e-2, 1e-3, 1e-4, 1e-5]            # kappa (1e-3 = production)
    t_grid = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]      # theta_mut, production-centered (1e-6) in x10 steps
    bench_scan = st.threshold_grid_scan(benchmarks, model, d_grid, t_grid, mode="approx",
                                        eps=PROD_EPS, cache=cache)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, g) in zip(axes.ravel(), bench_scan.groupby("peptide")):
        pivot = g.pivot(index="d_thresh", columns="t_thresh", values="n_mutants")
        ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t:g}" for t in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{d:g}" for d in pivot.index], fontsize=7)
        ax.set_xlabel(TMUT_SYM, fontsize=9)
        ax.set_ylabel(KAPPA_SYM, fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center",
                        color="w", fontsize=7)
    fig.suptitle(r"(b) #single-position mutants over the $\kappa\times\theta_{mut}$ grid "
                 "-- per benchmark peptide")
    fig.tight_layout()
    save(fig, "rq2_threshold_heatmaps.pdf")

    scan_full = st.threshold_grid_scan(sample_full, model, d_grid, t_grid, mode="approx", eps=PROD_EPS)
    print(f"running threshold grid on {len(sample_big)} peptides (this is the slow step) ...")
    scan_big = st.threshold_grid_scan(sample_big, model, d_grid, t_grid, mode="approx", eps=PROD_EPS)
    frames = {f"Veltri-positive sample (n={len(sample_full)})": st.cross_peptide_instability(scan_full),
              f"broad all_in sample (n={len(sample_big)})": st.cross_peptide_instability(scan_big)}
    cvbig = st.cross_peptide_instability(scan_big).sort_values("cv", ascending=False)
    cvbig.to_csv(CACHE / "_thesis_rq2_cv_big.csv", index=False)
    print(f"Most unstable cells (n={len(sample_big)} broad sample):")
    print(cvbig.head(6).to_string(index=False))
    print(f"CV over broad sample: min/median/max = "
          f"{cvbig.cv.min():.3f}/{cvbig.cv.median():.3f}/{cvbig.cv.max():.3f}")

    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 4.4), squeeze=False)
    for ax, (label, cv) in zip(axes.ravel(), frames.items()):
        pivot = cv.pivot(index="d_thresh", columns="t_thresh", values="cv")
        im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
        ax.set_title(f"CV of #mutants across peptides -- {label}", fontsize=9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{t:g}" for t in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{d:g}" for d in pivot.index], fontsize=7)
        ax.set_xlabel(TMUT_SYM)
        ax.set_ylabel(KAPPA_SYM)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "-", ha="center",
                        va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "rq2_cv.pdf")

    # ---- churn: theta_mut-sweep (flat) vs kappa-sweep (churns) ----
    tmut_path = [(PROD_KAPPA, t) for t in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]  # x10 steps incl prod 1e-6
    kappa_path = [(d, PROD_TMUT) for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]]  # x10 steps incl prod 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    all_t, all_k = [], []
    for name, seq in benchmarks.items():
        U, S = st.compute_svd(model, seq, mode="approx", eps=PROD_EPS, cache=cache)
        c_t = st.set_sensitivity_curve(seq, S, U, tmut_path)
        c_k = st.set_sensitivity_curve(seq, S, U, kappa_path)
        all_t += c_t
        all_k += c_k
        axes[0].plot(range(len(c_t)), c_t, marker="o", label=name)
        axes[1].plot(range(len(c_k)), c_k, marker="o", label=name)
    axes[0].set_xticks(range(len(tmut_path) - 1))
    axes[0].set_xticklabels([f"{tmut_path[i][1]:g}$\\to${tmut_path[i+1][1]:g}"
                             for i in range(len(tmut_path) - 1)], fontsize=7, rotation=30)
    axes[0].set_title(r"$\theta_{mut}$-sweep ($\kappa$=1e-3 fixed, prod.) -- nearly no churn")
    axes[0].set_xlabel(r"$\theta_{mut}$ step")
    axes[0].set_ylabel("Jaccard between adjacent settings")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xticks(range(len(kappa_path) - 1))
    axes[1].set_xticklabels([f"{kappa_path[i][0]:g}$\\to${kappa_path[i+1][0]:g}"
                             for i in range(len(kappa_path) - 1)], fontsize=7, rotation=30)
    axes[1].set_title(r"$\kappa$-sweep ($\theta_{mut}$=1e-6 fixed, prod.) -- set churns")
    axes[1].set_xlabel(r"$\kappa$ step")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7)
    fig.suptitle("(b) Mutation-set churn: the token threshold barely changes the set, "
                 r"the direction threshold $\kappa$ drives it")
    fig.tight_layout()
    save(fig, "rq2_jaccard_churn.pdf")
    print(f"mean adjacent Jaccard: theta_mut-sweep={np.mean(all_t):.3f}  "
          f"kappa-sweep={np.mean(all_k):.3f}  (lower = more churn)")

    # ---------------- (c) Jacobian eps sensitivity ----------------
    eps_grid = [1e-1, PROD_EPS, 1e-2, 3e-3, 2e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    D_C, T_C = PROD_KAPPA, PROD_TMUT
    eps_bench = st.eps_sensitivity_scan(benchmarks, model, eps_grid, D_C, T_C, k=10)
    eps_sample_peptides = sample_full[:60]  # matches the "60 sampled peptides" reported in the thesis
    print(f"(c) eps sweep over {len(eps_sample_peptides)} sampled peptides x {len(eps_grid)} eps ...")
    eps_sample = st.eps_sensitivity_scan(eps_sample_peptides, model, eps_grid, D_C, T_C, k=10)
    eps_sample.to_csv(CACHE / "_thesis_rq2_eps.csv", index=False)

    metrics = [("jac_rel_error", "Jacobian rel. Frobenius error", True),
               ("sv_rel_error", "top-10 singular-value rel. error", True),
               ("subspace_overlap", "top-10 subspace overlap (1 = identical)", False),
               ("mut_jaccard", "mutation-set Jaccard (approx vs exact)", False)]
    agg = eps_sample.groupby("eps")[[m[0] for m in metrics]].agg(["mean", "std"])
    n_samp = eps_sample.peptide.nunique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title, logy) in zip(axes.ravel(), metrics):
        for _, g in eps_bench.groupby("peptide"):
            ax.plot(g["eps"], g[col], color="0.75", lw=1, alpha=0.7)
        m, s = agg[col]["mean"], agg[col]["std"]
        ax.plot(m.index, m.values, color="C3", marker="o", lw=2, label=f"sample mean (n={n_samp})")
        ax.fill_between(m.index, (m - s).values, (m + s).values, color="C3", alpha=0.2)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()
        # production default finite-difference step (PepCompass paper appendix)
        ax.axvline(PROD_EPS, color="C0", ls="--", lw=1.2,
                   label=f"production default $\\varepsilon$={PROD_EPS:g}")
        ax.axvspan(1e-3, 1e-2, color="0.85", alpha=0.5, zorder=0)  # downstream sweet spot
        ax.set_xlabel("jacobian_eps")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(r"(c) Approx vs exact Jacobian across eps  (gray = 6 benchmarks, "
                 r"red = sample mean$\pm$std; $\kappa$=" + f"{D_C:g}" + r", $\theta_{mut}$=" + f"{T_C:g})")
    fig.tight_layout()
    save(fig, "rq2_eps_sensitivity.pdf")
    best = eps_sample.groupby("eps")["mut_jaccard"].mean().idxmax()
    worst = eps_sample.groupby("eps")["jac_rel_error"].mean().idxmax()
    print(f"eps maximizing mean mutation-set Jaccard vs exact: {best:g}")
    print(f"eps with worst mean Jacobian error: {worst:g}")
    print("mean metrics by eps:")
    print(eps_sample.groupby("eps")[[m[0] for m in metrics]].mean().to_string())


if __name__ == "__main__":
    main()
