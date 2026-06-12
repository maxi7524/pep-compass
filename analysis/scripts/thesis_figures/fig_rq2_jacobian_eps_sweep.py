"""RQ2(c) -- full epsilon sweep for fig:rq2-eps and tab:rq2-eps.

Replaces the original 60-peptide Veltri-positive subsample with a larger draw from
the full all_in collection (default N=500, seed=0). The six SAASBO benchmark seeds
are still overlaid as grey per-peptide curves. Writes:
    rq2_eps_sensitivity.pdf
    results/data/all_in/_cache/_thesis_rq2_eps.csv
    results/data/all_in/_cache/_thesis_rq2_eps_table.csv   (mean row per eps, for LaTeX)

Env:
    RQ2_EPS_N=500       sample size (peptides from peptides_apex.csv)
    RQ2_EPS_SEED=0
    RQ2_DEVICE=cuda|cpu
"""
from __future__ import annotations

import importlib.util as _ilu
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.enabled = False

from _common import PROJECT_ROOT, CACHE, save

_st_path = PROJECT_ROOT / "analysis" / "scripts" / "mutang" / "stability.py"
_spec = _ilu.spec_from_file_location("rq2_stability", _st_path)
st = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(st)  # type: ignore[union-attr]

PROD_KAPPA = 1e-3
PROD_TMUT = 1e-6
PROD_EPS = 5e-2
TOP_K = 10
EPS_GRID = [1e-1, PROD_EPS, 1e-2, 3e-3, 2e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
DATA_DIR = PROJECT_ROOT / "results" / "data" / "all_in"
APEX_CSV = DATA_DIR / "peptides_apex.csv"
PARQUET_FALLBACKS = [
    DATA_DIR / "_cache" / "parents_ext_veltri_negative_veltri_positive_dbaasp_clean_mic_data.parquet",
    DATA_DIR / "_cache" / "parents_hydramp_veltri_positive.parquet",
]

METRICS = [
    ("jac_rel_error", "Jacobian rel. Frobenius error", True),
    ("sv_rel_error", "top-10 singular-value rel. error", True),
    ("subspace_overlap", "top-10 subspace overlap (1 = identical)", False),
    ("mut_jaccard", "mutation-set Jaccard (approx vs exact)", False),
]


def pick_device(name: str | None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_peptides(n: int, seed: int) -> tuple[list[str], str]:
    if APEX_CSV.exists():
        df = pd.read_csv(APEX_CSV, usecols=["sequence"])
        source = f"csv:{APEX_CSV.name}"
    else:
        df = None
        source = ""
        for p in PARQUET_FALLBACKS:
            if p.exists():
                df = pd.read_parquet(p, columns=["sequence"])
                source = f"parquet:{p.name}"
                break
        if df is None:
            raise FileNotFoundError("No peptide source found (peptides_apex.csv or cache parquet).")
    seqs = df["sequence"].dropna().astype(str).str.strip()
    valid = seqs[(seqs.str.len() <= 25) & (seqs.str.len() > 0)
                 & seqs.apply(lambda s: set(s) <= st.STANDARD_AA)].drop_duplicates()
    if n > len(valid):
        print(f"  WARN: requested N={n} but source has only {len(valid)} valid sequences.")
    return valid.sample(n=min(n, len(valid)), random_state=seed).tolist(), source


def fmt_eps(eps: float) -> str:
    if abs(eps - PROD_EPS) < 1e-12:
        return r"$5\times10^{-2}$ (prod.)"
    if eps >= 0.01:
        return f"${eps:g}$"
    exp = int(round(np.log10(eps)))
    mant = eps / 10 ** exp
    if abs(mant - 1.0) < 1e-9:
        return f"$10^{{{exp}}}$"
    if abs(mant - 2.0) < 1e-9:
        return f"$2\\times10^{{{exp}}}$"
    if abs(mant - 3.0) < 1e-9:
        return f"$3\\times10^{{{exp}}}$"
    return f"${eps:g}$"


def main():
    n = int(os.environ.get("RQ2_EPS_N", 500))
    seed = int(os.environ.get("RQ2_EPS_SEED", 0))
    device = pick_device(os.environ.get("RQ2_DEVICE"))
    print(f"RQ2 eps sweep | N={n} seed={seed} device={device} grid={len(EPS_GRID)} steps")

    sample, source = sample_peptides(n, seed)
    benchmarks = st.BENCHMARK_PEPTIDES
    print(f"sampled {len(sample)} peptides from {source}; {len(benchmarks)} benchmark overlays")

    model = st.load_model(device=device)
    eps_bench = st.eps_sensitivity_scan(benchmarks, model, EPS_GRID, PROD_KAPPA, PROD_TMUT, k=TOP_K)
    print(f"running eps sweep: {len(sample)} x {len(EPS_GRID)} ...")
    eps_sample = st.eps_sensitivity_scan(sample, model, EPS_GRID, PROD_KAPPA, PROD_TMUT, k=TOP_K)
    eps_sample.to_csv(CACHE / "_thesis_rq2_eps.csv", index=False)

    cols = [m[0] for m in METRICS]
    means = eps_sample.groupby("eps")[cols].mean()
    stds = eps_sample.groupby("eps")[cols].std()
    table = means.copy()
    table.to_csv(CACHE / "_thesis_rq2_eps_table.csv")
    print(f"wrote {CACHE / '_thesis_rq2_eps.csv'}")
    print(f"wrote {CACHE / '_thesis_rq2_eps_table.csv'}")

    # ---- figure ----
    agg = eps_sample.groupby("eps")[cols].agg(["mean", "std"])
    n_samp = eps_sample.peptide.nunique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title, logy) in zip(axes.ravel(), METRICS):
        for _, g in eps_bench.groupby("peptide"):
            ax.plot(g["eps"], g[col], color="0.75", lw=1, alpha=0.7)
        m, s = agg[col]["mean"], agg[col]["std"]
        ax.plot(m.index, m.values, color="C3", marker="o", lw=2,
                label=f"sample mean (n={n_samp})")
        ax.fill_between(m.index, (m - s).values, (m + s).values, color="C3", alpha=0.2)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.axvline(PROD_EPS, color="C0", ls="--", lw=1.2,
                   label=f"production default $\\varepsilon$={PROD_EPS:g}")
        ax.axvspan(1e-3, 1e-2, color="0.85", alpha=0.5, zorder=0)
        ax.set_xlabel(r"finite-difference step $\varepsilon$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(r"(c) Approx vs exact Jacobian across $\varepsilon$  (gray = 6 benchmarks, "
                 rf"red = sample mean$\pm$std, $n={n_samp}$; $\kappa$={PROD_KAPPA:g}, "
                 rf"$\theta_{{mut}}$={PROD_TMUT:g})")
    fig.tight_layout()
    save(fig, "rq2_eps_sensitivity.pdf")

    # ---- LaTeX table rows ----
    print("\n=== tab:rq2-eps (mean over sample) ===")
    best_frob = means["jac_rel_error"].idxmin()
    best_jacc = means["mut_jaccard"].idxmax()
    best_overlap = means["subspace_overlap"].idxmax()
    for eps in EPS_GRID:
        row = means.loc[eps]
        cells = []
        for c in cols:
            v = row[c]
            bold = (
                (c == "jac_rel_error" and eps == best_frob)
                or (c == "sv_rel_error" and eps == means["sv_rel_error"].idxmin())
                or (c == "subspace_overlap" and eps == best_overlap)
                or (c == "mut_jaccard" and eps == best_jacc)
            )
            cells.append(f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}")
        print(f"    {fmt_eps(eps):24s} & {' & '.join(cells)} \\\\")

    prod = means.loc[PROD_EPS]
    min_row = means.loc[best_frob]
    print(f"\nproduction mut_jaccard mean = {prod['mut_jaccard']:.3f}")
    print(f"min-Frob eps = {best_frob:g}, Frob = {min_row['jac_rel_error']:.3f}, "
          f"Jaccard = {min_row['mut_jaccard']:.3f}")


if __name__ == "__main__":
    main()
