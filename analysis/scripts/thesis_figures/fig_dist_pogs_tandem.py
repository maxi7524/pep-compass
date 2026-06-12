"""How TANDEM-A aligns with the PoGS geodesic distance (local; from the cached RQ3 candidates).

Repeats the RQ3 feasibility analysis with feasibility = -d_pogs (PoGS, lambda=0; ambient chord
length of the ADAM-optimised path, no metric G -- see _pogs.py / _exp_pogs_distance_cache.py), to
see whether the TANDEM-A pairwise-coherence potential ranks candidates by PoGS-geodesic proximity:
  (a) top-p% selection: mean d_pogs of kept set relative to the full set, per peptide;
  (b) fixed-mutation-count median per-peptide Spearman(potential, -d_pogs).
Reports TANDEM-A in BOTH encodings -- one-hot (whitened) and diff (parent->target) -- with the
existing graph geodesic -d_geo alongside for reference. The diff encoding is constant within
single mutants, so it is reported only for n_mut >= 2. Saves rq5_tandem_pogs.pdf.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _common import CACHE, THESIS_FIG

# RECON=1 restricts the analysis to candidates that reconstruct through HydrAMP
# (argmax(Dec(enc(seq))) == seq); uses the recon-flagged cache and writes a separate figure.
RECON = bool(int(os.environ.get("RECON", "0")))
if RECON:
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs_recon.parquet").reset_index(drop=True)
    df = df[df["reconstructs"]].reset_index(drop=True)
    OUT_NAME = "rq5_tandem_pogs_recon.pdf"
    TITLE_SUFFIX = ", reconstructing mutants only"
else:
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs.parquet").reset_index(drop=True)
    OUT_NAME = "rq5_tandem_pogs.pdf"
    TITLE_SUFFIX = ""
# Only the parent->target difference encoding of TANDEM-A is shown in the main text (the one-hot
# encoding lives in the appendix); the encoding name is not surfaced in the curve label.
POTS = [("TANDEM-A", "score_A_diff", "C2")]
P_MASS = np.round(np.arange(0.025, 1.0 + 1e-9, 0.025), 3)   # dense grid, step 0.025
NMUTS = [1, 2, 3, 4]


def selection_curve(score_col, dist_col="dist_pogs"):
    """Absolute mean PoGS distance of the candidates holding the top-p fraction of the potential
    *probability mass*, median over peptides, for each p on the dense grid.

    The potential is turned into a per-peptide distribution by a soft-max over candidates,
    w_i = softmax(score_i); candidates are taken in decreasing w until the cumulative mass reaches
    p (nucleus / top-p selection). PoGS is an absolute distance (ambient chord length), so the
    kept-set mean is reported in absolute units (no MUTANG=1 normalisation); p=1 is the full set."""
    out = []
    for p in P_MASS:
        vals = []
        for _, g in df.groupby("pep"):
            if len(g) < 4 or g[score_col].nunique() < 2:
                continue
            s = g.sort_values(score_col, ascending=False)
            w = np.exp(s[score_col].to_numpy() - s[score_col].max())
            w = w / w.sum()
            k = int(np.searchsorted(np.cumsum(w), p) + 1)   # smallest set with cum mass >= p
            k = min(max(k, 1), len(s))
            vals.append(s[dist_col].to_numpy()[:k].mean())
        out.append(np.median(vals) if vals else np.nan)
    return np.array(out)


P_TABLE = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]   # top-p mass thresholds for the distribution table


def _tailmean(a, frac, top):
    """Mean of the worst (top=True -> largest) or best (top=False -> smallest) ``frac`` of ``a``."""
    m = max(1, int(np.ceil(frac * len(a))))
    s = np.sort(a)
    return s[-m:].mean() if top else s[:m].mean()


def threshold_table(score_col, dist_col="dist_pogs"):
    """At each top-p mass threshold, the kept-set distance distribution, median over peptides:
    mean, median, bottom-5% mean (closest), top-5% mean (farthest tail)."""
    rows = []
    for p in P_TABLE:
        mean_, med_, lo5_, hi5_, ret_ = [], [], [], [], []
        for _, g in df.groupby("pep"):
            if len(g) < 4 or g[score_col].nunique() < 2:
                continue
            s = g.sort_values(score_col, ascending=False)
            w = np.exp(s[score_col].to_numpy() - s[score_col].max()); w /= w.sum()
            k = int(np.searchsorted(np.cumsum(w), p) + 1); k = min(max(k, 1), len(s))
            d = s[dist_col].to_numpy()[:k]
            mean_.append(d.mean()); med_.append(np.median(d))
            lo5_.append(_tailmean(d, 0.05, top=False)); hi5_.append(_tailmean(d, 0.05, top=True))
            ret_.append(k / len(s))
        rows.append(dict(p=p, retention=np.median(ret_), mean=np.median(mean_),
                         median=np.median(med_), bottom5=np.median(lo5_), top5=np.median(hi5_)))
    return pd.DataFrame(rows)


def fixed_count_spearman(score_col, nmut, dist_col="dist_pogs"):
    sub = df[df.n_mut == nmut]
    rhos = []
    for _, g in sub.groupby("pep"):
        if len(g) < 4 or g[score_col].nunique() < 2 or g[dist_col].nunique() < 2:
            continue
        r, _ = spearmanr(g[score_col], -g[dist_col])   # feasibility = -distance
        if np.isfinite(r):
            rhos.append(r)
    return np.median(rhos) if rhos else np.nan


def main():
    print(f"candidates={len(df):,} peptides={df.pep.nunique()}")
    print("\n--- fixed-count median Spearman(potential, -distance), distance in {pogs, geo} ---")
    for name, col, _ in POTS:
        for dist_col in ("dist_pogs", "dist_geo"):
            vals = [fixed_count_spearman(col, n, dist_col) for n in NMUTS]
            print(f"  {name:18s} vs -{dist_col:9s}: "
                  + ", ".join(f"n={n}:{v:+.3f}" if np.isfinite(v) else f"n={n}:  -  "
                              for n, v in zip(NMUTS, vals)))

    # single panel: absolute PoGS distance of the top-p probability-mass set vs p, with the
    # optimal operating point (minimum distance) marked for each potential.
    print("\n--- kept-set distance distribution vs top-p mass (median over peptides, absolute d_PoGS) ---")
    for name, col, _ in POTS:
        t = threshold_table(col)
        t.insert(0, "potential", name)
        t.to_csv(CACHE / f"_thesis_pogs_tandem_dist_{col}.csv", index=False)
        print(f"  {name}")
        print(f"    {'top-p':>6} {'retain':>7} {'mean':>7} {'median':>7} {'bot5%':>7} {'top5%':>7}")
        for _, r in t.iterrows():
            print(f"    {r.p:>6.2f} {r.retention:>7.2f} {r['mean']:>7.3f} {r['median']:>7.3f} "
                  f"{r.bottom5:>7.3f} {r.top5:>7.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    full = {}
    print("\n--- top-p (softmax mass) selection: optimal operating point ---")
    for name, col, c in POTS:
        curve = selection_curve(col)
        ax.plot(P_MASS, curve, marker="o", ms=3, color=c, label=name)
        j = int(np.nanargmin(curve))
        p_opt, d_opt, d_full = P_MASS[j], curve[j], curve[-1]
        full[col] = (p_opt, d_opt, d_full)
        ax.scatter([p_opt], [d_opt], s=80, facecolors="none", edgecolors=c, linewidths=1.8, zorder=5)
        ax.annotate(f"p*={p_opt:.3f}\n$d$={d_opt:.2f}", (p_opt, d_opt),
                    textcoords="offset points", xytext=(6, -2), fontsize=8, color=c)
        red = 100 * (1 - d_opt / d_full)
        print(f"  {name:18s}: p*={p_opt:.3f}  d*={d_opt:.3f}  full(p=1)={d_full:.3f}  "
              f"reduction={red:+.1f}%")
    ax.set_xlabel("keep top-$p$ of potential probability mass (soft-max)")
    ax.set_ylabel("median (over peptides) mean $d_{\\mathrm{PoGS}}$ of kept set")
    ax.set_title(f"TANDEM-A top-$p$ selection vs PoGS distance "
                 f"({df.pep.nunique()} peptides{TITLE_SUFFIX})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = THESIS_FIG / OUT_NAME
    fig.savefig(out); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
