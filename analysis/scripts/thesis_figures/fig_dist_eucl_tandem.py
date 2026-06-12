"""How TANDEM does on the Euclidean latent distance (local; from the cached RQ3 candidates).

Repeats the RQ3 feasibility analysis with feasibility = -d_eucl instead of -d_geo, to see
whether the TANDEM pairwise-coherence potential ranks candidates by Euclidean proximity:
  (a) top-p% selection: mean d_eucl of kept set relative to the full set, per peptide;
  (b) fixed-mutation-count median per-peptide Spearman(potential, -d_eucl).
Saves rq5_tandem_eucl.pdf into the thesis figures dir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _common import CACHE, THESIS_FIG

df = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
POTS = [("TANDEM-A (whitened)", "score_A_onehot", "C0"),
        ("TANDEM-B (pullback)", "score_B_onehot", "C1")]
P_LEVELS = [100, 90, 80, 70, 50, 25]
NMUTS = [1, 2, 3, 4]


def selection_curve(score_col, dist_col="dist_eucl"):
    """Mean dist of top-p% (by score) relative to full set, median over peptides, per p%."""
    out = []
    for p in P_LEVELS:
        rels = []
        for _, g in df.groupby("pep"):
            if len(g) < 4 or g[dist_col].mean() == 0:
                continue
            k = max(1, int(round(p / 100 * len(g))))
            kept = g.nlargest(k, score_col)
            rels.append(kept[dist_col].mean() / g[dist_col].mean())
        out.append(np.median(rels))
    return out


def fixed_count_spearman(score_col, nmut, dist_col="dist_eucl"):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) selection curve on Euclidean
    for name, col, c in POTS:
        ax1.plot(P_LEVELS, selection_curve(col), marker="o", color=c, label=name)
    ax1.axhline(1.0, color="0.6", ls="--", lw=1)
    ax1.invert_xaxis()
    ax1.set_xlabel("selection level: keep top-$p\\%$ by potential")
    ax1.set_ylabel("mean $d_{\\mathrm{eucl}}$ of kept / full set")
    ax1.set_title("(a) TANDEM selection vs Euclidean proximity", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.8, 1.2)
    ax1.legend(fontsize=9)

    # (b) fixed-count Spearman(potential, -d_eucl)
    x = np.arange(len(NMUTS)); w = 0.35
    for i, (name, col, c) in enumerate(POTS):
        vals = [fixed_count_spearman(col, n) for n in NMUTS]
        ax2.bar(x + (i - 0.5) * w, vals, w, color=c, label=name)
        print(f"  {name}: fixed-count Spearman(pot,-d_eucl) by n_mut = "
              + ", ".join(f"{v:+.3f}" for v in vals))
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f"n={n}" for n in NMUTS])
    ax2.set_ylabel("median per-peptide Spearman(potential, $-d_{\\mathrm{eucl}}$)")
    ax2.set_title("(b) Fixed-count alignment with Euclidean proximity", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.3, 0.3); ax2.legend(fontsize=9)

    fig.suptitle("TANDEM pairwise-coherence potential vs Euclidean latent distance "
                 f"({df.pep.nunique()} peptides)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = THESIS_FIG / "rq5_tandem_eucl.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
