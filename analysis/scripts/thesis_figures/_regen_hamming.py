"""Regenerate the Hamming-controlled feasibility boxplot (rq5_feasibility_hamming.pdf) from the cached
RQ3 distances, WITHOUT the (now-redefined) MUTANG+ box.  Cache-only; no model."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import CACHE, save  # noqa: E402

METHODS = [("score_A_onehot", "A onehot"), ("score_B_onehot", "B onehot"),
           ("score_A_onehot_lin", "A onehot (lin)"), ("score_B_onehot_lin", "B onehot (lin)"),
           ("score_A_diff", "A diff"), ("score_B_diff", "B diff")]


def main():
    d = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    npep = d.pep.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, nm, lab in [(axes[0], 1, "single mutants"), (axes[1], 2, "double mutants")]:
        recs = []
        for col, plab in METHODS:
            if nm == 1 and "diff" in plab:      # diff potential is constant for single mutants
                continue
            for _p, g in d[d.n_mut == nm].groupby("pep"):
                if len(g) >= 8 and g[col].std() > 0 and g.dist_geo.std() > 0:
                    recs.append({"method": plab, "rho": spearmanr(g[col], -g.dist_geo).correlation})
        rdf = pd.DataFrame(recs)
        order = [pl for _c, pl in METHODS if pl in set(rdf["method"])]
        sns.boxplot(data=rdf, x="method", y="rho", ax=ax, order=order, width=0.6)
        ax.axhline(0, color="#c00", lw=0.8, ls="--")
        ax.set_title(f"within {lab} (n_mut={nm})", fontsize=11, fontweight="bold")
        ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"per-peptide Spearman$(\,$potential, $-d_{\mathrm{geo}}\,)$")
    fig.suptitle("RQ3: Hamming-controlled feasibility alignment (confound removed)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq5_feasibility_hamming.pdf")
    print(f"regenerated over {npep} peptides, TANDEM variants only")


if __name__ == "__main__":
    main()
