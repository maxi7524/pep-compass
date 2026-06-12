"""Net-displacement (superposition) filter: accuracy + outlier-detection figure.

(a) the first-order superposition proxy ||sum delta_i|| reproduces the exact net latent displacement
    ||z'-z|| almost perfectly (rho ~ 0.99), at a fraction of the cost (only single mutants encoded);
(b) as an outlier detector (top-decile far d_PoGS) it matches the exact displacement and dwarfs the
    TANDEM direction-cosine, within and across mutation count.
Reads _thesis_superpos_filter.parquet (pred_disp, dist_eucl, dist_pogs, n_mut, score_A_onehot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from _common import CACHE, save

df = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")
df["neg_tandem"] = -df["score_A_onehot"]


def auroc(col, nm=None):
    sub = df if nm is None else df[df.n_mut == nm]
    minsize = 20 if nm is None else 12
    au = []
    for _, g in sub.groupby("pep"):
        if len(g) < minsize or g[col].nunique() < 2:
            continue
        y = (g.dist_pogs >= np.quantile(g.dist_pogs, 0.9)).astype(int)
        if 0 < y.sum() < len(y):
            au.append(roc_auc_score(y, g[col].to_numpy()))
    return np.median(au) if au else np.nan


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) accuracy: pred vs exact dist_eucl
    s = df.sample(min(8000, len(df)), random_state=0)
    axA.hexbin(s.dist_eucl, s.pred_disp, gridsize=45, cmap="Blues", bins="log", mincnt=1)
    lo, hi = 0, float(np.quantile(df.dist_eucl, 0.999))
    axA.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
    rho = np.median([spearmanr(g.pred_disp, g.dist_eucl).correlation
                     for _, g in df[df.n_mut >= 2].groupby("pep")
                     if g.pred_disp.nunique() > 1])
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_xlabel("exact net displacement $\\| z'-z\\|$ (re-encode every candidate)")
    axA.set_ylabel("superposition proxy $\\|\\sum_i\\delta_i\\|$ (singles only)")
    axA.set_title(f"(a) first-order superposition is near-exact\nmedian within-count "
                  f"$\\rho={rho:.3f}$", fontsize=11, fontweight="bold")
    axA.legend(fontsize=9, loc="upper left")

    # (b) outlier-detection AUROC
    preds = [("$\\|\\sum\\delta_i\\|$ (superpos.)", "pred_disp", "C0"),
             ("$\\| z'-z\\|$ (exact)", "dist_eucl", "C7"),
             ("TANDEM-A cosine", "neg_tandem", "C3")]
    groups = ["n=2", "n=3", "n=4", "across\ncount"]
    x = np.arange(len(groups)); w = 0.26
    for i, (name, col, c) in enumerate(preds):
        vals = [auroc(col, 2), auroc(col, 3), auroc(col, 4), auroc(col, None)]
        axB.bar(x + (i - 1) * w, vals, w, color=c, label=name)
    axB.axhline(0.5, color="0.5", lw=0.8, ls=":")
    axB.set_xticks(x); axB.set_xticklabels(groups)
    axB.set_ylim(0.45, 0.92)
    axB.set_ylabel("median outlier-detection AUROC (top-decile far $d_{\\mathrm{PoGS}}$)")
    axB.set_title("(b) magnitude detects outliers; direction does not", fontsize=11, fontweight="bold")
    axB.legend(fontsize=9)

    fig.suptitle(f"Net-displacement (superposition) filter ({df.pep.nunique()} peptides)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq5_superpos.pdf")
    print(f"median within-count rho(pred, dist_eucl) = {rho:.3f}")
    for name, col, _ in preds:
        print(f"  AUROC {name:30s} n2={auroc(col,2):.3f} n3={auroc(col,3):.3f} "
              f"n4={auroc(col,4):.3f} across={auroc(col,None):.3f}")


if __name__ == "__main__":
    main()
