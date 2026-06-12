"""RQ3 -- fixed-mutation-count diagnostics behind the \\autoref{tab:rq5-fixedcount} table.

Both numbers in that table come straight from the cached per-candidate distance table
``_thesis_rq5_distances.parquet`` (the 385-peptide RQ3 set; one row per scored candidate,
with its mutation count ``n_mut``, the TANDEM/MUTANG+ scores, and the Euclidean / pullback /
geodesic distances). No model or GPU is needed -- this only re-reads the cache.

It answers, *within a fixed mutation count* (so distance cannot be inflated by simply
mutating more positions):
  * how well Euclidean / pullback latent proximity tracks the geodesic, and
  * whether the TANDEM coherence scores rank candidates by geodesic feasibility (-d_geo).
The whitened-product MUTANG+ filter is evaluated separately from the bury MUTANG+ cache;
the cache column ``score_mutangplus`` here is the old decoder-log-prob baseline and is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import CACHE

NMUTS = (1, 2, 3, 4)
# (row label, column in the parquet, +1 if the column is already a feasibility-like score,
#  -1 if it is a distance that must be negated to become a proximity / feasibility predictor)
# NOTE: the cache's `score_mutangplus` column is the *old* decoder-log-prob baseline (an
# artifact), NOT the current whitened-product MUTANG+ filter; it is deliberately excluded.
# The genuine whitened-MUTANG+ fixed-count check is computed from the bury MUTANG+ cache.
ROWS = [
    ("Euclidean latent proximity $-d_{\\mathrm{eucl}}$", "dist_eucl", -1),
    ("pullback latent proximity $-d_G$",                 "dist_maha", -1),
    ("TANDEM-A coherence",                               "score_A_onehot", +1),
    ("TANDEM-B coherence",                               "score_B_onehot", +1),
]


def median_spearman_vs_feasibility(df, col, sign, nmut):
    """Median over peptides of Spearman(sign*col, feas=-d_geo) within a fixed mutation count."""
    sub = df[df["n_mut"] == nmut]
    rhos = []
    for _, g in sub.groupby("pep"):
        x = sign * g[col].to_numpy()
        y = g["feas"].to_numpy()          # feas = -dist_geo
        if len(g) < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        r, _ = spearmanr(x, y)
        if np.isfinite(r):
            rhos.append(r)
    return (np.median(rhos) if rhos else np.nan), len(rhos)


def main():
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    print(f"rows={len(df):,}  peptides={df['pep'].nunique()}")

    print("\nMedian per-peptide Spearman with geodesic feasibility (-d_geo), within fixed n_mut:")
    header = "  " + " " * 38 + "".join(f"  n={n}" for n in NMUTS)
    print(header)
    for label, col, sign in ROWS:
        cells = []
        for n in NMUTS:
            rho, _ = median_spearman_vs_feasibility(df, col, sign, n)
            cells.append(f"{rho:+.2f}")
        print(f"  {label:38.38s}" + "".join(f"  {c}" for c in cells))

if __name__ == "__main__":
    main()
