"""RQ1 -- full DBAASP MUTANG substitution counts next to the DBAASP AMP-BLOSUM matrix.

Left: the full DBAASP single-position substitution-count matrix (off-diagonal,
log-scaled for visibility), counted from peptides_mutants_apex.csv by
fig_rq1_mutang_proposal.py. Right: the DBAASP-derived AMP-BLOSUM log-odds matrix saved at
figures/blosum_dbaasp.csv. Both panels use the same chemical-class axis ordering so the
proposed substitutions can be read against the biology-derived scores side by side.
    rq1_dbaasp_blosum.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from _common import (CACHE, THESIS_FIG, ALL_AA, SOFT_BLUE, SOFT_DIVERGING,
                     plot_aa_heatmap, save)

COUNTS = CACHE / "_thesis_mutang_proposal_counts_dbaasp.parquet"
BLOSUM = THESIS_FIG / "blosum_dbaasp.csv"


def load_counts():
    c = pd.read_parquet(COUNTS).reindex(index=ALL_AA, columns=ALL_AA).fillna(0.0)
    np.fill_diagonal(c.values, 0.0)
    return c


def load_blosum():
    b = pd.read_csv(BLOSUM, sep=r"\s+", index_col=0)
    b.columns = [str(c).strip() for c in b.columns]
    b.index = [str(i).strip() for i in b.index]
    return b.reindex(index=ALL_AA, columns=ALL_AA).astype(float)


def main():
    counts = load_counts()
    blosum = load_blosum()
    n_off = int(counts.values.sum())
    print(f"DBAASP off-diagonal substitutions: {n_off:,}")

    log_counts = np.log10(1.0 + counts)
    blim = float(np.nanmax(np.abs(blosum.values[~np.eye(20, dtype=bool)])))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_aa_heatmap(log_counts, classification="chemical_type", cmap=SOFT_BLUE,
                    mask_diagonal=True, vmin=0.0, ax=axes[0],
                    title="DBAASP MUTANG substitution counts",
                    cbar_label="$\\log_{10}(1 + \\#\\,\\mathrm{substitutions})$")
    plot_aa_heatmap(blosum, classification="chemical_type", cmap=SOFT_DIVERGING,
                    mask_diagonal=True, vmin=-blim, vmax=blim, center=0.0, ax=axes[1],
                    title="DBAASP AMP-BLOSUM (log-odds)",
                    cbar_label="log-odds score")
    fig.tight_layout()
    save(fig, "rq1_dbaasp_blosum.pdf")


if __name__ == "__main__":
    main()
