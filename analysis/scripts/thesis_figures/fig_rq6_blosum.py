"""RQ4 (Fig. 5.17 / Table 5.10) -- model-induced AMP-BLOSUM vs empirical references.

Consumes the discrete Henikoff log-odds AMP-BLOSUM matrices built on the bury cluster by
``_exp_rq6_blosum_v2.py`` (CPU; no GPU) and scp'd into CACHE:
    _thesis_rq6v2_blosum_mutang.csv                  -- MUTANG, no selection
    _thesis_rq6v2_blosum_tandemB_top25.csv           -- TANDEM-B, top-25%
    _thesis_rq6v2_blosum_mutangplus_argmax_tau0.2.csv-- MUTANG+, argmax, tau=0.2 (whitened)

Compares each, on the shared chemical-class ordering, to BOTH references used in the chapter:
the generic BLOSUM62 and the AMP-specific DBAASP AMP-BLOSUM (figures/blosum_dbaasp.csv,
\\autoref{sec:rq3}). Renders the 4-panel Fig. 5.17 (MUTANG+ tau=0.2, TANDEM-B top-25%, and the
two references; no agreement-vs-selection plot) and prints the Spearman numbers for Table 5.10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from Bio.Align import substitution_matrices

from _common import CACHE, THESIS_FIG, ALL_AA, SOFT_BLUE, plot_aa_heatmap, save

DBAASP = THESIS_FIG / "blosum_dbaasp.csv"
OFF = ~np.eye(20, dtype=bool)
MODELS = {  # label -> bury CSV slug
    "MUTANG+, $\\tau{=}0.2$": "mutangplus_argmax_tau0.2",
    "TANDEM-B, top-25%":      "tandemB_top25",
    "MUTANG (no selection)":  "mutang",
}


def load_matrix(slug):
    f = CACHE / f"_thesis_rq6v2_blosum_{slug}.csv"
    return pd.read_csv(f, index_col=0).reindex(index=ALL_AA, columns=ALL_AA).astype(float)


def blosum62():
    m = substitution_matrices.load("BLOSUM62")
    return (pd.DataFrame(np.array(m), index=list(m.alphabet), columns=list(m.alphabet))
            .reindex(index=ALL_AA, columns=ALL_AA).astype(float))


def load_dbaasp():
    b = pd.read_csv(DBAASP, sep=r"\s+", index_col=0)
    b.columns = [str(c).strip() for c in b.columns]; b.index = [str(i).strip() for i in b.index]
    return b.reindex(index=ALL_AA, columns=ALL_AA).astype(float)


def offdiag_spearman(A, B):
    a, b = np.asarray(A)[OFF], np.asarray(B)[OFF]
    m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).correlation if m.sum() >= 10 else np.nan


def main():
    induced = {label: load_matrix(slug) for label, slug in MODELS.items()}
    refs = {"BLOSUM62": blosum62(), "DBAASP AMP-BLOSUM": load_dbaasp()}

    print("\nOff-diagonal Spearman of each model-induced AMP-BLOSUM vs the references (Table 5.10):")
    print(f"  {'matrix':26s}  {'vs BLOSUM62':>12s}  {'vs DBAASP':>12s}")
    for label, M in induced.items():
        cells = "  ".join(f"{offdiag_spearman(M.values, R.values):>12.3f}" for R in refs.values())
        print(f"  {label:26s}  {cells}")

    # Figure: the two selected model matrices then the two references (drop MUTANG baseline panel).
    panels = [(l, induced[l]) for l in ("MUTANG+, $\\tau{=}0.2$", "TANDEM-B, top-25%")] + list(refs.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 7.2))
    for ax, (label, M) in zip(np.atleast_1d(axes), panels):
        vals = np.asarray(M)[OFF]
        plot_aa_heatmap(M, classification="chemical_type", cmap=SOFT_BLUE,
                        mask_diagonal=True, vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)),
                        ax=ax, title=label, cbar_label="log-odds")
    fig.tight_layout()
    save(fig, "rq6_blosum.pdf")


if __name__ == "__main__":
    main()
