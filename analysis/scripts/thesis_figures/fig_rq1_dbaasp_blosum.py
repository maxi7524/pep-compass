"""RQ1 -- DBAASP MUTANG proposals as a log-odds matrix next to the DBAASP AMP-BLOSUM.

The MUTANG DBAASP substitution *counts* cannot be read against a BLOSUM directly: a BLOSUM
is a log-odds that has divided out residue abundance, whereas raw counts (and any monotone
transform of them -- log, sqrt, ...) are dominated by how common each residue is, so they
share the same, only weak, rank agreement with the BLOSUM. We therefore put MUTANG's
proposals into the *same* units as the reference by applying the Henikoff log-odds
construction (\autoref{sec:tool-ampblosum}) to the symmetrised DBAASP substitution counts.

Left: MUTANG DBAASP proposal AMP-BLOSUM (symmetric Henikoff log-odds of the counts from
fig_rq1_mutang_proposal.py). Right: the DBAASP-peptide AMP-BLOSUM saved at
figures/blosum_dbaasp.csv. Both panels share the chemical-class axis ordering and the same
sequential scale (white = most disfavoured, blue = most favoured), so favoured cells should
coincide if the two agree.
    rq1_dbaasp_blosum.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

import matplotlib.pyplot as plt

from _common import (CACHE, THESIS_FIG, ALL_AA, AMINO_ACID_CLASSES, SOFT_BLUE,
                     plot_aa_heatmap, save)

COUNTS = CACHE / "_thesis_mutang_proposal_counts_dbaasp.parquet"
BLOSUM = THESIS_FIG / "blosum_dbaasp.csv"
PSEUDO = 0.5     # Henikoff pseudocount on the symmetric pair counts


def load_counts():
    c = pd.read_parquet(COUNTS).reindex(index=ALL_AA, columns=ALL_AA).fillna(0.0)
    np.fill_diagonal(c.values, 0.0)
    return c


def load_blosum():
    b = pd.read_csv(BLOSUM, sep=r"\s+", index_col=0)
    b.columns = [str(c).strip() for c in b.columns]
    b.index = [str(i).strip() for i in b.index]
    return b.reindex(index=ALL_AA, columns=ALL_AA).astype(float)


def henikoff_log_odds(counts, alpha=PSEUDO):
    """Symmetric Henikoff log-odds of a substitution-count matrix:
    log2( p_ab / (p_a p_b) ) on symmetrised counts with a pseudocount."""
    s = counts.values + counts.values.T + alpha
    tot = s.sum()
    p = s / tot
    pa = s.sum(axis=1, keepdims=True) / tot
    pb = s.sum(axis=0, keepdims=True) / tot
    lod = np.log2(p / (pa * pb))
    return pd.DataFrame(lod, index=ALL_AA, columns=ALL_AA)


def main():
    counts = load_counts()
    blosum = load_blosum()
    print(f"DBAASP off-diagonal substitutions: {int(counts.values.sum()):,}")

    mutang_lod = henikoff_log_odds(counts)

    off = ~np.eye(20, dtype=bool)
    b = blosum.values[off]
    for name, x in [("log10(1+count)", np.log10(1.0 + counts.values)[off]),
                    ("Henikoff log-odds", mutang_lod.values[off])]:
        m = np.isfinite(x) & np.isfinite(b)
        rho, _ = spearmanr(x[m], b[m])
        r, _ = pearsonr(x[m], b[m])
        print(f"  vs DBAASP AMP-BLOSUM  {name:18s} Spearman={rho:+.3f}  Pearson={r:+.3f}")

    # --- Extra agreement checks reported in the text (sec:rq3). ----------------------
    # (1) AMP-specificity: same proposals vs the generic BLOSUM62 (expected weaker).
    try:
        from Bio.Align import substitution_matrices
        m62 = substitution_matrices.load("BLOSUM62")
        b62 = (pd.DataFrame(np.array(m62), index=list(m62.alphabet), columns=list(m62.alphabet))
               .reindex(index=ALL_AA, columns=ALL_AA).astype(float))
        x, y = mutang_lod.values[off], b62.values[off]
        mk = np.isfinite(x) & np.isfinite(y)
        print(f"  vs generic BLOSUM62   Henikoff log-odds  Spearman={spearmanr(x[mk], y[mk])[0]:+.3f}")
    except Exception as e:  # Biopython optional; the DBAASP comparison is the headline one
        print(f"  (BLOSUM62 check skipped: {e})")

    # (2) Coarse class signal: are within-chemical-class substitutions proposed more often?
    # Assign each residue to the FIRST class it appears in, matching the block ordering the
    # heatmap uses (Y is in both aromatic and hydroxyl -> counted as aromatic, as drawn).
    cls = {}
    for c, mem in AMINO_ACID_CLASSES["chemical_type"].items():
        for aa in mem:
            cls.setdefault(aa, c)
    fr = counts.values
    within = [fr[i, j] for i, a in enumerate(ALL_AA) for j, c in enumerate(ALL_AA)
              if i != j and cls[a] == cls[c]]
    between = [fr[i, j] for i, a in enumerate(ALL_AA) for j, c in enumerate(ALL_AA)
               if i != j and cls[a] != cls[c]]
    u, p = mannwhitneyu(within, between, alternative="greater")
    print(f"  within-class freq median={np.median(within):.0f} (n={len(within)}) vs "
          f"between-class median={np.median(between):.0f} (n={len(between)})  MWU greater p={p:.2e}")

    # (3) Restricting to well-supported cells (reliable count estimates) before correlating.
    for q in (50, 75):
        thr = np.percentile(counts.values[off], q)
        mask = counts.values[off] >= thr
        rho = spearmanr(mutang_lod.values[off][mask], blosum.values[off][mask])[0]
        print(f"  count>=p{q} ({thr:.0f}): n={int(mask.sum()):3d}  Spearman(Henikoff,DBAASP)={rho:+.3f}")

    # (4) Are the few BLOSUM-favoured (>=0) pairs among MUTANG's most-proposed moves?
    favored = blosum.values[off] >= 0
    freq_off = counts.values[off]
    top_decile = freq_off >= np.percentile(freq_off, 90)
    print(f"  BLOSUM-favoured (>=0) off-diagonal pairs: {int(favored.sum())}; "
          f"of these in MUTANG's top-decile by frequency: {int((favored & top_decile).sum())}")

    def span(df):
        v = df.values[off]
        return float(np.nanmin(v)), float(np.nanmax(v))

    lmin, lmax = span(mutang_lod)
    bmin, bmax = span(blosum)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_aa_heatmap(mutang_lod, classification="chemical_type", cmap=SOFT_BLUE,
                    mask_diagonal=True, vmin=lmin, vmax=lmax, ax=axes[0],
                    title="DBAASP MUTANG proposal AMP-BLOSUM",
                    cbar_label="log-odds score (Henikoff)")
    plot_aa_heatmap(blosum, classification="chemical_type", cmap=SOFT_BLUE,
                    mask_diagonal=True, vmin=bmin, vmax=bmax, ax=axes[1],
                    title="DBAASP AMP-BLOSUM (log-odds)",
                    cbar_label="log-odds score")
    fig.tight_layout()
    save(fig, "rq1_dbaasp_blosum.pdf")


if __name__ == "__main__":
    main()
