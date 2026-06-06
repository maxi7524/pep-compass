"""RQ1 -- the substitution distribution MUTANG proposes, across four datasets.

Instead of re-running MUTANG enumeration (and capping AMPSphere to a subsample), this
counts the single-position substitutions directly from the production mutants table
``peptides_mutants_apex.csv`` -- the canonical enumeration used throughout the thesis,
generated at the production thresholds (kappa = 1e-3, theta_mut = 1e-6). Counting from the
CSV is cheap and lets every dataset, AMPSphere included, be used in full.

For each dataset (Veltri-positive, Veltri-negative, AMPSphere, DBAASP) we build the
row-normalised 20x20 "mutational-signature" matrix P(to | from) and show the four as a
2x2 grid:
    rq1_proposal_grid.pdf

The Veltri-negative proposal matrix (for the proposal-vs-benefit cross analysis) and the
full DBAASP substitution-count matrix (for the DBAASP-vs-BLOSUM comparison) are cached.
The per-dataset off-diagonal substitution counts are printed for the LaTeX text.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import PROJECT_ROOT, CACHE, ALL_AA, SOFT_BLUE, plot_aa_heatmap, save

KAPPA, THETA_MUT = 1e-3, 1e-6     # production thresholds the big table was enumerated at
DATA = PROJECT_ROOT / "results" / "data" / "all_in"
APEX = DATA / "peptides_apex.csv"
MUTANTS = DATA / "peptides_mutants_apex.csv"
STANDARD_AA = set(ALL_AA)
CHUNK = 2_000_000

# (display name, provenance tag, short slug)
DATASETS = [
    ("Veltri-positive", "hydramp_veltri_positive", "veltri_positive"),
    ("Veltri-negative", "hydramp_veltri_negative", "veltri_negative"),
    ("AMPSphere",       "ampsphere_amp",           "ampsphere"),
    ("DBAASP",          "hydramp_dbaasp_clean",    "dbaasp"),
]


def parent_sets():
    """Map each dataset slug to the set of its parent sequences (a parent may belong to
    several datasets, exactly as in the original per-dataset enumeration)."""
    ap = pd.read_csv(APEX, usecols=["sequence", "dataset"])
    ap["dataset"] = ap["dataset"].fillna("")
    sets = {}
    for _, tag, slug in DATASETS:
        mask = ap["dataset"].apply(lambda s: tag in s.split(";"))
        sets[slug] = set(ap.loc[mask, "sequence"].dropna().astype(str))
        print(f"  [{slug}] {len(sets[slug]):,} parent sequences")
    return sets


def count_substitutions(sets):
    counts = {slug: pd.DataFrame(0.0, index=ALL_AA, columns=ALL_AA) for _, _, slug in DATASETS}
    n_rows = 0
    for chunk in pd.read_csv(MUTANTS, usecols=["mutant", "position", "parent"],
                             chunksize=CHUNK):
        chunk = chunk.dropna(subset=["mutant", "position", "parent"])
        chunk = chunk.drop_duplicates(subset=["parent", "position", "mutant"])
        chunk["parent"] = chunk["parent"].astype(str)
        chunk["mutant"] = chunk["mutant"].astype(str)
        pos = chunk["position"].astype(int).to_numpy()
        plen = chunk["parent"].str.len().to_numpy()
        mlen = chunk["mutant"].str.len().to_numpy()
        par = chunk["parent"].to_numpy()
        mut = chunk["mutant"].to_numpy()
        ok = (pos >= 0) & (pos < plen) & (pos < mlen)
        pos, par, mut = pos[ok], par[ok], mut[ok]
        from_aa = np.array([s[i] for s, i in zip(par, pos)])
        to_aa = np.array([s[i] for s, i in zip(mut, pos)])
        keep = (from_aa != to_aa)
        sub = pd.DataFrame({"parent": par[keep], "f": from_aa[keep], "t": to_aa[keep]})
        sub = sub[sub["f"].isin(STANDARD_AA) & sub["t"].isin(STANDARD_AA)]
        for _, _, slug in DATASETS:
            d = sub[sub["parent"].isin(sets[slug])]
            if len(d):
                ct = (pd.crosstab(d["f"], d["t"])
                      .reindex(index=ALL_AA, columns=ALL_AA, fill_value=0))
                counts[slug] = counts[slug].add(ct, fill_value=0)
        n_rows += len(chunk)
        print(f"  ... {n_rows:,} rows scanned", flush=True)
    return {slug: counts[slug].reindex(index=ALL_AA, columns=ALL_AA).fillna(0.0)
            for _, _, slug in DATASETS}


def row_normalise(counts):
    m = counts.copy()
    np.fill_diagonal(m.values, 0.0)
    return m.div(m.sum(axis=1), axis=0).fillna(0.0)


def main():
    # Reuse cached per-dataset counts when present (counting the 14 GB table is the slow part;
    # layout tweaks should not require a re-scan). Set REGEN_COUNTS=1 to force a re-count.
    cache_files = {slug: CACHE / f"_thesis_mutang_proposal_counts_{slug}.parquet"
                   for _, _, slug in DATASETS}
    if os.environ.get("REGEN_COUNTS") != "1" and all(p.exists() for p in cache_files.values()):
        print("  reusing cached per-dataset counts")
        counts = {slug: pd.read_parquet(cache_files[slug]).reindex(
            index=ALL_AA, columns=ALL_AA).fillna(0.0) for _, _, slug in DATASETS}
    else:
        sets = parent_sets()
        counts = count_substitutions(sets)
        for _, _, slug in DATASETS:
            counts[slug].to_parquet(cache_files[slug])

    mats = {slug: row_normalise(counts[slug]) for _, _, slug in DATASETS}

    print("\nOff-diagonal single-residue substitutions per dataset "
          f"(kappa={KAPPA}, theta_mut={THETA_MUT}):")
    n_off = {}
    for name, _, slug in DATASETS:
        c = counts[slug].values
        n_off[slug] = int(c.sum() - np.trace(c))
        print(f"  {name:16s} {n_off[slug]:,}")

    vmax = float(np.nanpercentile(np.concatenate(
        [m.values[~np.eye(20, dtype=bool)] for m in mats.values()]), 98))

    fig, axes = plt.subplots(2, 2, figsize=(19, 18))
    for ax, (name, _, slug) in zip(axes.ravel(), DATASETS):
        plot_aa_heatmap(mats[slug], classification="chemical_type", cmap=SOFT_BLUE,
                        mask_diagonal=True, vmin=0.0, vmax=vmax, ax=ax,
                        title=f"{name}", cbar_label="P(mutate to | from)")
    fig.suptitle("MUTANG proposal signature across datasets "
                 f"($\\kappa$={KAPPA}, $\\theta_{{mut}}$={THETA_MUT}, row-normalised)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    # extra room between rows so the (horizontal) class labels under the top panels do not
    # collide with the titles of the bottom panels.
    fig.subplots_adjust(hspace=0.42, wspace=0.30)
    save(fig, "rq1_proposal_grid.pdf")

    # caches reused downstream
    mats["veltri_negative"].to_parquet(CACHE / "_thesis_mutang_proposal_neg.parquet")
    counts["veltri_negative"].to_parquet(CACHE / "_thesis_mutang_proposal_counts_neg.parquet")
    counts["dbaasp"].to_parquet(CACHE / "_thesis_mutang_proposal_counts_dbaasp.parquet")
    print("saved Veltri-negative proposal and DBAASP count matrices")


if __name__ == "__main__":
    main()
