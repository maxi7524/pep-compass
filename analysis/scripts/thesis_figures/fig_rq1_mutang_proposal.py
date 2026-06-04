"""RQ1 -- the substitution distribution MUTANG proposes, across four datasets.

Runs MUTANG single-position enumeration on parent peptides from four datasets
(Veltri-positive, Veltri-negative, AMPSphere, DBAASP) at the production thresholds
(direction-significance kappa = 1e-3, token threshold theta_mut = 0.1, finite-difference
Jacobian) and builds, for each, the row-normalised 20x20 "mutational-signature" matrix
P(to | from). The four matrices are shown as a 2x2 grid:
    rq1_proposal_grid.pdf

Only parent sequences and the HydrAMP model are needed (no APEX MIC), so no streaming of
the large mutants table is required. Large datasets are capped to N_CAP parents (sampled
reproducibly) so enumeration stays tractable. The Veltri-negative proposal matrix is also
saved on its own for the proposal-vs-benefit cross analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import PROJECT_ROOT, CACHE, ALL_AA, SOFT_BLUE, plot_aa_heatmap, save

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "scripts" / "mutang"
                      / "enumerate_mutations_single_position"))

from enumerate_mutations_single_position import (  # noqa: E402
    load_hydramp_model, get_mutants_from_single_position_mutations_from_df,
)

KAPPA, THETA_MUT = 0.001, 0.1
N_CAP = 5000                       # cap on parents enumerated per dataset
APEX = PROJECT_ROOT / "results" / "data" / "all_in" / "peptides_apex.csv"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# (display name, provenance tag, short slug)
DATASETS = [
    ("Veltri-positive", "hydramp_veltri_positive", "veltri_positive"),
    ("Veltri-negative", "hydramp_veltri_negative", "veltri_negative"),
    ("AMPSphere",       "ampsphere_amp",           "ampsphere"),
    ("DBAASP",          "hydramp_dbaasp_clean",    "dbaasp"),
]


def build_change_matrix(df, normalize=True):
    d = df.dropna(subset=["parent", "mutant", "position"]).drop_duplicates()
    d = d[(d["position"] < d["parent"].str.len()) & (d["position"] < d["mutant"].str.len())]
    d = d[d["parent"] != d["mutant"]]
    pos = d["position"].astype(int)
    from_aa = [s[i] for s, i in zip(d["parent"], pos)]
    to_aa = [s[i] for s, i in zip(d["mutant"], pos)]
    counts = (pd.DataFrame({"parent": from_aa, "mutant": to_aa})
              .value_counts().unstack(fill_value=0)
              .reindex(index=ALL_AA, columns=ALL_AA, fill_value=0).astype(float))
    matrix = counts.copy()
    np.fill_diagonal(matrix.values, 0.0)
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0.0)
    return matrix, counts


def parents_for_tag(tag, all_parents):
    mask = all_parents["dataset"].fillna("").apply(lambda s: tag in s.split(";"))
    seqs = all_parents.loc[mask, "sequence"].dropna().astype(str)
    seqs = seqs[(seqs.str.len() <= 25) & (seqs.str.len() > 0)
                & seqs.apply(lambda s: set(s) <= STANDARD_AA)].drop_duplicates()
    return seqs.reset_index(drop=True)


def enumerate_dataset(slug, tag, all_parents, model):
    cache = CACHE / f"_thesis_mutang_mutants_{slug}.parquet"
    if cache.exists():
        print(f"  [{slug}] cached")
        return pd.read_parquet(cache)
    seqs = parents_for_tag(tag, all_parents)
    if len(seqs) > N_CAP:
        seqs = seqs.sample(n=N_CAP, random_state=0).reset_index(drop=True)
        print(f"  [{slug}] capped to {N_CAP} of available parents")
    df = pd.DataFrame({"sequence": seqs})
    print(f"  [{slug}] enumerating MUTANG mutants for {len(df)} parents ...")
    mutants = get_mutants_from_single_position_mutations_from_df(
        df, "sequence", model, KAPPA, THETA_MUT)
    mutants.to_parquet(cache, index=False)
    return mutants


def main():
    all_parents = pd.read_csv(APEX, usecols=["sequence", "dataset"])
    model = load_hydramp_model(jacobian_mode="approx", jacobian_eps=1e-6)

    mats, counts_by_slug = {}, {}
    for name, tag, slug in DATASETS:
        mutants = enumerate_dataset(slug, tag, all_parents, model)
        m, c = build_change_matrix(mutants)
        mats[slug] = m
        counts_by_slug[slug] = c
        n_off = int(c.values.sum() - np.trace(c.values))
        print(f"  [{slug}] {n_off:,} off-diagonal substitutions")

    # shared colour scale across panels
    vmax = float(np.nanpercentile(np.concatenate(
        [m.values[~np.eye(20, dtype=bool)] for m in mats.values()]), 98))

    fig, axes = plt.subplots(2, 2, figsize=(19, 17))
    for ax, (name, tag, slug) in zip(axes.ravel(), DATASETS):
        plot_aa_heatmap(mats[slug], classification="chemical_type", cmap=SOFT_BLUE,
                        mask_diagonal=True, vmin=0.0, vmax=vmax, ax=ax,
                        title=f"{name}", cbar_label="P(mutate to | from)")
    fig.suptitle("MUTANG proposal signature across datasets "
                 f"(kappa={KAPPA}, theta_mut={THETA_MUT}, row-normalised)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "rq1_proposal_grid.pdf")

    # persist veltri-negative proposal for the cross analysis (reused by fig_rq1_proposal_vs_benefit)
    mats["veltri_negative"].to_parquet(CACHE / "_thesis_mutang_proposal_neg.parquet")
    counts_by_slug["veltri_negative"].to_parquet(
        CACHE / "_thesis_mutang_proposal_counts_neg.parquet")
    print("saved Veltri-negative proposal matrix for cross-analysis")


if __name__ == "__main__":
    main()
