"""RQ1 -- biological DlogMIC mutational signatures (veltri +/-, Gram-/Gram+).

Replicates karol_mutational_signature_analysis.ipynb from the parquet caches and writes
vector PDFs:
    rq1_sig_neg.pdf   -- veltri_negative, Gram- and Gram+ side by side
    rq1_sig_pos.pdf   -- veltri_positive, Gram- and Gram+ side by side
    rq1_sig_diff.pdf  -- veltri_negative / veltri_positive / (positive - negative), Gram-

Also prints the overall means and the top activity-improving/harming substitutions used
to populate the LaTeX tables.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (CACHE, ALL_AA, GREEN_WHITE_RED, plot_aa_heatmap, save)

# Gram classification of the 34 APEX strains (column order identical across all_in caches).
gram_neg = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 16, 17, 18, 19, 20, 22, 23, 24, 29, 30, 31, 32]
gram_pos = [7, 8, 9, 10, 14, 15, 21, 25, 26, 27, 28, 33]
assert sorted(gram_neg + gram_pos) == list(range(34))


def load_split(tag: str):
    parents = pd.read_parquet(CACHE / f"parents_hydramp_{tag}.parquet")
    mutants = pd.read_parquet(CACHE / f"mutants_hydramp_{tag}.parquet")
    return parents, mutants


def compute_diff(parents_df, mutants_df, value_cols):
    merged = mutants_df.merge(parents_df, left_on="parent", right_on="sequence",
                              suffixes=("_mutants", "_parents"))
    for c in value_cols:
        merged[f"{c}_log_diff"] = (np.log2(merged[f"{c}_mutants"])
                                   - np.log2(merged[f"{c}_parents"]))
    return merged


def prepare_diff(parents_df, mutants_df, value_cols):
    diff = compute_diff(parents_df, mutants_df, value_cols)
    diff = diff.dropna(subset=["parent", "mutant", "position"]).drop_duplicates(
        subset=["parent", "mutant", "position"])
    pos = diff["position"].astype(int)
    diff = diff[(pos < diff["parent"].str.len()) & (pos < diff["mutant"].str.len())]
    pos = diff["position"].astype(int)
    diff["parent_aa"] = [s[i] for s, i in zip(diff["parent"], pos)]
    diff["mutant_aa"] = [s[i] for s, i in zip(diff["mutant"], pos)]
    return diff[diff["parent_aa"] != diff["mutant_aa"]]


def build_signature_matrices(diff, neg_cols, pos_cols):
    neg_m = pd.DataFrame(np.nan, index=ALL_AA, columns=ALL_AA, dtype=float)
    pos_m = pd.DataFrame(np.nan, index=ALL_AA, columns=ALL_AA, dtype=float)
    cnt = pd.DataFrame(0, index=ALL_AA, columns=ALL_AA, dtype=int)
    for (p_aa, m_aa), g in diff.groupby(["parent_aa", "mutant_aa"]):
        if p_aa not in ALL_AA or m_aa not in ALL_AA:
            continue
        neg_m.at[p_aa, m_aa] = np.nanmean(g[neg_cols].to_numpy())
        pos_m.at[p_aa, m_aa] = np.nanmean(g[pos_cols].to_numpy())
        cnt.at[p_aa, m_aa] = len(g)
    return neg_m, pos_m, cnt


def extremes(mat, cnt, min_count=20, k=8):
    flat = [(p, m, mat.at[p, m], cnt.at[p, m]) for p in ALL_AA for m in ALL_AA
            if p != m and np.isfinite(mat.at[p, m]) and cnt.at[p, m] >= min_count]
    flat.sort(key=lambda r: r[2])
    return flat[:k], flat[-k:][::-1]


def main():
    parents_neg, mutants_neg = load_split("veltri_negative")
    parents_pos, mutants_pos = load_split("veltri_positive")
    bac_cols = parents_neg.columns.tolist()[4:]
    assert len(bac_cols) == 34, len(bac_cols)

    delta_cols = [f"{c}_log_diff" for c in bac_cols]
    neg_delta_cols = [delta_cols[i] for i in gram_neg]
    pos_delta_cols = [delta_cols[i] for i in gram_pos]

    diff_neg = prepare_diff(parents_neg, mutants_neg, bac_cols)
    diff_pos = prepare_diff(parents_pos, mutants_pos, bac_cols)
    print(f"veltri_negative usable single-AA mutations: {len(diff_neg):,}")
    print(f"veltri_positive usable single-AA mutations: {len(diff_pos):,}")

    neg_gn, neg_gp, neg_cnt = build_signature_matrices(diff_neg, neg_delta_cols, pos_delta_cols)
    pos_gn, pos_gp, pos_cnt = build_signature_matrices(diff_pos, neg_delta_cols, pos_delta_cols)

    allv = np.concatenate([m.to_numpy().ravel() for m in [neg_gn, neg_gp, pos_gn, pos_gp]])
    VLIM = float(np.nanpercentile(np.abs(allv), 98))
    print(f"Shared symmetric colour limit (98th pct of |DlogMIC|): +/-{VLIM:.3f}")

    cbar = "mean DlogMIC  (green: lower MIC / better; red: higher MIC / worse)"

    # rq1_sig_neg.pdf
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_aa_heatmap(neg_gn, title="veltri_negative -- Gram- strains", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[0], cbar_label=None)
    plot_aa_heatmap(neg_gp, title="veltri_negative -- Gram+ strains", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[1], cbar_label=cbar)
    fig.tight_layout()
    save(fig, "rq1_sig_neg.pdf")

    # rq1_sig_pos.pdf
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_aa_heatmap(pos_gn, title="veltri_positive -- Gram- strains", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[0], cbar_label=None)
    plot_aa_heatmap(pos_gp, title="veltri_positive -- Gram+ strains", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[1], cbar_label=cbar)
    fig.tight_layout()
    save(fig, "rq1_sig_pos.pdf")

    # rq1_sig_diff.pdf
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    plot_aa_heatmap(neg_gn, title="veltri_negative (Gram-)", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[0], cbar_label=None)
    plot_aa_heatmap(pos_gn, title="veltri_positive (Gram-)", vmin=-VLIM, vmax=VLIM,
                    center=0, ax=axes[1], cbar_label=None)
    plot_aa_heatmap(pos_gn - neg_gn, title="difference (positive - negative), Gram-",
                    vmin=-VLIM, vmax=VLIM, center=0, ax=axes[2],
                    cbar_label="D(pos - neg) mean DlogMIC")
    fig.tight_layout()
    save(fig, "rq1_sig_diff.pdf")

    # --- numbers for the tables ---
    print("\nOverall mean DlogMIC (negative => mutations improve activity on average):")
    for name, d in [("veltri_negative", diff_neg), ("veltri_positive", diff_pos)]:
        print(f"  {name:16s}  Gram-: {np.nanmean(d[neg_delta_cols].to_numpy()):+.4f}   "
              f"Gram+: {np.nanmean(d[pos_delta_cols].to_numpy()):+.4f}")
    for label, mat, cnt in [("veltri_positive Gram-", pos_gn, pos_cnt),
                            ("veltri_negative Gram-", neg_gn, neg_cnt)]:
        improving, harming = extremes(mat, cnt)
        print(f"\n=== {label} ===")
        print("  improving:", [f"{p}->{m} {v:+.3f} (n={n})" for p, m, v, n in improving])
        print("  harming:  ", [f"{p}->{m} {v:+.3f} (n={n})" for p, m, v, n in harming])

    # persist the benefit matrix for the proposal-vs-benefit cross analysis
    pos_gn.to_parquet(CACHE / "_thesis_benefit_pos_gramneg.parquet")
    neg_gn.to_parquet(CACHE / "_thesis_benefit_neg_gramneg.parquet")
    print("\nsaved benefit matrices for cross-analysis")


if __name__ == "__main__":
    main()
