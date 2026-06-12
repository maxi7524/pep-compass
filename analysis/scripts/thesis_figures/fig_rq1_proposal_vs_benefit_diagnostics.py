"""RQ1 -- robustness diagnostics for the proposal-vs-benefit cross analysis.

Companion to ``fig_rq1_proposal_vs_benefit.py``. The main script reports a single
overall Spearman correlation between MUTANG's proposal frequency P(a|r) (row-normalised,
Veltri-negative parents) and the activity benefit b = -mean DlogMIC (Gram-,
Veltri-negative). This script adds four robustness checks and produces a decile-binned
companion figure for Section 5.1.4 of the thesis:

  1. Per-source-row Spearman: the per-residue conditional question
     "given source `r`, does MUTANG prefer the more beneficial targets?".
     Row Spearmans are reported individually and aggregated as a row-mass weighted
     mean (weights = sum of proposal counts in the row).
  2. Restriction to well-supported pairs (n_events >= 20): cleans up the noise on
     undersampled cells, matching the cut used by the extremes table.
  3. Robustness on Veltri-positive benefit: the same proposals scored against the
     Gram- benefit measured on the active split.
  4. Permutation null on the overall Spearman: 10 000 random pairings of proposal
     and benefit, to confirm rho ~ 0 is consistent with chance.

Also produces a decile-binned chart (mean benefit by proposal-frequency decile) so
the absence of monotone structure is shown visually rather than only by a Mann-Whitney
tertile comparison.

Outputs (when the script writes):
  results/data/all_in/_cache/_thesis_rq1_propvbenefit_robust.csv
  figures/rq1_proposal_vs_benefit_decile.pdf   (in the thesis figures directory)

All printed diagnostics are intended to be pasted into chapters/results.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

from _common import CACHE, ALL_AA, save

RNG_SEED = 20260606
N_PERMUTATIONS = 10_000
MIN_EVENTS = 20
N_DECILES = 10


def load_inputs():
    """Return per-cell proposal P(a|r) and counts on the Veltri-negative parents,
    together with the Gram- benefit matrices on Veltri-negative and Veltri-positive."""
    proposal_neg = pd.read_parquet(CACHE / "_thesis_mutang_proposal_neg.parquet")
    counts_neg = pd.read_parquet(CACHE / "_thesis_mutang_proposal_counts_neg.parquet")
    benefit_neg = pd.read_parquet(CACHE / "_thesis_benefit_neg_gramneg.parquet")
    benefit_pos = pd.read_parquet(CACHE / "_thesis_benefit_pos_gramneg.parquet")
    return (proposal_neg.reindex(index=ALL_AA, columns=ALL_AA),
            counts_neg.reindex(index=ALL_AA, columns=ALL_AA).fillna(0),
            benefit_neg.reindex(index=ALL_AA, columns=ALL_AA),
            benefit_pos.reindex(index=ALL_AA, columns=ALL_AA))


def long_pairs(proposal, counts, benefit):
    rows = []
    for a in ALL_AA:
        for b in ALL_AA:
            if a == b:
                continue
            p = float(proposal.at[a, b])
            d = float(benefit.at[a, b])
            n = int(counts.at[a, b])
            if not np.isfinite(d):
                continue
            rows.append({"from": a, "to": b, "proposal": p, "benefit": -d, "n": n})
    return pd.DataFrame(rows)


def overall_correlations(df, label):
    rho, p_rho = spearmanr(df["proposal"], df["benefit"])
    r, p_r = pearsonr(df["proposal"], df["benefit"])
    print(f"[{label}] n={len(df)}  Spearman rho={rho:+.4f} (p={p_rho:.3g})  "
          f"Pearson r={r:+.4f} (p={p_r:.3g})")
    return rho, p_rho, r, p_r


def per_row_spearman(df, counts):
    """Per-source-row Spearman, with row-mass weights."""
    out = []
    for a in ALL_AA:
        sub = df[df["from"] == a]
        if len(sub) < 3:
            out.append({"from": a, "rho": np.nan, "n_targets": len(sub),
                        "row_mass": float(counts.loc[a].sum())})
            continue
        rho, p = spearmanr(sub["proposal"], sub["benefit"])
        out.append({"from": a, "rho": rho, "p": p, "n_targets": len(sub),
                    "row_mass": float(counts.loc[a].sum())})
    rdf = pd.DataFrame(out)
    finite = rdf["rho"].notna()
    weights = rdf.loc[finite, "row_mass"].to_numpy()
    rho_w = float(np.average(rdf.loc[finite, "rho"].to_numpy(),
                             weights=weights)) if weights.sum() > 0 else np.nan
    rho_u = float(rdf.loc[finite, "rho"].mean())
    print(f"  per-row Spearman: unweighted mean={rho_u:+.4f}  "
          f"row-mass weighted mean={rho_w:+.4f}  ({finite.sum()} usable rows)")
    return rdf, rho_w, rho_u


def restricted_correlation(df, min_events):
    sub = df[df["n"] >= min_events]
    if len(sub) < 5:
        print(f"  restricted n>={min_events}: only {len(sub)} pairs, skipped.")
        return np.nan, np.nan, len(sub)
    rho, p = spearmanr(sub["proposal"], sub["benefit"])
    print(f"  restricted n>={min_events}: n={len(sub)}  Spearman rho={rho:+.4f} (p={p:.3g})")
    return rho, p, len(sub)


def count_weighted_spearman(df):
    """Weighted Spearman using event counts as weights (ties broken by rank average)."""
    w = df["n"].clip(lower=0).to_numpy(dtype=float)
    if w.sum() == 0:
        return np.nan
    pr = pd.Series(df["proposal"].to_numpy()).rank()
    br = pd.Series(df["benefit"].to_numpy()).rank()
    pm = np.average(pr, weights=w)
    bm = np.average(br, weights=w)
    cov = np.average((pr - pm) * (br - bm), weights=w)
    vp = np.average((pr - pm) ** 2, weights=w)
    vb = np.average((br - bm) ** 2, weights=w)
    rho = cov / np.sqrt(vp * vb) if vp > 0 and vb > 0 else np.nan
    print(f"  count-weighted Spearman: rho={rho:+.4f}  (events-as-weight)")
    return rho


def permutation_null(df, n_perm=N_PERMUTATIONS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    obs, _ = spearmanr(df["proposal"], df["benefit"])
    b = df["benefit"].to_numpy().copy()
    null = np.empty(n_perm, dtype=float)
    p = df["proposal"].to_numpy()
    for i in range(n_perm):
        rng.shuffle(b)
        null[i], _ = spearmanr(p, b)
    p_two = float((np.abs(null) >= abs(obs)).mean())
    pct = (np.percentile(null, [2.5, 50, 97.5]))
    print(f"  permutation null ({n_perm}): observed rho={obs:+.4f}  "
          f"null 2.5/50/97.5 = {pct[0]:+.4f} / {pct[1]:+.4f} / {pct[2]:+.4f}  "
          f"two-sided p={p_two:.3g}")
    return obs, p_two, null


def decile_chart(df, label, out_name):
    df = df.copy().sort_values("proposal").reset_index(drop=True)
    df["decile"] = pd.qcut(df["proposal"], N_DECILES, labels=False,
                           duplicates="drop")
    agg = (df.groupby("decile")["benefit"]
           .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.errorbar(agg["decile"] + 1, agg["mean"], yerr=agg["sem"],
                fmt="o-", color="#1f6feb", lw=1.4, capsize=3, markersize=5,
                label="mean $\\pm$ SEM")
    ax.axhline(0.0, color="#9aa0a6", lw=0.8, ls=":")
    ax.set_xlabel("Proposal-frequency decile (1 = least proposed, 10 = most proposed)",
                  fontsize=11)
    ax.set_ylabel("Mean activity benefit  $-\\overline{\\Delta\\log\\mathrm{MIC}}$",
                  fontsize=11)
    ax.set_xticks(range(1, N_DECILES + 1))
    ax.set_title(f"Decile-binned benefit by MUTANG proposal frequency ({label})",
                 fontsize=12, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.tight_layout()
    save(fig, out_name)
    return agg


def main():
    proposal_neg, counts_neg, benefit_neg, benefit_pos = load_inputs()
    df_neg = long_pairs(proposal_neg, counts_neg, benefit_neg)
    df_pos = long_pairs(proposal_neg, counts_neg, benefit_pos)

    print("=== Section 5.1.4 robustness diagnostics ===\n")
    print("(1) Original analysis: Veltri-negative proposal vs Veltri-negative Gram- benefit")
    overall_correlations(df_neg, "Veltri-neg/Veltri-neg")
    per_row_spearman(df_neg, counts_neg)
    restricted_correlation(df_neg, MIN_EVENTS)
    count_weighted_spearman(df_neg)
    permutation_null(df_neg)

    print("\n(2) Robustness: same proposals vs Veltri-positive Gram- benefit")
    overall_correlations(df_pos, "Veltri-neg/Veltri-pos")
    restricted_correlation(df_pos, MIN_EVENTS)

    print("\n(3) Tertile Mann-Whitney (Veltri-neg) -- kept from the thesis text:")
    proposed = df_neg.copy()
    q_hi = proposed["proposal"].quantile(2 / 3)
    q_lo = proposed["proposal"].quantile(1 / 3)
    hi = proposed[proposed["proposal"] >= q_hi]["benefit"]
    lo = proposed[proposed["proposal"] <= q_lo]["benefit"]
    u, pu = mannwhitneyu(hi, lo, alternative="two-sided")
    print(f"  top third mean benefit={hi.mean():+.4f} (n={len(hi)})  "
          f"bottom third={lo.mean():+.4f} (n={len(lo)})  "
          f"Mann-Whitney U={u:.0f} p={pu:.3g}")

    print("\n(4) Decile-binned mean benefit (Veltri-neg)")
    agg = decile_chart(df_neg, "Veltri-negative, Gram-",
                       "rq1_proposal_vs_benefit_decile.pdf")
    print(agg.to_string(index=False))

    # Persist all diagnostics for reproducibility / future thesis edits.
    out = {
        "metric": ["overall_spearman_neg", "overall_pearson_neg",
                   "overall_spearman_pos", "overall_pearson_pos",
                   "restricted_spearman_neg_n>=20", "count_weighted_spearman_neg",
                   "permutation_pvalue_neg",
                   "mann_whitney_pvalue_neg",
                   "top_third_mean_benefit_neg", "bottom_third_mean_benefit_neg",
                   "n_pairs_neg", "n_pairs_pos"],
        "value": [],
    }
    rho_neg, p_neg, r_neg, _ = overall_correlations(df_neg, "  (redo, persisted)")
    rho_pos, p_pos, r_pos, _ = overall_correlations(df_pos, "  (redo, persisted)")
    sr_neg, _, n_restr = restricted_correlation(df_neg, MIN_EVENTS)
    cw_neg = count_weighted_spearman(df_neg)
    obs_neg, perm_p_neg, _ = permutation_null(df_neg)
    out["value"] = [rho_neg, r_neg, rho_pos, r_pos, sr_neg, cw_neg, perm_p_neg,
                    float(pu), float(hi.mean()), float(lo.mean()),
                    len(df_neg), len(df_pos)]
    csv_path = CACHE / "_thesis_rq1_propvbenefit_robust.csv"
    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
