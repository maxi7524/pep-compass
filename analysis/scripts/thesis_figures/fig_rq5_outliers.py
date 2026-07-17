"""RQ5 follow-up -- are geodesic-distance OUTLIERS the reason potential-based selection fails,
and can the distance be made useful once they are handled?

The RQ5 selection result (top-p% by potential is no closer to the parent) uses the *mean* geodesic
distance.  A natural worry (and a natural hope) is that a few far-distance candidates dominate that
mean, so that (a) the null is an outlier artifact and (b) a robust statistic, or trimming the
outliers, would reveal that selection actually helps.  We test this directly, reusing the cached
per-candidate geodesic distances of the RQ5 analysis (``_thesis_rq5_distances.parquet``); no model
or geodesic is recomputed.

For every peptide, within a fixed mutation count n_mut in {2,3,4} (so the Hamming confound is
controlled exactly as in RQ5), we measure:
  (a) outlier severity      : per-group max / median d_geo, and the fraction of groups whose
                              farthest candidate exceeds 10x the median;
  (b) trimmed alignment     : Spearman(potential, -d_geo) on the full group and after dropping the
                              top 5% / 10% most-distant candidates;
  (c) outlier flagging (AUC): can a low potential identify the top-decile (farthest) candidates?
  (d) robust selection      : ratio of the selected-to-full d_geo under the mean, the median, and a
                              5%-trimmed mean, for the top-50% of candidates by potential.
Potentials: TANDEM-A (whitened latent cosine), TANDEM-B (pullback-metric cosine), MUTANG+ (decoder
log-probability).  Writes ``rq5_distance_outliers.pdf`` and ``_thesis_rq5_outliers.csv``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import CACHE, save  # noqa: E402

POTS = {"TANDEM-A": "score_A_onehot", "TANDEM-B": "score_B_onehot"}
COLORS = {"TANDEM-A": "#4292c6", "TANDEM-B": "#fd8d3c"}
NMUTS = (2, 3, 4)
OUTLIER_MULT = 10.0      # "severe" outlier: farthest candidate > 10x the within-group median
MIN_GROUP = 12           # minimum candidates in a (peptide, n_mut) group to use it


def _auc(score, label):
    """AUC that ``score`` ranks positives (label==1) above negatives (Mann-Whitney form)."""
    label = np.asarray(label)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(score)).rank().values
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _trimmed_mean(x, q=0.05):
    x = np.sort(np.asarray(x))
    k = int(len(x) * q)
    return x[k:len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def main():
    d = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    print(f"[outliers] {len(d)} candidates, {d.pep.nunique()} peptides")

    sev = []                                                # (a) outlier severity per group
    align = {p: {"raw": [], "trim5": [], "trim10": []} for p in POTS}   # (b)
    auc_tail = {p: [] for p in POTS}                        # (c)
    selrat = {p: {"mean": [], "median": [], "trim": []} for p in POTS}  # (d)

    for (_pep, nm), g in d.groupby(["pep", "n_mut"]):
        if len(g) < MIN_GROUP or nm not in NMUTS:
            continue
        dg = g.dist_geo.values
        med = np.median(dg)
        if med > 0:
            sev.append(dg.max() / med)
        thr90 = np.quantile(dg, 0.90)
        tail = (dg >= thr90).astype(int)
        keep5 = dg <= np.quantile(dg, 0.95)
        keep10 = dg <= np.quantile(dg, 0.90)
        for p, col in POTS.items():
            s = g[col].values
            if np.std(s) == 0:
                continue
            if np.std(dg) > 0:
                align[p]["raw"].append(spearmanr(s, -dg).correlation)
            if keep5.sum() >= 8 and np.std(dg[keep5]) > 0 and np.std(s[keep5]) > 0:
                align[p]["trim5"].append(spearmanr(s[keep5], -dg[keep5]).correlation)
            if keep10.sum() >= 8 and np.std(dg[keep10]) > 0 and np.std(s[keep10]) > 0:
                align[p]["trim10"].append(spearmanr(s[keep10], -dg[keep10]).correlation)
            auc_tail[p].append(_auc(-s, tail))             # low potential -> flagged as far
            thr = np.quantile(s, 0.5)
            sel = s >= thr
            if sel.sum() >= 5:
                for name, fn in (("mean", np.mean), ("median", np.median),
                                 ("trim", lambda z: _trimmed_mean(z, 0.05))):
                    full = fn(dg)
                    if full > 0:
                        selrat[p][name].append(fn(dg[sel]) / full)

    sev = np.asarray(sev)
    frac_severe = float(np.mean(sev > OUTLIER_MULT))
    print(f"(a) groups n={len(sev)}; median max/med={np.median(sev):.2f}; "
          f"frac > {OUTLIER_MULT:g}x = {frac_severe:.1%}")

    rows = []
    for p in POTS:
        rec = {
            "potential": p,
            "align_raw": np.nanmedian(align[p]["raw"]),
            "align_trim5": np.nanmedian(align[p]["trim5"]),
            "align_trim10": np.nanmedian(align[p]["trim10"]),
            "auc_tail": np.nanmedian(auc_tail[p]),
            "sel_mean": np.nanmedian(selrat[p]["mean"]),
            "sel_median": np.nanmedian(selrat[p]["median"]),
            "sel_trim": np.nanmedian(selrat[p]["trim"]),
        }
        rows.append(rec)
        print(f"  {p:10s} align raw/trim5/trim10 = "
              f"{rec['align_raw']:+.3f}/{rec['align_trim5']:+.3f}/{rec['align_trim10']:+.3f}  "
              f"AUC_tail={rec['auc_tail']:.3f}  sel(mean/med/trim)="
              f"{rec['sel_mean']:.3f}/{rec['sel_median']:.3f}/{rec['sel_trim']:.3f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(CACHE / "_thesis_rq5_outliers.csv", index=False)
    meta = pd.DataFrame([{"n_groups": len(sev), "median_max_over_med": float(np.median(sev)),
                          f"frac_above_{int(OUTLIER_MULT)}x": frac_severe}])
    meta.to_csv(CACHE / "_thesis_rq5_outliers_meta.csv", index=False)

    # ---------------------------- figure (2 x 2) ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pnames = list(POTS)
    xpos = np.arange(len(pnames))
    cols = [COLORS[p] for p in pnames]

    # (a) ECDF of per-group max/median d_geo (heavy tail)
    ax = axes[0, 0]
    xs = np.sort(sev)
    ax.plot(xs, np.linspace(0, 1, len(xs)), color="#333", lw=1.6)
    ax.axvline(OUTLIER_MULT, color="#d7191c", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"per-group $\max / \mathrm{median}$ of $d_{\mathrm{geo}}$")
    ax.set_ylabel("cumulative fraction of (peptide, $n_{\\mathrm{mut}}$) groups")
    ax.set_title("(a) geodesic-distance outliers are heavy-tailed", fontweight="bold", fontsize=11)
    ax.text(OUTLIER_MULT * 1.15, 0.08, f"{frac_severe:.0%} of groups\n> {OUTLIER_MULT:g}$\\times$",
            color="#d7191c", fontsize=9, va="bottom")

    # (b) trimmed alignment
    ax = axes[0, 1]
    w = 0.26
    for j, key in enumerate(("raw", "trim5", "trim10")):
        vals = [np.nanmedian(align[p][key]) for p in pnames]
        ax.bar(xpos + (j - 1) * w, vals, w, label={"raw": "full", "trim5": "drop top 5%",
               "trim10": "drop top 10%"}[key], color=plt.cm.Greys(0.4 + 0.22 * j), edgecolor="white")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(pnames)
    ax.set_ylabel(r"median Spearman(potential, $-d_{\mathrm{geo}}$)")
    ax.set_title("(b) trimming outliers does not rescue alignment", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    # (c) outlier flagging AUC
    ax = axes[1, 0]
    vals = [np.nanmedian(auc_tail[p]) for p in pnames]
    ax.bar(xpos, vals, 0.6, color=cols, edgecolor="white")
    ax.axhline(0.5, color="#d7191c", ls="--", lw=1.0)
    ax.set_ylim(0.45, max(0.62, max(vals) + 0.03))
    ax.set_xticks(xpos); ax.set_xticklabels(pnames)
    ax.set_ylabel("AUC: low potential flags top-decile $d_{\\mathrm{geo}}$")
    ax.set_title("(c) neither variant flags the far tail", fontweight="bold", fontsize=11)
    for x, v in zip(xpos, vals):
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)

    # (d) robust selection ratio (top-50% by potential)
    ax = axes[1, 1]
    for j, key in enumerate(("mean", "median", "trim")):
        vals = [np.nanmedian(selrat[p][key]) for p in pnames]
        ax.bar(xpos + (j - 1) * w, vals, w, label={"mean": "mean", "median": "median",
               "trim": "5%-trimmed mean"}[key], color=plt.cm.Blues(0.4 + 0.22 * j), edgecolor="white")
    ax.axhline(1.0, color="#d7191c", ls="--", lw=1.0)
    ax.set_ylim(0.85, 1.05)
    ax.set_xticks(xpos); ax.set_xticklabels(pnames)
    ax.set_ylabel(r"$d_{\mathrm{geo}}$ ratio, top-50%-by-potential / full")
    ax.set_title("(d) robust selection brings no candidate closer", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes.ravel():
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Geodesic-distance outliers and robust feasibility selection",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, "rq5_distance_outliers.pdf")


if __name__ == "__main__":
    main()
