"""TANDEM as a feasibility *filter*: cumulative-density operating curve + matched-random lift.

Tests the claim that TANDEM removes the far/outlier mutants (not that it ranks all candidates).
Everything is computed WITHIN a fixed mutation count and against a SIZE-matched random baseline, so
neither the mutation-count confound nor the set-size effect can manufacture a signal. Reads only
cached scores/distances -- no new geometry.

Selection: keep the smallest candidate set holding the top-p of soft-max(potential) (nucleus).
Tail metric: CVaR@10% (mean of the worst 10% kept distances) and P90; cost axis: retention.
Lift = (random size-k subset metric) - (TANDEM top-k metric), per (peptide, n_mut) group, aggregated
with median + bootstrap CI + Wilcoxon signed-rank across groups.

Outputs rq5_pogs_filter_eval.pdf and _thesis_pogs_filter_eval.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from _common import CACHE, save

DIST = "dist_pogs"
POTS = [("TANDEM-A one-hot", "score_A_onehot", "C0"),
        ("TANDEM-A diff", "score_A_diff", "C2")]
COUNTS = (2, 3, 4)
MINSIZE = 12                       # min candidates in a (pep, n_mut) group for stable tail stats
P_GRID = np.round(np.arange(0.05, 1.0 + 1e-9, 0.05), 3)
R = 100                            # random subsets per group (size-matched baseline)
rng = np.random.default_rng(0)


def cvar10(a):                     # mean of the worst (largest) 10%
    m = max(1, int(np.ceil(0.10 * len(a))))
    return np.sort(a)[-m:].mean()


def nucleus_k(scores, p):
    w = np.exp(scores - scores.max()); w /= w.sum()
    order = np.argsort(-scores)
    k = int(np.searchsorted(np.cumsum(w[order]), p) + 1)
    return min(max(k, 1), len(scores)), order


def main():
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs.parquet").reset_index(drop=True)
    groups = [(pep, nm, g) for nm in COUNTS
              for pep, g in df[df.n_mut == nm].groupby("pep") if len(g) >= MINSIZE]
    print(f"{DIST}: {len(groups)} (pep,n_mut) groups with >= {MINSIZE} candidates "
          f"({df.pep.nunique()} peptides)")

    rows = []
    for name, col, _ in POTS:
        # precompute per group: distances, potentials (drop groups with constant potential)
        gd = [(g[DIST].to_numpy(), g[col].to_numpy()) for _, _, g in groups]
        gd = [(d, s) for d, s in gd if np.std(s) > 0 and np.std(d) > 0]
        rand_orders = [np.argsort(rng.random((R, len(d))), axis=1) for d, _ in gd]  # reuse across p
        for p in P_GRID:
            l_cvar, l_p90, ret, k_cvar, r_cvar = [], [], [], [], []
            for (d, s), ro in zip(gd, rand_orders):
                N = len(d)
                k, order = nucleus_k(s, p)
                kept = d[order[:k]]
                sub = d[ro[:, :k]]                              # (R, k) random size-k subsets
                m = max(1, int(np.ceil(0.10 * k)))
                rand_cvar = np.sort(sub, axis=1)[:, -m:].mean(1).mean()
                rand_p90 = np.quantile(sub, 0.90, axis=1).mean()
                kc, kp = cvar10(kept), np.quantile(kept, 0.90)
                l_cvar.append(rand_cvar - kc); l_p90.append(rand_p90 - kp)
                ret.append(k / N); k_cvar.append(kc); r_cvar.append(rand_cvar)
            l_cvar, l_p90 = np.array(l_cvar), np.array(l_p90)
            # bootstrap CI of the median CVaR lift
            bs = [np.median(rng.choice(l_cvar, len(l_cvar))) for _ in range(2000)]
            w_p = wilcoxon(l_cvar).pvalue if np.any(l_cvar != 0) else 1.0
            rows.append(dict(pot=name, col=col, p=p, retention=np.median(ret),
                             kept_cvar=np.median(k_cvar), rand_cvar=np.median(r_cvar),
                             lift_cvar=np.median(l_cvar), lift_cvar_lo=np.percentile(bs, 2.5),
                             lift_cvar_hi=np.percentile(bs, 97.5), lift_p90=np.median(l_p90),
                             wilcoxon_p=w_p, n_groups=len(l_cvar)))
    res = pd.DataFrame(rows)
    res.to_csv(CACHE / "_thesis_pogs_filter_eval.csv", index=False)

    print("\n--- best CVaR-lift operating point per potential ---")
    for name, col, _ in POTS:
        r = res[res.col == col]
        b = r.loc[r.lift_cvar.idxmax()]
        print(f"  {name:18s}: p*={b.p:.2f}  retention={b.retention:.2f}  "
              f"kept_CVaR={b.kept_cvar:.3f} vs rand_CVaR={b.rand_cvar:.3f}  "
              f"lift={b.lift_cvar:+.3f} [{b.lift_cvar_lo:+.3f},{b.lift_cvar_hi:+.3f}]  "
              f"Wilcoxon p={b.wilcoxon_p:.1e}")

    # ---- figure: (a) CVaR operating curve kept vs random, (b) CVaR lift +/- CI ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, col, c in POTS:
        r = res[res.col == col].sort_values("retention")
        ax1.plot(r.retention, r.kept_cvar, "-o", ms=3, color=c, label=f"{name} (TANDEM)")
        ax1.plot(r.retention, r.rand_cvar, "--", color=c, alpha=0.6, label=f"{name} (random)")
        ax2.plot(r.p, r.lift_cvar, "-o", ms=3, color=c, label=name)
        ax2.fill_between(r.p, r.lift_cvar_lo, r.lift_cvar_hi, color=c, alpha=0.15)
    ax1.set_xlabel("retention (fraction of candidates kept)")
    ax1.set_ylabel("CVaR@10% of kept $d_{\\mathrm{PoGS}}$ (worst-tail)")
    ax1.set_title("(a) Tail-distance vs retention: TANDEM vs size-matched random",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("keep top-$p$ of potential probability mass")
    ax2.set_ylabel("CVaR@10% lift over random  (random $-$ TANDEM)")
    ax2.set_title("(b) Outlier-removal lift over random ($\\pm$95% CI)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    fig.suptitle(f"TANDEM-A as a PoGS-feasibility filter: tail removal, within fixed count "
                 f"({len(groups)} groups)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq5_pogs_filter_eval.pdf")


if __name__ == "__main__":
    main()
