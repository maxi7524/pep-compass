"""Head-to-head: the proposed magnitude-faithful potential V = -||sum_i v_i|| vs TANDEM-A.

The proposed potential is the (negative) norm of the summed single-mutation latent displacements --
TANDEM's per-mutation geometry, but un-normalised, summed, and sign-corrected so that cancelling
(opposed) mutations are rewarded. Here it is realised by ``pred_disp`` = ||sum_i delta_i|| with
delta_i = enc(parent_with_only_i) - enc(parent) (the net-displacement / superposition proxy).

We reproduce, for BOTH potentials, the exact diagnostics TANDEM failed:
  (1) fixed-count median Spearman(potential, -d_PoGS);
  (2) far-tail (top-decile d_PoGS) detection AUROC, across count;
  (3) the kept-set distance DISTRIBUTION under top-p selection (mean / median / closest-5% / farthest-5%),
      the table where TANDEM neither trimmed the far tail nor concentrated the close candidates.
For (3) each potential keeps the p-fraction of candidates it deems most feasible: highest TANDEM
score (softmax-mass nucleus, as in the thesis) vs smallest proposed net-displacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from _common import CACHE

df = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")
P_TABLE = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _tail(a, frac, top):
    m = max(1, int(np.ceil(frac * len(a))))
    s = np.sort(a)
    return s[-m:].mean() if top else s[:m].mean()


def fixed_count(col, sign):
    print(f"  {col} (sign {sign:+d}) fixed-count median rho(.,-d_PoGS):")
    for nm in (2, 3, 4):
        r = []
        for _, g in df[df.n_mut == nm].groupby("pep"):
            if len(g) < 6 or g[col].nunique() < 2 or g.dist_pogs.nunique() < 2:
                continue
            v = spearmanr(sign * g[col], -g.dist_pogs).correlation
            if np.isfinite(v):
                r.append(v)
        print(f"    n={nm}: {np.median(r):+.3f}")


def tail_auroc(col, sign):
    a = []
    for _, g in df.groupby("pep"):
        if len(g) < 20 or g[col].nunique() < 2:
            continue
        y = (g.dist_pogs >= np.quantile(g.dist_pogs, 0.9)).astype(int)
        if 0 < y.sum() < len(y):
            a.append(auroc(y, sign * g[col].to_numpy()))
    return np.nanmedian(a)


def kept_distribution(mode):
    """mode='tandem' keeps highest score_A_diff by softmax mass; mode='proposed' keeps smallest
    pred_disp (closest predicted) by count fraction."""
    rows = []
    for p in P_TABLE:
        mean_, med_, lo_, hi_, ret_ = [], [], [], [], []
        for _, g in df.groupby("pep"):
            if len(g) < 8:
                continue
            if mode == "tandem":
                col = "score_A_diff"
                if g[col].nunique() < 2:
                    continue
                s = g.sort_values(col, ascending=False)
                w = np.exp(s[col].to_numpy() - s[col].max()); w /= w.sum()
                k = int(np.searchsorted(np.cumsum(w), p) + 1)
            else:  # proposed: keep the p-fraction with smallest net displacement
                s = g.sort_values("pred_disp", ascending=True)
                k = int(np.ceil(p * len(s)))
            k = min(max(k, 1), len(s))
            d = s["dist_pogs"].to_numpy()[:k]
            mean_.append(d.mean()); med_.append(np.median(d))
            lo_.append(_tail(d, 0.05, top=False)); hi_.append(_tail(d, 0.05, top=True))
            ret_.append(k / len(s))
        rows.append((p, np.median(ret_), np.median(mean_), np.median(med_),
                     np.median(lo_), np.median(hi_)))
    return rows


def main():
    print(f"{len(df):,} candidates, {df.pep.nunique()} peptides\n")
    print("(1) fixed-count alignment with feasibility:")
    fixed_count("score_A_diff", +1)
    fixed_count("pred_disp", -1)   # feasibility = -net displacement
    print("\n(2) far-tail detection AUROC (across count):")
    print(f"  TANDEM-A diff            : {tail_auroc('score_A_diff', +1):.3f}")
    print(f"  proposed -||sum v_i||    : {tail_auroc('pred_disp', +1):.3f}")
    print("\n(3) kept-set d_PoGS distribution under top-p selection:")
    for mode, label in (("tandem", "TANDEM-A diff (highest score)"),
                        ("proposed", "proposed (smallest net displacement)")):
        print(f"\n  {label}")
        print(f"    {'p':>5} {'retain':>7} {'mean':>7} {'median':>7} {'close5%':>8} {'far5%':>7}")
        for p, ret, mn, md, lo, hi in kept_distribution(mode):
            print(f"    {p:>5.2f} {ret:>7.2f} {mn:>7.3f} {md:>7.3f} {lo:>8.3f} {hi:>7.3f}")


if __name__ == "__main__":
    main()
