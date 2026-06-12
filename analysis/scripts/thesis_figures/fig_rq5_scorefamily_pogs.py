"""Score-family alignment heatmap under the PoGS geodesic distance (local).

The PoGS analogue of fig_rq7_scorings.py's `rq5_scorefamily_alignment.pdf`: the same score family
(2 similarities x 8 transforms x 2 aggregations + 2 hard reject filters) but correlated with
feasibility $-d_{\mathrm{PoGS}}$ ($\lambda=0$) instead of $-d_{\mathrm{geo}}$. The family scores are
already saved per candidate in _thesis_rq7_scores.parquet; PoGS distances are in
_thesis_rq5_distances_pogs.parquet. Both carry dist_geo, which is a unique per-candidate key within
a (pep, n_mut) group, so we attach dist_pogs by an exact merge on (pep, n_mut, dist_geo) -- no
recomputation. Writes rq5_scorefamily_alignment_pogs.pdf (and a CSV of the numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from _common import CACHE, save

TRANSFORMS = ["log", "linear", "hinge", "quad", "count", "rational", "exp", "sigmoid"]
AGGS = ["mean", "min"]
SIMS = ["A", "B"]
REJECTS = ["reject.3", "reject.5"]
SCORE_COLS = ([f"{s}/{t}/{a}" for s in SIMS for t in TRANSFORMS for a in AGGS]
              + [f"{s}/{r}" for s in SIMS for r in REJECTS])
NMUTS = (2, 3, 4)


def main():
    sc = pd.read_parquet(CACHE / "_thesis_rq7_scores_alt.parquet")  # 8-transform reuse-mode scores
    di = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs.parquet")
    # attach dist_pogs via the unique (pep, n_mut, dist_geo) key
    for d in (sc, di):
        d["gk"] = d["dist_geo"].round(9)
    di2 = di[["pep", "n_mut", "gk", "dist_pogs"]].drop_duplicates(["pep", "n_mut", "gk"])
    df = sc.merge(di2, on=["pep", "n_mut", "gk"], how="left")
    matched = df["dist_pogs"].notna().mean()
    df = df[df["dist_pogs"].notna()].reset_index(drop=True)
    print(f"score rows={len(sc):,}  matched dist_pogs={matched:.3f}  used={len(df):,} "
          f"peptides={df.pep.nunique()}")

    def med(col, nm, dist):
        v = []
        for _, s in df[df.n_mut == nm].groupby("pep"):
            if len(s) >= 8 and s[col].std() > 0 and s[dist].std() > 0:
                v.append(spearmanr(s[col], -s[dist]).correlation)
        return float(np.median(v)) if v else np.nan

    rows = []
    print("\n[scorefamily/PoGS] median per-peptide Spearman(score, -d_pogs)")
    print(f"  {'variant':22s} | {'nmut2':>7} {'nmut3':>7} {'nmut4':>7}")
    for col in SCORE_COLS:
        r = {nm: med(col, nm, "dist_pogs") for nm in NMUTS}
        rows.append({"variant": col, **{f"nmut{k}": v for k, v in r.items()}})
        print(f"  {col:22s} | {r[2]:+7.3f} {r[3]:+7.3f} {r[4]:+7.3f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(CACHE / "_thesis_rq5_scorefamily_pogs.csv", index=False)

    # extremes for the thesis caption/prose
    flat = tab.melt("variant", value_name="rho")
    hi = flat.loc[flat.rho.idxmax()]
    print(f"\n  strongest cell: {hi.variant} {hi['variable']} rho={hi.rho:+.3f}")
    print(f"  range over all cells: [{flat.rho.min():+.3f}, {flat.rho.max():+.3f}]")

    H = tab.set_index("variant")[["nmut2", "nmut3", "nmut4"]]
    fig, ax = plt.subplots(figsize=(6.8, 12))
    sns.heatmap(H, annot=True, fmt="+.2f", center=0, cmap="vlag", vmin=-0.4, vmax=0.4,
                cbar_kws={"label": r"median Spearman(score, $-d_{\mathrm{PoGS}}$)"}, ax=ax)
    ax.set_title(f"scoring alignment with PoGS feasibility (n={df.pep.nunique()})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("within mutation count"); ax.set_ylabel("similarity / transform / aggregation")
    fig.tight_layout()
    save(fig, "rq5_scorefamily_alignment_pogs.pdf")


if __name__ == "__main__":
    main()
