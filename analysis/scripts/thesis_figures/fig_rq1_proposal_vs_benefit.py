"""RQ1 -- does MUTANG preferentially propose biologically beneficial substitutions?

Cross-analysis joining the two 20x20 matrices produced by the other RQ1 scripts:
  * proposal frequency  P(to | from)   -- how often MUTANG enumerates a substitution
                                           (fig_rq1_mutang_proposal.py)
  * benefit  b = -mean DlogMIC (Gram-)  -- how much the substitution improves activity,
                                           sign-flipped so higher = lowers MIC = better
                                           (fig_rq1_biological_signature.py)

If MUTANG's proposals were biologically informed, frequently-proposed substitutions would
tend to be the beneficial ones (positive correlation). Writes:
    rq1_proposal_vs_benefit.pdf
and prints Spearman / Pearson and a top- vs bottom-proposed benefit comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

from _common import CACHE, ALL_AA, save


def load_matrices():
    proposal = pd.read_parquet(CACHE / "_thesis_mutang_proposal_neg.parquet")
    counts = pd.read_parquet(CACHE / "_thesis_mutang_proposal_counts_neg.parquet")
    benefit_dlog = pd.read_parquet(CACHE / "_thesis_benefit_neg_gramneg.parquet")
    return proposal, counts, benefit_dlog


def main():
    proposal, counts, benefit_dlog = load_matrices()
    rows = []
    for a in ALL_AA:
        for b in ALL_AA:
            if a == b:
                continue
            p = float(proposal.at[a, b])
            d = float(benefit_dlog.at[a, b])
            if not np.isfinite(d):
                continue
            rows.append({"from": a, "to": b, "pair": f"{a}>{b}",
                         "proposal": p, "benefit": -d, "dlogmic": d})
    df = pd.DataFrame(rows)
    proposed = df[df["proposal"] > 0].copy()

    rho_all, p_all = spearmanr(df["proposal"], df["benefit"])
    rho_pr, p_pr = spearmanr(proposed["proposal"], proposed["benefit"])
    r_pr, rp_pr = pearsonr(proposed["proposal"], proposed["benefit"])
    print(f"All {len(df)} pairs w/ finite benefit:  Spearman={rho_all:+.3f} (p={p_all:.2g})")
    print(f"Only {len(proposed)} actually-proposed pairs: "
          f"Spearman={rho_pr:+.3f} (p={p_pr:.2g})  Pearson={r_pr:+.3f} (p={rp_pr:.2g})")

    # top vs bottom proposed-frequency tertiles, benefit comparison
    q_hi = proposed["proposal"].quantile(2 / 3)
    q_lo = proposed["proposal"].quantile(1 / 3)
    hi = proposed[proposed["proposal"] >= q_hi]["benefit"]
    lo = proposed[proposed["proposal"] <= q_lo]["benefit"]
    u, pu = mannwhitneyu(hi, lo, alternative="two-sided")
    print(f"mean benefit  top-third proposed = {hi.mean():+.4f} (n={len(hi)})  "
          f"bottom-third = {lo.mean():+.4f} (n={len(lo)})  Mann-Whitney p={pu:.2g}")

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = 30 + 3000 * proposed["proposal"]
    sc = ax.scatter(proposed["proposal"], proposed["benefit"], s=sizes,
                    c=proposed["benefit"], cmap="RdYlGn", vmin=-0.15, vmax=0.15,
                    edgecolor="#444", linewidth=0.5, alpha=0.85)
    ax.axhline(0.0, color="#9aa0a6", lw=1.0, ls=":")
    # label the most-proposed and the most-beneficial pairs
    label_set = set(proposed.nlargest(8, "proposal")["pair"]) | \
        set(proposed.nlargest(5, "benefit")["pair"]) | \
        set(proposed.nsmallest(5, "benefit")["pair"])
    for _, r in proposed.iterrows():
        if r["pair"] in label_set:
            ax.annotate(r["pair"], (r["proposal"], r["benefit"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8, color="#222")
    ax.set_xlabel("MUTANG proposal frequency  P(mutate to | from)", fontsize=11)
    ax.set_ylabel("Activity benefit  $-\\overline{\\Delta\\log\\mathrm{MIC}}$  "
                  "(higher = lowers MIC = better)", fontsize=11)
    ax.set_title("Does MUTANG propose the beneficial substitutions?\n"
                 f"Spearman $\\rho$ = {rho_pr:+.2f} (p = {p_pr:.2g}), "
                 f"{len(proposed)} proposed AA$\\to$AA pairs",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("benefit  $-\\overline{\\Delta\\log\\mathrm{MIC}}$")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "rq1_proposal_vs_benefit.pdf")


if __name__ == "__main__":
    main()
