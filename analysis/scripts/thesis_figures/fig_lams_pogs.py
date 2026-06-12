"""LAMS (Latent-Aligned Mutation Selection) on the PoGS geodesic distance -- local replot.

LAMS is the displacement-aware viability filter formerly called MUTANG+: per anchor mutation it
keeps the cross-position partners viable under the *whitened* (Euclidean-latent) similarity, which
retains net-displacement magnitude. We score every retained candidate by the absolute PoGS geodesic
distance d_pogs (lambda=0; ambient chord length of the ADAM-optimised path, no metric G), and report
the mean d_pogs of the retained set and the mean number of positions changed against the viability
threshold tau, for the two selections argmax and product.

Only the DIFF mutation encoding is shown (the parent->target direction); the one-hot encoding lives
in the appendix. Reads the precomputed summary CSV (_thesis_mutangplus_pogs_summary.csv) produced by
_exp_mutangplus_pogs.py on Bury -- no GPU recompute. Saves rq5_lams_pogs.pdf.

To reproduce under a different name, change FILTER_NAME below.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import CACHE, save

FILTER_NAME = "LAMS"                 # display name of the filter (was MUTANG+)
ENC = "diff"                         # only the parent->target difference encoding
SEL_STYLE = {"argmax": dict(ls="-", marker="o", color="C3"),
             "product": dict(ls="--", marker="s", color="C0")}

# RECON=1 plots the variant restricted to candidates that pass the HydrAMP recovery test
# (argmax(Dec(enc(seq)))==seq) -- the recon summary + recon full-pool mean from the Bury job.
RECON = bool(int(os.environ.get("RECON", "0")))
if RECON:
    SUMMARY = "_thesis_mutangplus_pogs_recon_summary.csv"
    OUT_NAME = "rq5_lams_pogs_recon.pdf"
    TITLE_SUFFIX = " (reconstructing mutants only)"
    FULL_POOL_DPOGS = float(os.environ.get("FULL_POOL_DPOGS", "2.27"))  # recon full-pool mean
else:
    SUMMARY = "_thesis_mutangplus_pogs_summary.csv"
    OUT_NAME = "rq5_lams_pogs.pdf"
    TITLE_SUFFIX = ""
    FULL_POOL_DPOGS = float(os.environ.get("FULL_POOL_DPOGS", "2.56"))  # full-pool mean


def main():
    summ = pd.read_csv(CACHE / SUMMARY)
    summ = summ[summ.enc == ENC]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in ("argmax", "product"):
        d_ = summ[summ.method == m].sort_values("tau")
        st = SEL_STYLE[m]
        axes[0].plot(d_["tau"], d_["dabs"], ms=3, lw=1.9, label=f"{FILTER_NAME} / {m}", **st)
        axes[1].plot(d_["tau"], d_["pos"], ms=3, lw=1.9, label=f"{FILTER_NAME} / {m}", **st)
    axes[0].axhline(FULL_POOL_DPOGS, color="0.5", ls=":", lw=1.2,
                    label=f"full MUTANG pool ({FULL_POOL_DPOGS:.2f})")
    axes[0].set_xlabel(r"viability threshold $\tau$")
    axes[0].set_ylabel(r"mean $d_{\mathrm{PoGS}}$ of retained set (absolute)")
    axes[0].set_title("(a) feasibility distance vs $\\tau$", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel(r"viability threshold $\tau$")
    axes[1].set_ylabel("mean \\# positions changed")
    axes[1].set_title("(b) positions changed vs $\\tau$", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"{FILTER_NAME} viability filter vs the absolute PoGS geodesic distance{TITLE_SUFFIX}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, OUT_NAME)

    # report the operating points used in the text
    print(f"[{FILTER_NAME}{TITLE_SUFFIX}] full-pool mean d_PoGS = {FULL_POOL_DPOGS:.2f}")
    for m in ("argmax", "product"):
        d_ = summ[summ.method == m].set_index("tau")
        for tau in (0.0, 0.2, 0.4, 0.6):
            if tau in d_.index:
                r = d_.loc[tau]
                print(f"  {m:7s} tau={tau:+.1f}: pos={r['pos']:.2f}  d_PoGS={r['dabs']:.2f}  "
                      f"rel={r['dabs']/FULL_POOL_DPOGS:.2f}")


if __name__ == "__main__":
    main()
