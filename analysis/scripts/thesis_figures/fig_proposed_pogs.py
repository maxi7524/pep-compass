"""Proposed magnitude-faithful potential V = -||sum_i v_i|| as a selection rule vs the PoGS distance.

Companion (right panel) to the TANDEM selection figure rq5_tandem_pogs.pdf: keep the p-fraction of
candidates with the SMALLEST proposed net-displacement and plot, against the kept fraction, the
median-over-peptides mean d_PoGS of the kept set AND its farthest-5% tail. Unlike TANDEM (whose mean
barely moves and whose far tail is immovable), both the mean and the far tail fall monotonically.

Realised with pred_disp = ||sum_i delta_i|| (delta_i = enc(parent_with_only_i) - enc(parent)); read
from the cached superposition parquet -- no recompute. Saves rq5_proposed_pogs.pdf.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CACHE, THESIS_FIG

# RECON=1 restricts to candidates that pass the HydrAMP recovery test (on-manifold) before selecting.
RECON = bool(int(os.environ.get("RECON", "0")))
df = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")
if RECON:
    rc = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs_recon.parquet")[["pep", "seq", "reconstructs"]]
    df = df.merge(rc, on=["pep", "seq"], how="inner")
    df = df[df["reconstructs"]].reset_index(drop=True)
    OUT_NAME = "rq5_proposed_pogs_recon.pdf"
    TITLE_SUFFIX = " (reconstructing mutants only)"
else:
    OUT_NAME = "rq5_proposed_pogs.pdf"
    TITLE_SUFFIX = ""
P = np.round(np.arange(0.05, 1.0 + 1e-9, 0.025), 3)
P_TABLE = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]


def _far5(a):
    m = max(1, int(np.ceil(0.05 * len(a))))
    return np.sort(a)[-m:].mean()


def _close5(a):
    m = max(1, int(np.ceil(0.05 * len(a))))
    return np.sort(a)[:m].mean()


def select(p):
    """median-over-peptides (mean, median, close5, far5) of the kept closest-p set."""
    mm, md, lo, ff = [], [], [], []
    for _, g in df.groupby("pep"):
        if len(g) < 8:
            continue
        s = g.sort_values("pred_disp", ascending=True)
        k = min(max(int(np.ceil(p * len(s))), 1), len(s))
        d = s["dist_pogs"].to_numpy()[:k]
        mm.append(d.mean()); md.append(np.median(d)); lo.append(_close5(d)); ff.append(_far5(d))
    return np.median(mm), np.median(md), np.median(lo), np.median(ff)


def main():
    mean_c = np.array([select(p)[0] for p in P])
    far_c = np.array([select(p)[3] for p in P])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(P, mean_c, marker="o", ms=3, color="C0", label="kept-set mean")
    ax.plot(P, far_c, marker="s", ms=3, ls="--", color="C3", label="kept-set farthest 5\\%")
    ax.axhline(mean_c[-1], color="0.6", ls=":", lw=1, label="full set (mean)")
    ax.set_xlabel("keep fraction $p$ of candidates (smallest net displacement)")
    ax.set_ylabel("median (over peptides) $d_{\\mathrm{PoGS}}$ of kept set")
    ax.set_title(f"Proposed potential top-$p$ selection vs PoGS distance "
                 f"({df.pep.nunique()} peptides{TITLE_SUFFIX})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = THESIS_FIG / OUT_NAME
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}  ({len(df):,} candidates, {df.pep.nunique()} peptides)")
    print(f"  {'p':>5} {'mean':>7} {'median':>7} {'close5':>7} {'far5':>7}")
    for p in P_TABLE:
        mn, md, lo, fr = select(p)
        print(f"  {p:>5.2f} {mn:>7.3f} {md:>7.3f} {lo:>7.3f} {fr:>7.3f}")


if __name__ == "__main__":
    main()
