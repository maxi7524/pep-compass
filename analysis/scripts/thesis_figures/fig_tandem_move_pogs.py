"""TANDEM and MOVE selection curves on ONE plot (full + reconstructing panels).

Both potentials are built from the same per-mutation directions; here we put their top-p selection
curves on shared axes so the comparison is direct: keeping the top-p of candidates by each method's
score, the median-over-peptides kept-set mean d_PoGS vs p.
  TANDEM -- highest pairwise-coherence potential (soft-max nucleus on score_A_diff);
  MOVE   -- smallest net displacement ||sum_i delta_i|| (pred_disp).
Panel (a) full candidate set, (b) reconstructing candidates only. Saves rq5_tandem_move_pogs.pdf.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CACHE, THESIS_FIG

P = np.round(np.arange(0.05, 1.0 + 1e-9, 0.025), 3)
full = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")
rc = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs_recon.parquet")[["pep", "seq", "reconstructs"]]
recon = full.merge(rc, on=["pep", "seq"], how="inner")
recon = recon[recon["reconstructs"]].reset_index(drop=True)


def tandem_curve(df):
    out = []
    for p in P:
        v = []
        for _, g in df.groupby("pep"):
            if len(g) < 8 or g.score_A_diff.nunique() < 2:
                continue
            s = g.sort_values("score_A_diff", ascending=False)
            w = np.exp(s.score_A_diff.to_numpy() - s.score_A_diff.max()); w /= w.sum()
            k = min(max(int(np.searchsorted(np.cumsum(w), p) + 1), 1), len(s))
            v.append(s.dist_pogs.to_numpy()[:k].mean())
        out.append(np.median(v) if v else np.nan)
    return np.array(out)


def move_curve(df):
    out = []
    for p in P:
        v = []
        for _, g in df.groupby("pep"):
            if len(g) < 8:
                continue
            s = g.sort_values("pred_disp", ascending=True)
            k = min(max(int(np.ceil(p * len(s))), 1), len(s))
            v.append(s.dist_pogs.to_numpy()[:k].mean())
        out.append(np.median(v) if v else np.nan)
    return np.array(out)


def panel(ax, df, title):
    t = tandem_curve(df); m = move_curve(df)
    ax.plot(P, t, marker="o", ms=3, color="C0", label="TANDEM (coherence)")
    ax.plot(P, m, marker="s", ms=3, color="C3", label="MOVE (magnitude)")
    ax.axhline(t[-1], color="0.6", ls=":", lw=1, label="full set")
    jt = int(np.nanargmin(t))
    ax.scatter([P[jt]], [t[jt]], s=70, facecolors="none", edgecolors="C0", linewidths=1.6, zorder=5)
    ax.set_xlabel("keep top-$p$ of candidates")
    ax.set_ylabel("median kept-set $d_{\\mathrm{PoGS}}$")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panel(axes[0], full, f"(a) full candidate set ({full.pep.nunique()} peptides)")
    panel(axes[1], recon, f"(b) reconstructing candidates ({recon.pep.nunique()} peptides)")
    fig.tight_layout()
    out = THESIS_FIG / "rq5_tandem_move_pogs.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")
    for name, df in (("full", full), ("recon", recon)):
        t = tandem_curve(df); m = move_curve(df)
        jt = int(np.nanargmin(t))
        print(f"  {name}: TANDEM min {t[jt]:.2f} at p={P[jt]:.2f} ({100*(1-t[jt]/t[-1]):+.0f}%); "
              f"MOVE at p=0.5 {m[int(np.argmin(np.abs(P-0.5)))]:.2f}, p=0.05 {m[0]:.2f}")


if __name__ == "__main__":
    main()
