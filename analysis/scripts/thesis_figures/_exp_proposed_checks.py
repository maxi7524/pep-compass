"""Two checks on the proposed magnitude-faithful potential V=-||sum_i v_i|| (pred_disp):
  (1) mutation-count confound -- does selecting by it just keep fewer mutations? and does a
      count-normalised variant still beat TANDEM at a FIXED count?
  (2) manifold -- are the candidates it selects actually on the decoder manifold (reconstructing,
      argmax(Dec(enc(seq)))==seq)? Is small net displacement correlated with reconstruction?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from _common import CACHE

sp = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")[["pep", "seq", "n_mut", "pred_disp", "dist_pogs"]]
rc = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs_recon.parquet")[["pep", "seq", "reconstructs"]]
df = sp.merge(rc, on=["pep", "seq"], how="inner")
print(f"merged {len(df):,} candidates over {df.pep.nunique()} peptides "
      f"(recon rate {df.reconstructs.mean()*100:.1f}%)\n")


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s); return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ---- Concern 1: count confound -------------------------------------------------
print("=== (1) mutation-count confound ===")
df["per_mut"] = df["pred_disp"] / df["n_mut"]              # count-normalised variant ||sum v||/n
df["per_rt"]  = df["pred_disp"] / np.sqrt(df["n_mut"])     # ||sum v||/sqrt(n)
# across-count selection by pred_disp: mean n_mut of the kept closest p-fraction
print("  across-count selection by smallest pred_disp -> mean n_mut of kept set:")
for p in (1.0, 0.5, 0.25, 0.1):
    nm = []
    for _, g in df.groupby("pep"):
        if len(g) < 8:
            continue
        s = g.sort_values("pred_disp"); k = max(1, int(np.ceil(p * len(s))))
        nm.append(s["n_mut"].to_numpy()[:k].mean())
    print(f"    p={p:.2f}: mean n_mut kept = {np.median(nm):.2f}")
print("  fixed-count median rho(., -d_PoGS)  [confound-free]:")
for col in ("pred_disp", "per_mut", "per_rt"):
    out = []
    for nm in (2, 3, 4):
        r = []
        for _, g in df[df.n_mut == nm].groupby("pep"):
            if len(g) < 6 or g[col].nunique() < 2 or g.dist_pogs.nunique() < 2:
                continue
            v = spearmanr(-g[col], -g.dist_pogs).correlation
            if np.isfinite(v):
                r.append(v)
        out.append(np.median(r))
    print(f"    {col:10s}: n=2/3/4 = {out[0]:+.3f}/{out[1]:+.3f}/{out[2]:+.3f}")

# ---- Concern 2: manifold -------------------------------------------------------
print("\n=== (2) manifold (do selected candidates reconstruct?) ===")
print(f"  overall recon rate: {df.reconstructs.mean()*100:.1f}%")
print("  recon rate by pred_disp quintile (within peptide), 1=closest:")
def quintile_recon():
    rows = {q: [] for q in range(1, 6)}
    for _, g in df.groupby("pep"):
        if len(g) < 25:
            continue
        q = pd.qcut(g["pred_disp"].rank(method="first"), 5, labels=False)
        for qi in range(5):
            rows[qi + 1].append(g["reconstructs"].to_numpy()[q == qi].mean())
    for qi in range(1, 6):
        print(f"    Q{qi} (pred_disp {'low' if qi==1 else 'high' if qi==5 else 'mid'}): "
              f"recon {np.nanmedian(rows[qi])*100:5.1f}%")
quintile_recon()
print("  fixed-count corr(small pred_disp, reconstructs) [AUROC, recon=positive]:")
for nm in (1, 2, 3, 4):
    au = []
    for _, g in df[df.n_mut == nm].groupby("pep"):
        if len(g) < 8 or g.reconstructs.nunique() < 2:
            continue
        au.append(auroc(g.reconstructs.astype(int), -g.pred_disp.to_numpy()))
    print(f"    n={nm}: AUROC={np.nanmedian(au):.3f}  (recon rate {df[df.n_mut==nm].reconstructs.mean()*100:.0f}%)")
# selection: recon rate among the closest-p by pred_disp vs full
print("  recon rate of the kept closest-p set (across count):")
for p in (1.0, 0.5, 0.25, 0.1):
    rr = []
    for _, g in df.groupby("pep"):
        if len(g) < 8:
            continue
        s = g.sort_values("pred_disp"); k = max(1, int(np.ceil(p * len(s))))
        rr.append(s["reconstructs"].to_numpy()[:k].mean())
    print(f"    p={p:.2f}: recon {np.median(rr)*100:.1f}%")


if __name__ == "__main__":
    pass
