"""Reconstruction (self-consistency) filter for the RQ5 candidates -- local.

A MUTANG candidate is "reconstructing" if HydrAMP round-trips it exactly:
    argmax(Dec(enc(seq))) == seq   (every residue recovered).
Candidates that do not reconstruct are off the decoder manifold (the model cannot faithfully
represent them). We add a ``reconstructs`` flag to the RQ5 PoGS cache, report the reconstructing
fraction (overall and by mutation count), and re-run the TANDEM-A alignment + magnitude
outlier-detection on the reconstructing subset vs the full set, under d_PoGS.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = int(os.environ.get("RC_BATCH", 2048))


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6, field_eps=1e-6, device=DEV)
    hyd.eval()
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances_pogs.parquet").reset_index(drop=True)
    uniq = pd.unique(df["seq"])
    print(f"[rc] {len(df):,} candidates, {len(uniq):,} unique seqs | device={DEV}")

    recon = {}
    for i in range(0, len(uniq), BATCH):
        chunk = list(uniq[i:i + BATCH])
        with torch.no_grad():
            z = hyd.encode_peptides(chunk).detach()
            dec = hyd.decode_peptides(z, batch_size=BATCH)
        for s, d in zip(chunk, dec):
            recon[s] = (d == s)
        if (i // BATCH + 1) % 10 == 0:
            print(f"  ...{min(i + BATCH, len(uniq))}/{len(uniq)}", flush=True)
    df["reconstructs"] = df["seq"].map(recon).astype(bool)
    df.to_parquet(CACHE / "_thesis_rq5_distances_pogs_recon.parquet")

    print("\n=== reconstructing fraction (argmax(Dec(enc(seq)))==seq) ===")
    for nm in (1, 2, 3, 4, "all"):
        sub = df if nm == "all" else df[df.n_mut == nm]
        print(f"  n_mut={str(nm):>3}: {sub.reconstructs.mean()*100:5.1f}%  ({int(sub.reconstructs.sum()):,}/{len(sub):,})")

    def fc(d, col, nm, dist="dist_pogs"):
        r = []
        for _, g in d[d.n_mut == nm].groupby("pep"):
            if len(g) < 6 or g[col].nunique() < 2 or g[dist].nunique() < 2:
                continue
            v = spearmanr(g[col], -g[dist]).correlation
            if np.isfinite(v):
                r.append(v)
        return np.median(r) if r else np.nan

    rec = df[df.reconstructs]
    print("\n=== TANDEM-A fixed-count Spearman(potential, -d_pogs): FULL vs RECON subset ===")
    for col in ("score_A_onehot", "score_A_diff"):
        print(f"  {col}")
        for nm in (2, 3, 4):
            print(f"    n={nm}: full={fc(df,col,nm):+.3f}   recon={fc(rec,col,nm):+.3f}")

    print("\n=== outlier AUROC (top-decile d_pogs, across count, >=20/peptide): FULL vs RECON ===")
    def auroc(d, col, sign):
        au = []
        for _, g in d.groupby("pep"):
            if len(g) < 20 or g[col].nunique() < 2:
                continue
            y = (g.dist_pogs >= np.quantile(g.dist_pogs, 0.9)).astype(int)
            if 0 < y.sum() < len(y):
                au.append(roc_auc_score(y, sign * g[col].to_numpy()))
        return np.median(au) if au else np.nan
    for label, col, sign in [("dist_eucl (magnitude)", "dist_eucl", +1),
                             ("-TANDEM-A one-hot", "score_A_onehot", -1)]:
        print(f"  {label:22s} full={auroc(df,col,sign):.3f}   recon={auroc(rec,col,sign):.3f}")


if __name__ == "__main__":
    main()
