"""First-order net-displacement (superposition) outlier filter -- local test.

Each single mutation i has latent displacement delta_i = enc(parent_with_only_i) - enc(parent).
A multi-mutation candidate's net displacement is approximated by || sum_i delta_i || (first-order
superposition), computed WITHOUT re-encoding the exponential candidate set -- only the O(L*A) single
mutants are encoded per parent. We test (a) how well this proxy tracks the true dist_eucl and the
geodesic dist_pogs, and (b) its outlier-detection AUROC vs TANDEM and vs the exact dist_eucl.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIM = int(os.environ.get("SP_LIMIT", 0))


def roc_auc_score(y, score):
    """AUROC via the rank statistic (no sklearn dependency)."""
    y = np.asarray(y); score = np.asarray(score, float)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
INPUT = os.environ.get("SP_INPUT", "_thesis_rq5_distances_pogs.parquet")
OUTPUT = os.environ.get("SP_OUTPUT", "_thesis_superpos_filter.parquet")


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6, field_eps=1e-6, device=DEV)
    hyd.eval()
    df = pd.read_parquet(CACHE / INPUT).reset_index(drop=True)
    has_tandem = "score_A_onehot" in df.columns
    print(f"[sp] input={INPUT}  TANDEM cols={'yes' if has_tandem else 'no'}")
    peps = list(dict.fromkeys(df["pep"]))
    if LIM:
        peps = peps[:LIM]; df = df[df.pep.isin(peps)].reset_index(drop=True)
    print(f"[sp] {len(df):,} candidates over {len(peps)} parents | device={DEV}")

    pred = np.full(len(df), np.nan)
    idx_by_pep = df.groupby("pep").indices
    for ip, pep in enumerate(peps):
        rows = idx_by_pep[pep]
        seqs = df["seq"].values[rows]
        # collect all single (pos,aa) edits appearing in this parent's candidates
        singles = {}
        cand_muts = []
        for s in seqs:
            muts = tuple((i, s[i]) for i in range(min(len(s), len(pep))) if s[i] != pep[i])
            cand_muts.append(muts)
            for m in muts:
                singles.setdefault(m, None)
        single_keys = list(singles)
        single_seqs = []
        for (i, aa) in single_keys:
            ls = list(pep); ls[i] = aa; single_seqs.append("".join(ls))
        with torch.no_grad():
            z0 = hyd.encode_peptides([pep]).detach().cpu().numpy()[0]
            if single_seqs:
                zz = hyd.encode_peptides(single_seqs).detach().cpu().numpy()
            else:
                zz = np.zeros((0, z0.shape[0]))
        delta = {k: zz[j] - z0 for j, k in enumerate(single_keys)}
        for r, muts in zip(rows, cand_muts):
            if muts:
                pred[r] = np.linalg.norm(np.sum([delta[m] for m in muts], axis=0))
            else:
                pred[r] = 0.0
        if (ip + 1) % 50 == 0:
            print(f"  ...{ip + 1}/{len(peps)}", flush=True)
    df["pred_disp"] = pred
    df.to_parquet(CACHE / OUTPUT)

    # (a) how well does the cheap proxy track the true distances? (within fixed count)
    print("\n=== superposition proxy vs true distance (median per-peptide Spearman, within count) ===")
    for nm in (1, 2, 3, 4):
        sub = df[df.n_mut == nm]
        re, rp = [], []
        for _, g in sub.groupby("pep"):
            if len(g) < 8 or g.pred_disp.nunique() < 2:
                continue
            re.append(spearmanr(g.pred_disp, g.dist_eucl).correlation)
            rp.append(spearmanr(g.pred_disp, g.dist_pogs).correlation)
        print(f"  n={nm}: rho(pred, dist_eucl)={np.median(re):+.3f}   rho(pred, dist_pogs)={np.median(rp):+.3f}")

    # (b) outlier-detection AUROC (top-decile dist_pogs), within and across count
    print("\n=== outlier-detection AUROC (predict top-decile far d_pogs) ===")
    def auroc(col, within):
        if within:
            out = []
            for nm in (2, 3, 4):
                au = []
                for _, g in df[df.n_mut == nm].groupby("pep"):
                    if len(g) < 12 or g[col].nunique() < 2:
                        continue
                    y = (g.dist_pogs >= np.quantile(g.dist_pogs, 0.9)).astype(int)
                    if 0 < y.sum() < len(y):
                        au.append(roc_auc_score(y, g[col].to_numpy()))
                out.append(np.median(au) if au else np.nan)
            return out
        else:
            au = []
            for _, g in df.groupby("pep"):
                if len(g) < 20 or g[col].nunique() < 2:
                    continue
                y = (g.dist_pogs >= np.quantile(g.dist_pogs, 0.9)).astype(int)
                if 0 < y.sum() < len(y):
                    au.append(roc_auc_score(y, g[col].to_numpy()))
            return np.median(au) if au else np.nan
    cols = [("pred_disp (superpos)", "pred_disp"), ("dist_eucl (exact)", "dist_eucl")]
    if has_tandem:
        df["neg_tandem"] = -df["score_A_onehot"]
        cols.append(("-TANDEM-A one-hot", "neg_tandem"))
    print("  predictor              within n=2/3/4            across-count")
    for label, col in cols:
        w = auroc(col, True); a = auroc(col, False)
        print(f"  {label:22s} {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}          {a:.3f}")


if __name__ == "__main__":
    main()
