"""Net-displacement superposition as an OPERATING outlier filter.

Per peptide (across counts), rank candidates by a predictor and DROP the top-q most-suspect; measure
(i) outlier recall = fraction of the true far outliers (top-decile actual d_PoGS) removed, and
(ii) the retained set's mean and worst-5% (CVaR) d_PoGS. Compare dropping by the superposition proxy
||sum delta_i|| (drop largest), by TANDEM coherence (drop least-coherent), and at random.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import CACHE

df = pd.read_parquet(CACHE / "_thesis_superpos_filter.parquet")
QS = [0.10, 0.25, 0.50]
rng = np.random.default_rng(0)


def cvar5(a):
    m = max(1, int(np.ceil(0.05 * len(a))))
    return np.sort(a)[-m:].mean()


def main():
    print(f"{len(df):,} candidates, {df.pep.nunique()} peptides | drop top-q most-suspect, across count")
    # predictor -> suspicion score (higher = more likely an outlier, dropped first)
    preds = {"superpos ||sum d_i||": df["pred_disp"].to_numpy(),
             "TANDEM (-coherence)": -df["score_A_onehot"].to_numpy(),
             "random": None}
    groups = list(df.groupby("pep").indices.items())
    print(f"\n{'predictor':22s} {'q':>5} {'outlier-recall':>15} {'kept mean':>10} {'kept CVaR5%':>12}")
    for name, score in preds.items():
        for q in QS:
            rec, km, kc = [], [], []
            for pep, idx in groups:
                d = df["dist_pogs"].to_numpy()[idx]
                if len(d) < 20:
                    continue
                thr = np.quantile(d, 0.9)
                is_out = d >= thr
                ndrop = max(1, int(round(q * len(d))))
                if score is None:
                    order = rng.permutation(len(d))
                else:
                    order = np.argsort(-score[idx])     # most-suspect first
                dropped = np.zeros(len(d), bool); dropped[order[:ndrop]] = True
                rec.append(is_out[dropped].sum() / max(1, is_out.sum()))
                kept = d[~dropped]
                km.append(kept.mean()); kc.append(cvar5(kept))
            print(f"{name:22s} {q:>5.2f} {np.median(rec)*100:>13.1f}% {np.median(km):>10.3f} "
                  f"{np.median(kc):>12.3f}")
    # full-set reference
    base = [df['dist_pogs'].to_numpy()[idx] for _, idx in groups if len(idx) >= 20]
    print(f"\nfull set: median mean={np.median([b.mean() for b in base]):.3f}  "
          f"median CVaR5%={np.median([cvar5(b) for b in base]):.3f}  (outlier rate=10%)")


if __name__ == "__main__":
    main()
