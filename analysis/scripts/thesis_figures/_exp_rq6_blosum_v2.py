"""RQ4 helper (run on bury): build model-induced AMP-BLOSUM *count* matrices.

CPU-only (encode + approx Jacobian + SVD + substitution counting; no geodesics, no GPU).
Saves three 20x20 discrete Henikoff log-odds matrices and their raw counts to CACHE, for the
local fig_rq6_blosum.py to compare against BLOSUM62 and the DBAASP AMP-BLOSUM and plot:

  * MUTANG (no selection)   -- all enumerated single+combined candidates of each parent.
  * TANDEM-B, top-25%       -- top quartile by the pullback-cosine (Variant B) potential.
  * MUTANG+, argmax, tau=0.2-- whitened (Variant A onehot) viability: for each single-position
                               mutation (anchor), add the best-aligned residue at every other
                               position whose whitened cosine to the anchor exceeds tau.

Sample: N_BLOSUM_PEPS Veltri-positive parents (same pool/seed as fig_rq56_potentials).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fig_rq56_potentials import (  # noqa: E402
    HydrAMPEncoderDecoder, ALL_AA, ALPHABET, MAXLEN, N_BLOSUM_PEPS,
    setup, discrete_blosum, POTS,
)
from _exp_mutangplus_fulldist import parent_geometry, seq_of  # noqa: E402

IDX = {a: i for i, a in enumerate(ALL_AA)}
TAU = float(os.environ.get("RQ6_TAU", 0.2))
N = int(os.environ.get("RQ6_N", N_BLOSUM_PEPS))


def count_subs(counts, parent, seq):
    for a, b in zip(parent, seq):
        if a in IDX and b in IDX:
            counts[IDX[a], IDX[b]] += 1


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device="cpu")
    hyd.eval()
    pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")["sequence"]
    pool = pool.dropna().astype(str)
    pool = pool[(pool.str.len() <= MAXLEN) & (pool.str.len() > 3)
                & pool.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates()
    peps = pool.sample(n=min(N, len(pool)), random_state=0).tolist()
    print(f"[rq6v2] {len(peps)} parents, tau={TAU}")

    c_mutang = np.zeros((20, 20)); c_tandemB = np.zeros((20, 20)); c_mplus = np.zeros((20, 20))
    n_used = 0
    for ip, pep in enumerate(peps):
        try:
            z, ts, muts, _ = setup(hyd, pep)
            if len(muts) < 2:
                continue
            # --- MUTANG (all candidates) + TANDEM-B top-25%, via the Variant-B potential ranking
            scB = POTS["B_onehot"](ts).compute_with_identities(pep, muts)
            if not scB:
                continue
            positions = sorted(muts.keys()); padded = pep.ljust(MAXLEN)
            items = sorted(scB.items(), key=lambda kv: -kv[1])
            seqs = ["".join(ALPHABET[k[positions.index(p)]] if p in positions else padded[p]
                            for p in range(len(pep))) for k, _ in items]
            for s in seqs:                                   # MUTANG = no selection
                count_subs(c_mutang, pep, s)
            for s in seqs[:max(1, round(0.25 * len(seqs)))]: # TANDEM-B top-25%
                count_subs(c_tandemB, pep, s)
            # --- MUTANG+ argmax at tau (whitened Variant-A onehot)
            _, _, mp_muts, mp_pos, parent_aa, labels, Voh, _ = parent_geometry(hyd, pep, "cpu")
            if len(labels) >= 2:
                S = Voh @ Voh.T
                pos_items = {}
                for k, (l, a) in enumerate(labels):
                    pos_items.setdefault(l, []).append((k, a))
                for i, (li, ai) in enumerate(labels):
                    chosen = {li: ai}
                    for l, its in pos_items.items():
                        if l == li:
                            continue
                        bs, ba = max(((float(S[i, k]), a) for k, a in its), key=lambda x: x[0])
                        if bs > TAU:
                            chosen[l] = ba
                    count_subs(c_mplus, pep, seq_of(pep, chosen))
            n_used += 1
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {pep[:12]} ({ex})")
        if (ip + 1) % 25 == 0:
            print(f"  ...{ip + 1}/{len(peps)}")

    print(f"[rq6v2] used {n_used} parents")
    out = {"mutang": c_mutang, "tandemB_top25": c_tandemB, f"mutangplus_argmax_tau{TAU}": c_mplus}
    for name, c in out.items():
        pd.DataFrame(c, index=ALL_AA, columns=ALL_AA).to_csv(CACHE / f"_thesis_rq6v2_counts_{name}.csv")
        pd.DataFrame(discrete_blosum(c), index=ALL_AA, columns=ALL_AA).to_csv(
            CACHE / f"_thesis_rq6v2_blosum_{name}.csv")
        print(f"  saved {name}: {int(c.sum())} substitutions")


if __name__ == "__main__":
    main()
