"""RQ2(c) -- supporting count for section 5.2.3.

Fills the TODO in chapters/results.tex that asks for the median number of
single-position mutants per peptide over the 66 peptides used in the eps scan
(6 benchmark seeds + the 60 sampled Veltri-positive peptides recorded in
``_thesis_rq2_eps.csv``). The Jacobian is computed exactly (no finite-difference
step) at production thresholds kappa=1e-3, theta_mut=1e-6, and the per-peptide
mutation set is enumerated once.

Outputs:
    results/data/all_in/_cache/_thesis_rq2_eps_setsize.csv

Prints the median, IQR, and full distribution so the LaTeX text can be filled in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PROJECT_ROOT, CACHE

sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.mutang import stability as st  # noqa: E402

PROD_KAPPA = 1e-3
PROD_TMUT = 1e-6


def main():
    eps_csv = CACHE / "_thesis_rq2_eps.csv"
    sampled = (pd.read_csv(eps_csv, usecols=["peptide"])["peptide"]
               .drop_duplicates().tolist())
    benchmarks = list(st.BENCHMARK_PEPTIDES.values())
    peptides = list(dict.fromkeys(list(benchmarks) + sampled))
    print(f"benchmarks: {len(benchmarks)}  sampled: {len(sampled)}  "
          f"unique combined: {len(peptides)}")

    model = st.load_model(device="cpu")
    print(f"HydrAMP loaded | latent {model.latent_dim} | ambient {model.ambient_dim}")

    rows = []
    for seq in peptides:
        z = st.encode(model, seq)
        jac_exact = st.jacobian_strict_np(model, z)
        U_e, S_e = st.svd_of_jacobian(jac_exact)
        exact_set = st.enumerate_set(seq, S_e, U_e, PROD_KAPPA, PROD_TMUT)
        rows.append({"peptide": seq, "length": len(seq), "n_single": len(exact_set)})
        if len(rows) % 10 == 0:
            print(f"  ... {len(rows)}/{len(peptides)}")
    df = pd.DataFrame(rows)
    out = CACHE / "_thesis_rq2_eps_setsize.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    n = df["n_single"]
    print("\n=== single-position mutants per peptide "
          "(exact Jacobian, production kappa=1e-3, theta_mut=1e-6) ===")
    print(f"  n peptides       : {len(n)}")
    print(f"  median           : {int(round(np.median(n)))}")
    print(f"  IQR (25%/75%)    : {int(round(np.percentile(n, 25)))} / "
          f"{int(round(np.percentile(n, 75)))}")
    print(f"  mean +/- std     : {n.mean():.1f} +/- {n.std():.1f}")
    print(f"  min / max        : {int(n.min())} / {int(n.max())}")
    print(f"  total enumerated : {int(n.sum())}")


if __name__ == "__main__":
    main()
