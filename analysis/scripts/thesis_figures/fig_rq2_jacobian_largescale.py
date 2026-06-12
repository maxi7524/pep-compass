"""RQ2(c) -- large-scale check of the production finite-difference step.

Section 5.2.3 currently reports the approximate-vs-exact Jacobian comparison on
$66$ peptides (6 benchmarks + 60 sampled). This script runs the same comparison
at the production step only (epsilon = 5e-2 vs autograd-exact) on a much larger
sample drawn from the full all_in collection, so the production-default claim
is anchored on thousands of peptides rather than the original 66.

For each sampled peptide we compute, at the production thresholds kappa = 1e-3,
theta_mut = 1e-6:
    - relative Frobenius error of the finite-difference Jacobian against exact
    - relative top-k singular-value error
    - top-k left-singular subspace overlap
    - mutation-set Jaccard between the approx-Jacobian and exact-Jacobian
      single-position enumerations
    - the size of the exact-Jacobian enumeration

Usage:
    # Default: 5000 peptides on the available device, seed=0, eps=5e-2.
    python fig_rq2_jacobian_largescale.py

    # Smaller smoke test, fixed eps:
    RQ2_N=200 RQ2_SEED=0 RQ2_EPS=5e-2 RQ2_DEVICE=cpu \
        python fig_rq2_jacobian_largescale.py

Outputs:
    results/data/all_in/_cache/_thesis_rq2_eps_prod_large.csv
    results/data/all_in/_cache/_thesis_rq2_eps_prod_large_summary.json
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# cuDNN's RNN kernels lack a batched backward used by torch.func.vmap inside
# ``decoder_jacobian_strict``; running with cudnn disabled forces the standard
# autograd path that supports vmap on CUDA. The decoder is small so the
# throughput penalty is negligible.
torch.backends.cudnn.enabled = False

from _common import PROJECT_ROOT, CACHE

# Import the local stability module directly by file path. The default
# ``from scripts.mutang import stability`` route collides on Bury, where a
# ``scripts.mutang`` package without a ``stability`` submodule is installed in
# site-packages and shadows ``analysis/scripts/mutang/`` regardless of
# sys.path order; loading from the file path is collision-free.
_st_path = PROJECT_ROOT / "analysis" / "scripts" / "mutang" / "stability.py"
_spec = _ilu.spec_from_file_location("rq2_stability", _st_path)
st = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(st)  # type: ignore[union-attr]

PROD_KAPPA = 1e-3
PROD_TMUT = 1e-6
PROD_EPS = 5e-2
TOP_K = 10
DATA_DIR = PROJECT_ROOT / "results" / "data" / "all_in"
APEX_CSV = DATA_DIR / "peptides_apex.csv"
# Fallback parquet sources, in order of preference. The first one already
# present on disk is used when ``peptides_apex.csv`` is unavailable (e.g.\
# on Bury). Both expose a ``sequence`` column.
PARQUET_FALLBACKS = [
    DATA_DIR / "_cache" / "parents_ext_veltri_negative_veltri_positive_dbaasp_clean_mic_data.parquet",
    DATA_DIR / "_cache" / "parents_hydramp_veltri_positive.parquet",
]


def pick_device(name: str | None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_sequence_source() -> tuple[pd.Series, str]:
    if APEX_CSV.exists():
        df = pd.read_csv(APEX_CSV, usecols=["sequence"])
        return df["sequence"], f"csv:{APEX_CSV.name}"
    for p in PARQUET_FALLBACKS:
        if p.exists():
            df = pd.read_parquet(p, columns=["sequence"])
            return df["sequence"], f"parquet:{p.name}"
    raise FileNotFoundError(
        f"None of the peptide sources are present: {APEX_CSV} or any of "
        f"{[p.name for p in PARQUET_FALLBACKS]}."
    )


def sample_peptides(n: int, seed: int, max_len: int = 25) -> tuple[list[str], str]:
    series, source = _load_sequence_source()
    seqs = series.dropna().astype(str).str.strip()
    valid = seqs[(seqs.str.len() <= max_len) & (seqs.str.len() > 0)
                 & seqs.apply(lambda s: set(s) <= st.STANDARD_AA)].drop_duplicates()
    if n > len(valid):
        print(f"  WARN: requested N={n} but source has only {len(valid)} valid sequences; "
              "using all of them.")
    return valid.sample(n=min(n, len(valid)), random_state=seed).tolist(), source


def main():
    n = int(os.environ.get("RQ2_N", 5000))
    seed = int(os.environ.get("RQ2_SEED", 0))
    eps = float(os.environ.get("RQ2_EPS", PROD_EPS))
    device = pick_device(os.environ.get("RQ2_DEVICE"))
    print(f"config | N={n} seed={seed} eps={eps:g} kappa={PROD_KAPPA:g} "
          f"theta_mut={PROD_TMUT:g} top_k={TOP_K} device={device}")

    peptides, source = sample_peptides(n, seed)
    print(f"sampled {len(peptides)} unique peptides from {source}")

    model = st.load_model(device=device)
    print(f"HydrAMP loaded | latent {model.latent_dim} | ambient {model.ambient_dim}")

    rows = []
    t0 = time.time()
    for i, seq in enumerate(peptides, start=1):
        z = st.encode(model, seq)
        jac_exact = st.jacobian_strict_np(model, z)
        U_e, S_e = st.svd_of_jacobian(jac_exact)
        exact_set = st.enumerate_set(seq, S_e, U_e, PROD_KAPPA, PROD_TMUT)

        jac_a = st.jacobian_approx_np(model, z, eps)
        U_a, S_a = st.svd_of_jacobian(jac_a)
        approx_set = st.enumerate_set(seq, S_a, U_a, PROD_KAPPA, PROD_TMUT)

        rows.append({
            "peptide": seq,
            "length": len(seq),
            "n_single_exact": len(exact_set),
            "n_single_approx": len(approx_set),
            "jac_rel_error": st.relative_frobenius_error(jac_a, jac_exact),
            "sv_rel_error": st.singular_value_rel_error(S_a, S_e, TOP_K),
            "subspace_overlap": st.subspace_overlap(U_a, U_e, TOP_K),
            "mut_jaccard": st.jaccard(approx_set, exact_set),
        })

        if i % 50 == 0 or i == len(peptides):
            dt = time.time() - t0
            rate = i / dt
            eta = (len(peptides) - i) / max(rate, 1e-6)
            print(f"  ... {i}/{len(peptides)}  ({rate:.2f} pep/s, "
                  f"elapsed {dt/60:.1f} min, ETA {eta/60:.1f} min)",
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = CACHE / "_thesis_rq2_eps_prod_large.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    summary = {
        "n_peptides": int(len(df)),
        "eps": eps,
        "kappa": PROD_KAPPA,
        "theta_mut": PROD_TMUT,
        "top_k": TOP_K,
        "device": device,
        "seed": seed,
        "source": source,
        "elapsed_sec": round(time.time() - t0, 1),
        "metrics": {},
    }
    for col in ["jac_rel_error", "sv_rel_error", "subspace_overlap",
                "mut_jaccard", "n_single_exact"]:
        s = df[col]
        summary["metrics"][col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    out_json = CACHE / "_thesis_rq2_eps_prod_large_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}\n")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
