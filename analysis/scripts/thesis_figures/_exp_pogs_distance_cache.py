"""Add a PoGS (Potential-minimizing Geodesic Search, lambda=0) distance column to the candidate
caches (run on bury). PoGS path energy is ADAM-minimised per candidate; the reported distance is
the ambient chord *length* of the optimised path -- the proper geodesic distance with no metric G
(see _pogs.py). Forward decodes + autograd only, GPU-batched per parent.

Inputs / outputs (CACHE = results/data/all_in/_cache):
  _thesis_rq5_distances.parquet        -> _thesis_rq5_distances_pogs.parquet   (385 peptides; TANDEM)
  _thesis_mutangplus_fulldist.parquet  -> _thesis_mutangplus_pogs.parquet      (845 peptides; MUTANG+)

Env: PGS_WHICH (rq5|mutangplus|both, default both), PGS_LIMIT (#peptides cap, 0=all),
     PGS_STEPS, PGS_SEG, PGS_MU, PGS_LR, PGS_CHUNK.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder  # noqa: E402
from _pogs import pogs_distance_to_parent  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIMIT = int(os.environ.get("PGS_LIMIT", 0))
STEPS = int(os.environ.get("PGS_STEPS", 500))
SEG = int(os.environ.get("PGS_SEG", 8))
MU = float(os.environ.get("PGS_MU", 1e-2))
LR = float(os.environ.get("PGS_LR", 1e-3))
CHUNK = int(os.environ.get("PGS_CHUNK", 1024))
WHICH = os.environ.get("PGS_WHICH", "both")

JOBS = {
    "rq5": ("_thesis_rq5_distances.parquet", "_thesis_rq5_distances_pogs.parquet"),
    "mutangplus": ("_thesis_mutangplus_fulldist.parquet", "_thesis_mutangplus_pogs.parquet"),
}


def run_one(hyd, src, dst):
    df = pd.read_parquet(CACHE / src).reset_index(drop=True)
    peps = list(dict.fromkeys(df["pep"]))
    if LIMIT:
        peps = peps[:LIMIT]
        df = df[df["pep"].isin(peps)].reset_index(drop=True)
    print(f"[pogs] {src}: {len(df):,} candidates over {len(peps)} parents | device={DEV}", flush=True)

    dist = np.full(len(df), np.nan)
    idx_by_pep = df.groupby("pep").indices
    t0 = time.time()
    for ip, pep in enumerate(peps):
        rows = idx_by_pep[pep]
        seqs = df["seq"].values[rows].tolist()
        with torch.no_grad():
            z0 = hyd.encode_peptides([pep]).detach()[0]
            zz = hyd.encode_peptides(seqs).detach()
        d = pogs_distance_to_parent(hyd, z0, zz, lam=0.0, mu=MU, n_seg=SEG,
                                    steps=STEPS, lr=LR, chunk=CHUNK)
        dist[rows] = d
        if (ip + 1) % 25 == 0:
            print(f"  ...{ip + 1}/{len(peps)} ({time.time() - t0:.0f}s)", flush=True)
    df["dist_pogs"] = dist

    df.to_parquet(CACHE / dst)
    print(f"[pogs] wrote {dst} ({len(df):,} rows, {time.time() - t0:.0f}s)", flush=True)

    fin = np.isfinite(df["dist_pogs"])
    for other in ("dist_geo", "dist_eucl", "dist_maha"):
        if other in df:
            m = fin & np.isfinite(df[other])
            print(f"  Spearman(dist_pogs, {other:9s}) = "
                  f"{spearmanr(df['dist_pogs'][m], df[other][m]).correlation:+.3f}")


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    keys = ["rq5", "mutangplus"] if WHICH == "both" else [WHICH]
    print(f"[pogs] SEG={SEG} STEPS={STEPS} MU={MU} LR={LR} LIMIT={LIMIT or 'all'} jobs={keys}")
    for k in keys:
        run_one(hyd, *JOBS[k])


if __name__ == "__main__":
    main()
