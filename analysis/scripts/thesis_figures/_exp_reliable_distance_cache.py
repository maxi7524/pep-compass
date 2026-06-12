"""Build the consolidated reliable-distance cache (run on bury).

Starts from the MUTANG+ candidate set already scored for geometry
(_thesis_mutangplus_fulldist.parquet: pep, dataset, seq, n_mut, dist_geo, dist_eucl, dist_maha)
and ADDS the decoder-output ('ambient') distance

    dist_ambient = || softmax(Dec(enc(mutant))) - softmax(Dec(enc(parent))) ||_2 ,

the absolute, set-independent distance recommended by the benchmark (cheap, 100% coverage,
tracks both Euclidean and the Isomap geodesic). Forward passes only -- no Jacobians, no geodesics.

Output: _thesis_reliable_distances.parquet with
    pep, dataset, seq, n_mut, dist_eucl, dist_ambient, dist_geo, dist_maha
plus a quick agreement sanity print.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fig_rq56_potentials import HydrAMPEncoderDecoder  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = int(os.environ.get("RD_BATCH", 4000))


def ambient(hyd, seqs):
    """softmax decoder output (B, L*A) for a list of sequences, batched."""
    out = []
    for i in range(0, len(seqs), BATCH):
        with torch.no_grad():
            z = hyd.encode_peptides(seqs[i:i + BATCH]).detach()
            a = hyd.decoder_forward(z, softmax=True, flatten=True).detach().cpu().numpy()
        out.append(a)
        print(f"    ...{min(i + BATCH, len(seqs))}/{len(seqs)}", flush=True)
    return np.concatenate(out, axis=0)


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    df = pd.read_parquet(CACHE / "_thesis_mutangplus_fulldist.parquet").reset_index(drop=True)
    print(f"[rd] {len(df):,} candidates over {df.pep.nunique()} parents | device={DEV}")

    parents = sorted(df["pep"].unique())
    a_par = {p: v for p, v in zip(parents, ambient(hyd, parents))}      # parent ambient vectors
    print("  parents decoded; decoding candidates ...")
    a_cand = ambient(hyd, df["seq"].tolist())
    par_mat = np.stack([a_par[p] for p in df["pep"]])
    df["dist_ambient"] = np.linalg.norm(a_cand - par_mat, axis=1)

    cols = ["pep", "dataset", "seq", "n_mut", "dist_eucl", "dist_ambient", "dist_geo", "dist_maha"]
    df[cols].to_parquet(CACHE / "_thesis_reliable_distances.parquet")
    print(f"[rd] wrote _thesis_reliable_distances.parquet ({len(df):,} rows)")

    off = np.isfinite(df["dist_ambient"])
    for a, b in [("dist_ambient", "dist_eucl"), ("dist_ambient", "dist_geo"),
                 ("dist_eucl", "dist_geo")]:
        m = off & np.isfinite(df[a]) & np.isfinite(df[b])
        print(f"  Spearman({a:12s}, {b:9s}) = {spearmanr(df[a][m], df[b][m]).correlation:+.3f}")


if __name__ == "__main__":
    main()
