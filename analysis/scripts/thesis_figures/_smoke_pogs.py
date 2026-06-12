"""Local smoke test for _pogs.py: build HydrAMP, run PoGS (lam=0) on a few cached candidates,
check finiteness/positivity, that PoGS length >= direct ambient chord, and that it correlates
with the cached graph geodesic / Euclidean distance."""
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
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder  # noqa: E402
from _pogs import pogs_distance_to_parent, direct_chord_to_parent  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PEP = int(os.environ.get("SMOKE_N", 3))
N_STEPS = int(os.environ.get("SMOKE_STEPS", 80))
N_SEG = int(os.environ.get("SMOKE_SEG", 8))


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    df = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    peps = list(dict.fromkeys(df["pep"]))[:N_PEP]
    print(f"device={DEV}  peptides={peps}  n_seg={N_SEG} steps={N_STEPS}")

    all_pogs, all_chord, all_geo, all_eucl, all_nmut = [], [], [], [], []
    for pep in peps:
        g = df[df.pep == pep].head(60).reset_index(drop=True)
        with torch.no_grad():
            z0 = hyd.encode_peptides([pep]).detach()[0]
            zz = hyd.encode_peptides(g["seq"].tolist()).detach()
        d_pogs = pogs_distance_to_parent(hyd, z0, zz, lam=0.0, n_seg=N_SEG, steps=N_STEPS)
        d_chord = direct_chord_to_parent(hyd, z0, zz)
        print(f"  {pep[:14]:14s} n={len(g):3d}  "
              f"pogs[min/med/max]={d_pogs.min():.3f}/{np.median(d_pogs):.3f}/{d_pogs.max():.3f}  "
              f"chord_med={np.median(d_chord):.3f}  "
              f">=chord: {(d_pogs >= d_chord - 1e-6).mean()*100:.0f}%  "
              f"finite: {np.isfinite(d_pogs).mean()*100:.0f}%")
        all_pogs += list(d_pogs); all_chord += list(d_chord)
        all_geo += list(g["dist_geo"]); all_eucl += list(g["dist_eucl"]); all_nmut += list(g["n_mut"])

    a = lambda x: np.asarray(x, float)
    p, c, geo, eu, nm = map(a, (all_pogs, all_chord, all_geo, all_eucl, all_nmut))
    m = np.isfinite(p) & np.isfinite(geo)
    print(f"\noverall n={m.sum()}")
    print(f"  Spearman(pogs, chord)    = {spearmanr(p[m], c[m]).correlation:+.3f}")
    print(f"  Spearman(pogs, dist_geo) = {spearmanr(p[m], geo[m]).correlation:+.3f}")
    print(f"  Spearman(pogs, dist_eucl)= {spearmanr(p[m], eu[m]).correlation:+.3f}")
    print(f"  pogs >= direct chord in {(p[m] >= c[m] - 1e-6).mean()*100:.1f}% of candidates")
    for nmv in (1, 2, 3):
        mm = m & (nm == nmv)
        if mm.sum() > 3:
            print(f"  n_mut={nmv}: median pogs={np.median(p[mm]):.3f}  median geo={np.median(geo[mm]):.3f}")


if __name__ == "__main__":
    main()
