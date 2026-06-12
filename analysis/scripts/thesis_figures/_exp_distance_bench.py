"""Benchmark candidate distance approximations on a small peptide sample (run on bury).

For each parent peptide we enumerate a capped MUTANG candidate set, re-encode it, and score
every candidate with five distances-to-parent, then report agreement / coverage / cost:

  eucl    : Euclidean latent            ||z' - z||
  maha    : first-order pullback         sqrt((z'-z)^T G(z) (z'-z)),  G = J(z)^T J(z) at parent
  ambient : decoder-output distance      ||softmax(dec z') - softmax(dec z)||_2   (the 'main' notion)
  geo_graph: graph/Isomap geodesic       (the current cache method, _geodesic.py)
  geo_energy: energy-minimising path geodesic (NEW): straight-line init, optimise interior
             control points to minimise the discrete Riemannian energy sum_m d_m^T g(mid_m) d_m
             (g recomputed each step; no Christoffels, no inversion), report path length.

Env: DB_N (#parents, default 6), DB_CAP (#candidates/parent), DB_M (path segments), DB_STEPS.
Writes _thesis_distbench.csv (per-candidate) and prints the summary.
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
from fig_rq56_potentials import HydrAMPEncoderDecoder, ALPHABET, MAXLEN, setup, BENCHMARK_PEPTIDES  # noqa: E402
from pep_compass.models.encoder_decoder.utils import decoder_jacobian  # noqa: E402
from pep_compass.geometry.utils import metric_from_jac  # noqa: E402
from _geodesic import geodesic_distance_to_parent  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = int(os.environ.get("DB_N", 6))
CAP = int(os.environ.get("DB_CAP", 60))
M = int(os.environ.get("DB_M", 4))           # path segments for the energy geodesic
STEPS = int(os.environ.get("DB_STEPS", 40))
JAC_EPS, REG = 1e-6, 1e-8
rng = np.random.default_rng(0)


def build_candidates(pep, muts, cap):
    """All single mutations + a uniform sample of multi-position combinations, capped."""
    positions = sorted(muts.keys()); padded = pep.ljust(MAXLEN)
    seqs, nmut = [], []
    for p in positions:                      # singles
        for a in muts[p]:
            s = list(padded); s[p] = ALPHABET[a]
            seqs.append("".join(s[:len(pep)])); nmut.append(1)
    while len(seqs) < cap:                   # random multi-position
        chosen = {}
        for p in positions:
            opts = list(muts[p])
            if opts and rng.random() < 0.5:
                chosen[p] = int(rng.choice(opts))
        if len(chosen) >= 2:
            s = list(padded)
            for p, a in chosen.items():
                s[p] = ALPHABET[a]
            seqs.append("".join(s[:len(pep)])); nmut.append(len(chosen))
    return seqs[:cap], nmut[:cap]


def jac_metric(z_pts):                       # g = J^T J at given latent points, batched
    J = decoder_jacobian(lambda x: hyd.decoder_forward(x, softmax=True, flatten=True),
                         z_pts, "approx", {"jacobian_eps": JAC_EPS})
    g = metric_from_jac(J)
    return g + REG * torch.eye(g.shape[-1], device=g.device)[None]


def energy_geodesic(z, zz, M=4, steps=40, lr=0.05):
    """Length of the energy-minimising discrete geodesic from z to each row of zz."""
    Nc, Z = zz.shape
    t = torch.linspace(0, 1, M + 1, device=zz.device)[1:-1]               # interior fractions
    interior = (z[None] + t[:, None, None] * (zz - z)[None]).permute(1, 0, 2).contiguous()
    interior = interior.detach().requires_grad_(True)                    # (Nc, M-1, Z)
    z0 = z[None].expand(Nc, 1, Z); z1 = zz[:, None, :]
    opt = torch.optim.Adam([interior], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        path = torch.cat([z0, interior, z1], dim=1)                      # (Nc, M+1, Z)
        d = path[:, 1:] - path[:, :-1]                                   # (Nc, M, Z)
        mids = 0.5 * (path[:, 1:] + path[:, :-1]).reshape(-1, Z)
        g = jac_metric(mids).reshape(Nc, M, Z, Z)
        seg = torch.einsum("nmi,nmij,nmj->nm", d, g, d).clamp(min=0)     # (Nc, M)
        seg.sum().backward()
        opt.step()
    with torch.no_grad():
        path = torch.cat([z0, interior, z1], dim=1)
        d = path[:, 1:] - path[:, :-1]
        mids = 0.5 * (path[:, 1:] + path[:, :-1]).reshape(-1, Z)
        g = jac_metric(mids).reshape(Nc, M, Z, Z)
        seg = torch.einsum("nmi,nmij,nmj->nm", d, g, d).clamp(min=0)
        return torch.sqrt(seg).sum(dim=1).cpu().numpy()                  # (Nc,) path length


def main():
    global hyd
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=JAC_EPS,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")["sequence"]
    pool = pool.dropna().astype(str)
    pool = pool[(pool.str.len() <= MAXLEN) & (pool.str.len() > 3)
                & pool.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates()
    peps = list(BENCHMARK_PEPTIDES.values())[:N] if N <= 6 else pool.sample(N, random_state=0).tolist()
    print(f"[bench] device={DEV} parents={len(peps)} cap={CAP} M={M} steps={STEPS}")

    timing = {k: 0.0 for k in ["eucl", "maha", "ambient", "geo_graph", "geo_energy"]}
    rows = []
    for pep in peps:
        z, ts, muts, G = setup(hyd, pep)
        if len(muts) < 2:
            continue
        seqs, nmut = build_candidates(pep, muts, CAP)
        with torch.no_grad():
            zz = hyd.encode_peptides(seqs).detach().to(DEV)
            z1 = hyd.encode_peptides([pep]).detach().to(DEV)
        z_np = z1.cpu().numpy()[0]; zz_np = zz.cpu().numpy()
        Gmat = metric_from_jac(decoder_jacobian(
            lambda x: hyd.decoder_forward(x, softmax=True, flatten=True), z1, "approx",
            {"jacobian_eps": JAC_EPS}))[0].detach().cpu().numpy()

        def timed(key, fn):
            t0 = time.time(); out = fn(); timing[key] += time.time() - t0; return out

        d_eucl = timed("eucl", lambda: np.linalg.norm(zz_np - z_np, axis=1))
        diff = zz_np - z_np
        d_maha = timed("maha", lambda: np.sqrt(np.clip(np.einsum("ni,ij,nj->n", diff, Gmat, diff), 0, None)))
        def amb():
            with torch.no_grad():
                a0 = hyd.decoder_forward(z1, softmax=True, flatten=True)
                a1 = hyd.decoder_forward(zz, softmax=True, flatten=True)
                return torch.linalg.norm(a1 - a0, dim=1).cpu().numpy()
        d_amb = timed("ambient", amb)
        d_gg = timed("geo_graph", lambda: geodesic_distance_to_parent(hyd, z1, zz, k=12))
        d_ge = timed("geo_energy", lambda: energy_geodesic(z1[0], zz, M=M, steps=STEPS))

        for i, s in enumerate(seqs):
            rows.append({"pep": pep, "seq": s, "n_mut": nmut[i], "eucl": d_eucl[i],
                         "maha": d_maha[i], "ambient": d_amb[i], "geo_graph": d_gg[i],
                         "geo_energy": d_ge[i]})
        print(f"  {pep[:12]:12s} cand={len(seqs)}")

    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "_thesis_distbench.csv", index=False)
    cols = ["eucl", "maha", "ambient", "geo_graph", "geo_energy"]

    print("\n=== coverage (finite fraction) ===")
    for c in cols:
        print(f"  {c:11s} {np.isfinite(df[c]).mean():.3f}")
    print("\n=== cost (total s over all parents) ===")
    for c in cols:
        print(f"  {c:11s} {timing[c]:.2f}s")

    print("\n=== overall off-diagonal Spearman between methods ===")
    print("            " + "".join(f"{c:>11s}" for c in cols))
    for a in cols:
        line = f"  {a:9s}"
        for b in cols:
            m = np.isfinite(df[a]) & np.isfinite(df[b])
            line += f"{spearmanr(df[a][m], df[b][m]).correlation:>11.3f}"
        print(line)

    print("\n=== median per-peptide Spearman vs geo_graph and vs geo_energy, within fixed n_mut ===")
    for ref in ("geo_graph", "geo_energy"):
        print(f"  reference = {ref}")
        for nm in (1, 2, 3):
            sub = df[df.n_mut == nm]
            for c in cols:
                if c == ref:
                    continue
            rr = {c: np.median([spearmanr(g[c], g[ref]).correlation
                                for _, g in sub.groupby("pep")
                                if len(g) >= 4 and g[c].nunique() > 1 and g[ref].nunique() > 1] or [np.nan])
                  for c in cols if c != ref}
            print(f"    n_mut={nm}: " + "  ".join(f"{c}={v:+.2f}" for c, v in rr.items()))


if __name__ == "__main__":
    main()
