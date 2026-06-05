"""Benchmark the two candidate geodesic distances for RQ5 (Phase 1).

Sizes wall-clock + peak GPU memory for:
  * decoder Jacobian (approx / strict), single and batched-across-points
  * frozen-Gamma build (1 + Z perturbed Jacobians) per parent
  * Candidate 1  -- frozen-Gamma RK4 + log_map_shooting, batched over a parent's mutants
  * Candidate 2  -- batched node-metric sweep + per-parent kNN graph + Dijkstra

then extrapolates to the full RQ5 scale (~506 parents, ~270k mutants) and prints a
JSON summary. Pure timing/feasibility -- correctness is covered by
``analysis/scripts/thesis_figures/_geometry_selftest.py``.

Run (on a GPU node):
    python scripts/bench_geodesic_distance.py --device cuda
    python scripts/bench_geodesic_distance.py --device cpu --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from einops import einsum, rearrange
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (  # noqa: E402
    HydrAMPEncoderDecoder,
)
from pep_compass.models.encoder_decoder.utils import decoder_jacobian  # noqa: E402
from pep_compass.geometry.utils import (  # noqa: E402
    metric_from_jac,
    approx_dg_from_jac,
    log_map_shooting,
    exponential_map,
)

BENCH_PEPTIDES = [
    "FLYKWWIRIGRLKL", "KYCRRFRWLTFRWL", "KFRNRHRWKFKLIFRN",
    "KKYWLIRKWIRLWFLT", "KTLKIIRLLF", "RMARNLVRYVQGLKKKKVI",
]

# full-RQ5 scale (from fig_rq56_potentials.py: ~500 sample + 6 bench parents,
# <=500 mutants/parent for samples, 3000 for benchmarks)
N_PARENTS = 506
MUTANTS_PER_PARENT = 500
TOTAL_MUTANTS = 270_000

JAC_EPS = 1e-6      # inner finite-difference eps for the approx Jacobian
GAMMA_EPS = 0.05    # outer perturbation eps for dg (matches geodesic_mutation_check)
METRIC_REG = 1.0    # lambda for the regularised metric g_reg = J^T J + lambda I


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, n_warmup=1, n_rep=3):
    for _ in range(n_warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n_rep


def christoffel_regularised(jac, dg, reg=METRIC_REG):
    g = einsum(jac, jac, "b m i, b m j -> b i j")
    g_reg = g + reg * torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)[None]
    g_inv = torch.linalg.inv(g_reg)
    T = (rearrange(dg, "b l i j -> b l i j")
         + rearrange(dg, "b l j i -> b l i j")
         - rearrange(dg, "b i j l -> b l i j"))
    return 0.5 * einsum(g_inv, T, "b k l, b l i j -> b k i j")


def build_frozen_gamma(hyd, z_parent, device):
    """Regularised Christoffel symbols at z_parent (1 + Z perturbed approx-Jacobians)."""
    Z = hyd.latent_dim
    eye = torch.eye(Z, device=device)
    pts = torch.cat([z_parent, z_parent + GAMMA_EPS * eye], dim=0)  # (1+Z, Z)
    fwd = lambda x: hyd.decoder_forward(x, softmax=True, flatten=True)
    J_all = decoder_jacobian(fwd, pts, "approx", {"jacobian_eps": JAC_EPS})  # (1+Z,M,Z)
    J = J_all[:1]                                   # (1,M,Z)
    J_pert = J_all[1:].unsqueeze(0)                 # (1,Z,M,Z)
    dg = approx_dg_from_jac(J, J_pert, GAMMA_EPS)   # (1,Z,Z,Z)
    g_reg = metric_from_jac(J) + METRIC_REG * torch.eye(Z, device=device)[None]
    Gamma = christoffel_regularised(J, dg)          # (1,Z,Z,Z)
    return Gamma, g_reg


def batched_metric_sweep(hyd, latents, device, chunk=512, reg=0.0):
    """Compute g(z)=J^T J (+ reg I) for many latents via batched approx Jacobians.
    -> (N,Z,Z).  Candidate 2 uses reg~0 (no inversion, just a PSD guard)."""
    fwd = lambda x: hyd.decoder_forward(x, softmax=True, flatten=True)
    Z = hyd.latent_dim
    out = []
    for i in range(0, latents.shape[0], chunk):
        xb = latents[i:i + chunk]
        J = decoder_jacobian(fwd, xb, "approx", {"jacobian_eps": JAC_EPS})  # (b,M,Z)
        g = metric_from_jac(J)
        if reg:
            g = g + reg * torch.eye(Z, device=device)[None]
        out.append(g)
    return torch.cat(out, dim=0)


def candidate1_distances(z_parent, mutants, Gamma, g_reg, device,
                         n_steps=24, n_iters=12, chunk=128):
    """Frozen-Gamma geodesic distance parent->each mutant, batched in chunks."""
    M = mutants.shape[0]
    dists = torch.empty(M, device=device)
    for i in range(0, M, chunk):
        yb = mutants[i:i + chunk]
        b = yb.shape[0]
        xb = z_parent.expand(b, -1)
        Gb = Gamma.expand(b, -1, -1, -1)
        gamma_fn = lambda _x, _G=Gb: _G
        v = log_map_shooting(xb, yb, gamma_fn, t1=1.0, n_steps=n_steps,
                             n_iters=n_iters, lr=0.3)
        gb = g_reg.expand(b, -1, -1)
        d2 = einsum(v, gb, v, "b i, b i j, b j -> b")
        dists[i:i + b] = torch.sqrt(torch.clamp(d2, min=0.0))
    return dists


def knn_edges(nodes_np, k=12):
    """Unique undirected kNN edges. -> (E,2) int array."""
    n = nodes_np.shape[0]
    tree = cKDTree(nodes_np)
    _, idx = tree.query(nodes_np, k=min(k + 1, n))
    edges = set()
    for a in range(n):
        for b in idx[a, 1:]:
            edges.add((a, int(b)) if a < b else (int(b), a))
    return np.array(sorted(edges), dtype=np.int64)


def candidate2_graph(hyd, z_parent, mutants, device, k=12):
    """kNN graph over {parent}+mutants; edge weight = MIDPOINT pullback length
    sqrt(diff^T g(mid) diff); Dijkstra from parent. Returns (dists, n_edges)."""
    nodes = torch.cat([z_parent, mutants], dim=0)            # (n,Z)
    nnp = nodes.detach().cpu().numpy()
    n = nnp.shape[0]
    edges = knn_edges(nnp, k=k)                              # (E,2)
    a_idx = torch.as_tensor(edges[:, 0], device=device)
    b_idx = torch.as_tensor(edges[:, 1], device=device)
    mids = 0.5 * (nodes[a_idx] + nodes[b_idx])              # (E,Z)
    g_mid = batched_metric_sweep(hyd, mids, device, reg=1e-8)  # (E,Z,Z)
    diff = nodes[b_idx] - nodes[a_idx]
    L = torch.sqrt(torch.clamp(
        einsum(diff, g_mid, diff, "e i, e i j, e j -> e"), min=0.0))
    Lnp = L.detach().cpu().numpy()
    g = coo_matrix((Lnp, (edges[:, 0], edges[:, 1])), shape=(n, n))
    d = dijkstra(g, directed=False, indices=0)
    return d[1:], edges.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="smaller sweep sizes")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes; just exercise code paths")
    args = ap.parse_args()
    if args.smoke:
        args.quick = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device = {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print("GPU:", torch.cuda.get_device_name(0))

    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=JAC_EPS,
                                field_eps=1e-6, device=device)
    hyd.eval()
    Z = hyd.latent_dim
    fwd = lambda x: hyd.decoder_forward(x, softmax=True, flatten=True)

    with torch.no_grad():
        z = hyd.encode_peptides(BENCH_PEPTIDES)  # (6, Z)
    z0 = z[:1]
    res = {"device": str(device), "Z": Z}

    # 1) single Jacobians  (approx is what both candidates use; strict is reference-only
    #    and fails on cuDNN RNN backward in eval mode -> guarded)
    res["jac_approx_single_s"] = timed(
        lambda: decoder_jacobian(fwd, z0, "approx", {"jacobian_eps": JAC_EPS}), device)
    try:
        res["jac_strict_single_s"] = timed(
            lambda: decoder_jacobian(fwd, z0, "strict", None), device, n_warmup=0, n_rep=1)
    except Exception as e:
        res["jac_strict_single_s"] = f"skipped ({type(e).__name__})"

    # 2) batched approx-Jacobian throughput (Candidate-2 building block)
    sweep_sizes = [16] if args.smoke else ([64, 256] if args.quick else [64, 256, 1024])
    res["batched_jac"] = {}
    for B in sweep_sizes:
        xb = z0 + 0.1 * torch.randn(B, Z, device=device)
        s = timed(lambda: decoder_jacobian(fwd, xb, "approx", {"jacobian_eps": JAC_EPS}),
                  device, n_warmup=1, n_rep=2)
        res["batched_jac"][B] = {"s": s, "pts_per_s": B / s}

    # 3) frozen-Gamma build per parent
    res["frozen_gamma_build_s"] = timed(
        lambda: build_frozen_gamma(hyd, z0, device), device, n_warmup=1, n_rep=2)
    Gamma, g_reg = build_frozen_gamma(hyd, z0, device)
    res["gamma_norm"] = float(Gamma.norm())
    res["gamma_max"] = float(Gamma.abs().max())

    # 4) Candidate 1: per-parent shooting over a mutant batch
    M = 8 if args.smoke else (128 if args.quick else MUTANTS_PER_PARENT)
    mutants = z0 + 0.3 * torch.randn(M, Z, device=device)
    t_c1 = timed(lambda: candidate1_distances(z0, mutants, Gamma, g_reg, device),
                 device, n_warmup=0, n_rep=1)
    res["cand1"] = {
        "mutants": M, "per_parent_shoot_s": t_c1,
        "per_distance_ms": 1e3 * t_c1 / M,
        "est_full_s": N_PARENTS * (res["frozen_gamma_build_s"] + t_c1 * MUTANTS_PER_PARENT / M),
    }

    # 5) Candidate 2: midpoint-edge metric sweep throughput + per-parent graph
    N_sweep = 64 if args.smoke else (2000 if args.quick else 8000)
    latents = z0 + 0.5 * torch.randn(N_sweep, Z, device=device)
    t_sweep = timed(lambda: batched_metric_sweep(hyd, latents, device, reg=1e-8),
                    device, n_warmup=1, n_rep=1)
    metrics_sweep_rate = N_sweep / t_sweep      # midpoint-metric Jacobians per second
    # per-parent graph at the real node count (parent + MUTANTS_PER_PARENT mutants)
    _, n_edges = candidate2_graph(hyd, z0, mutants, device)
    t_graph = timed(lambda: candidate2_graph(hyd, z0, mutants, device), device,
                    n_warmup=0, n_rep=1)
    edges_full = N_PARENTS * n_edges * (MUTANTS_PER_PARENT / M)  # scale node count -> 500
    res["cand2"] = {
        "metric_sweep_pts_per_s": metrics_sweep_rate,
        "graph_nodes": 1 + M, "n_edges": int(n_edges),
        "per_parent_graph_s": t_graph,
        "est_midpoint_metrics": int(edges_full),
        "est_metric_sweep_s": edges_full / metrics_sweep_rate,
        "est_full_s": N_PARENTS * t_graph * (MUTANTS_PER_PARENT / M),
    }

    if device.type == "cuda":
        res["peak_gpu_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9

    print(json.dumps(res, indent=2))
    out = os.path.join(ROOT, "results", "geodesic_bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved {out}")
    print("\n--- full-scale estimates (506 parents, ~270k mutants) ---")
    print(f"Candidate 1 (frozen-Gamma RK4 shooting): {res['cand1']['est_full_s']/60:.1f} min")
    print(f"Candidate 2 (graph/Isomap, midpoint edges): {res['cand2']['est_full_s']/60:.1f} min "
          f"(~{res['cand2']['est_midpoint_metrics']:,} midpoint metrics @ "
          f"{res['cand2']['metric_sweep_pts_per_s']:.0f}/s "
          f"= {res['cand2']['est_metric_sweep_s']/60:.1f} min of Jacobians)")


if __name__ == "__main__":
    main()
