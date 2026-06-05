"""Graph / Isomap geodesic distance on HydrAMP's decoder-pullback manifold.

The first-order pullback (Mahalanobis) distance d_G = ||J(z)(z'-z)|| uses the metric
at the parent only, where the softmax decoder saturates so G(z)~0 and a single
substitution looks "free" (d_G ~ 1e-3). The geodesic distance instead accumulates the
metric ALONG the path. We approximate it discretely (Isomap-style):

  * nodes        = {parent} ∪ {re-encoded mutants}  (latents already available)
  * edges        = symmetric kNN graph
  * edge weight  = sqrt(d^T g(mid) d), the pullback length at the edge MIDPOINT,
                   where g(mid)=J(mid)^T J(mid) (this is where the inter-peptide metric
                   blow-up lives — the node metrics alone would miss it)
  * d_geo(parent, mutant) = Dijkstra shortest path from the parent node.

No metric inversion and no ODE -> robust to the near-degenerate / ill-conditioned
pullback metric (the failure mode the thesis hit with shooting / energy minimisation).
Only a tiny PSD guard (reg) is added to tame finite-difference noise in J^T J.
"""
from __future__ import annotations

import numpy as np
import torch
from einops import einsum
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.geometry.utils import metric_from_jac

JAC_EPS = 1e-6     # inner finite-difference eps for the approx decoder Jacobian
PSD_REG = 1e-8     # tiny diagonal guard so finite-difference noise stays PSD


def _knn_edges(nodes_np: np.ndarray, k: int) -> np.ndarray:
    """Unique undirected kNN edges. -> (E, 2) int64."""
    n = nodes_np.shape[0]
    tree = cKDTree(nodes_np)
    _, idx = tree.query(nodes_np, k=min(k + 1, n))
    edges = set()
    for a in range(n):
        for b in idx[a, 1:]:
            edges.add((a, int(b)) if a < int(b) else (int(b), a))
    return np.array(sorted(edges), dtype=np.int64)


def _midpoint_metrics(hyd, mids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """g(mid) = J(mid)^T J(mid) for every edge midpoint, batched. -> (E, Z, Z)."""
    fwd = lambda x: hyd.decoder_forward(x, softmax=True, flatten=True)
    out = []
    for i in range(0, mids.shape[0], chunk):
        J = decoder_jacobian(fwd, mids[i:i + chunk], "approx", {"jacobian_eps": JAC_EPS})
        out.append(metric_from_jac(J))
    return torch.cat(out, dim=0)


def geodesic_distance_to_parent(
    hyd,
    z_parent: torch.Tensor,   # (1, Z) or (Z,)
    z_mutants: torch.Tensor,  # (N, Z)
    k: int = 12,
    chunk: int = 512,
    reg: float = PSD_REG,
) -> np.ndarray:
    """Graph geodesic distance from the parent to each mutant. -> (N,) numpy.

    Disconnected mutants (kNN graph not connected) fall back to the single-segment
    midpoint length parent->mutant, a valid upper bound on the geodesic.
    """
    device = z_parent.device
    z_parent = z_parent.reshape(1, -1)
    nodes = torch.cat([z_parent, z_mutants], dim=0)          # (1+N, Z)
    nnp = nodes.detach().cpu().numpy()
    n = nnp.shape[0]

    edges = _knn_edges(nnp, k)                                # (E, 2)
    a_idx = torch.as_tensor(edges[:, 0], device=device)
    b_idx = torch.as_tensor(edges[:, 1], device=device)
    mids = 0.5 * (nodes[a_idx] + nodes[b_idx])              # (E, Z)
    g_mid = _midpoint_metrics(hyd, mids, chunk=chunk)        # (E, Z, Z)
    if reg:
        g_mid = g_mid + reg * torch.eye(g_mid.shape[-1], device=device)[None]
    diff = nodes[b_idx] - nodes[a_idx]
    length = torch.sqrt(torch.clamp(
        einsum(diff, g_mid, diff, "e i, e i j, e j -> e"), min=0.0))
    Lnp = length.detach().cpu().numpy()

    graph = coo_matrix((Lnp, (edges[:, 0], edges[:, 1])), shape=(n, n))
    d = dijkstra(graph, directed=False, indices=0)           # (n,)
    d_mut = d[1:].copy()

    # fallback for unreachable mutants: direct parent->mutant midpoint length
    bad = ~np.isfinite(d_mut)
    if bad.any():
        idx = np.flatnonzero(bad)
        mids_d = 0.5 * (z_parent + z_mutants[idx])
        g_d = _midpoint_metrics(hyd, mids_d, chunk=chunk)
        if reg:
            g_d = g_d + reg * torch.eye(g_d.shape[-1], device=device)[None]
        diff_d = z_mutants[idx] - z_parent
        ld = torch.sqrt(torch.clamp(
            einsum(diff_d, g_d, diff_d, "e i, e i j, e j -> e"), min=0.0))
        d_mut[idx] = ld.detach().cpu().numpy()
    return d_mut
