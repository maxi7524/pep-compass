"""Potential-minimizing Geodesic Search (PoGS) distance on HydrAMP's decoder manifold.

PoGS (PepCompass, Mozejko et al.) finds a discrete latent path Z = (z_0, ..., z_N) joining a
parent z_0 = z to a target z_N = z' by ADAM-minimising the energy

    E_{lam,mu}(Z) =  sum_{k=0}^{N-1} || X_{k+1} - X_k ||_2^2      (kinetic: ambient geometric sim.)
                   + lam * sum_{k=0}^{N}   Phi(X_k)               (potential: property / activity)
                   + mu  * sum_{k=0}^{N-1} || z_{k+1} - z_k ||_2^2 (latent regulariser, Euclidean)

with X_k = Dec(z_k) the (softmax, flattened) decoder output. The interior way-points
{z_1, ..., z_{N-1}} are free; the endpoints are fixed. NOTE every norm here is a *plain
Euclidean* norm -- the kinetic term is the proper chord distance in ambient (soft-peptide)
space and the regulariser is the Euclidean latent step; the pull-back metric G = J^T J is never
formed (this is the whole point, and is what the earlier G-based ``energy_geodesic`` got wrong).

The reported PoGS *distance* is the ambient chord *length* of the optimised path (un-squared):

    d_PoGS^{(lam)}(z, z') = sum_{k=0}^{N-1} || Dec(z_k*) - Dec(z_{k+1}*) ||_2 ,
        {z_k*} = argmin_{z_1..z_{N-1}} E_{lam,mu}(Z),  z_0* = z, z_N* = z'.

With lam = 0 the potential vanishes and d_PoGS^{(0)} is the (mu-regularised) geodesic distance.
Only forward decodes + autograd through the decoder are used -- no Jacobians, no Christoffels,
no metric inversion -- so it is cheap and GPU-batchable over a parent's whole candidate set.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

# Defaults (all overridable by callers / env in the batch job). LR=1e-3 is important: larger
# rates (e.g. 5e-2) make ADAM overshoot, pushing interior way-points into saturated decoder
# regions so the realised path length *grows* instead of shrinking. At 1e-3 the energy decreases
# monotonically and the length settles just above the direct ambient chord (the geodesic floor).
N_SEG = 8         # number of path segments -> N+1 way-points, N-1 free interior points
MU = 1e-2         # latent Euclidean regulariser weight
LR = 1e-3         # ADAM learning rate
STEPS = 500       # ADAM iterations (length is rank-stable from ~300 on; energy still easing)


def _decode_path(hyd, path: torch.Tensor) -> torch.Tensor:
    """path: (M, P, Z) latent way-points -> X: (M, P, LA) softmax decoder outputs (grad-enabled)."""
    M, P, Z = path.shape
    X = hyd.decoder_forward(path.reshape(M * P, Z), softmax=True, flatten=True)
    return X.reshape(M, P, -1)


def pogs_distance_to_parent(
    hyd,
    z_parent: torch.Tensor,           # (Z,) or (1, Z)
    z_mutants: torch.Tensor,          # (M, Z)
    lam: float = 0.0,
    mu: float = MU,
    n_seg: int = N_SEG,
    steps: int = STEPS,
    lr: float = LR,
    phi: Callable[[torch.Tensor], torch.Tensor] | None = None,
    chunk: int = 1024,
    return_components: bool = False,
):
    """PoGS path length from the parent to each mutant. -> (M,) numpy (ambient chord length).

    ``phi`` (optional) maps the ambient way-points X (M, N+1, LA) to a per-candidate potential
    sum (M,); with ``phi=None`` (default, lam irrelevant) this is the pure geodesic distance.
    With ``return_components=True`` also returns the per-candidate kinetic energy and latent
    regulariser of the optimised path (handy for sanity checks / the notebook).
    """
    device = z_parent.device
    z0_full = z_parent.reshape(1, -1)                 # (1, Z)
    Z = z0_full.shape[1]
    M = z_mutants.shape[0]
    if M == 0:
        empty = np.zeros(0, dtype=np.float64)
        return (empty, empty, empty) if return_components else empty

    lengths = np.empty(M, dtype=np.float64)
    kin_out = np.empty(M, dtype=np.float64)
    reg_out = np.empty(M, dtype=np.float64)

    t_int = torch.linspace(0.0, 1.0, n_seg + 1, device=device)[1:-1]  # (N-1,) interior fractions

    for s in range(0, M, chunk):
        zz = z_mutants[s:s + chunk]                   # (m, Z)
        m = zz.shape[0]
        z_start = z0_full.expand(m, 1, Z)             # (m, 1, Z)
        z_end = zz[:, None, :]                         # (m, 1, Z)
        # straight-line init of the interior way-points
        interior = (z0_full[:, None, :]
                    + t_int[None, :, None] * (zz - z0_full)[:, None, :]).contiguous()
        interior = interior.detach().requires_grad_(True)   # (m, N-1, Z)

        opt = torch.optim.Adam([interior], lr=lr)
        # HydrAMP's decoder is an RNN; cuDNN cannot run RNN backward while the module is in eval
        # mode, so disable cuDNN for the optimisation (native RNN backward works in eval mode).
        with torch.backends.cudnn.flags(enabled=False):
            for _ in range(steps):
                opt.zero_grad()
                path = torch.cat([z_start, interior, z_end], dim=1)   # (m, N+1, Z)
                X = _decode_path(hyd, path)                            # (m, N+1, LA)
                dX = X[:, 1:] - X[:, :-1]                              # (m, N, LA)
                kinetic = (dX ** 2).sum(dim=(-1, -2))                  # (m,)
                dz = path[:, 1:] - path[:, :-1]                        # (m, N, Z)
                reg = mu * (dz ** 2).sum(dim=(-1, -2))                 # (m,)
                loss = kinetic + reg
                if phi is not None and lam != 0.0:
                    loss = loss + lam * phi(X)
                loss.sum().backward()
                opt.step()

        with torch.no_grad():
            path = torch.cat([z_start, interior, z_end], dim=1)
            X = _decode_path(hyd, path)
            dX = X[:, 1:] - X[:, :-1]
            length = torch.linalg.norm(dX, dim=-1).sum(dim=-1)    # (m,) un-squared chord length
            lengths[s:s + m] = length.detach().cpu().numpy()
            if return_components:
                kin_out[s:s + m] = (dX ** 2).sum(dim=(-1, -2)).detach().cpu().numpy()
                dz = path[:, 1:] - path[:, :-1]
                reg_out[s:s + m] = (mu * (dz ** 2).sum(dim=(-1, -2))).detach().cpu().numpy()

    if return_components:
        return lengths, kin_out, reg_out
    return lengths


def direct_chord_to_parent(hyd, z_parent: torch.Tensor, z_mutants: torch.Tensor,
                           chunk: int = 4096) -> np.ndarray:
    """Direct (single-segment) ambient chord || Dec(z') - Dec(z) ||_2 to each mutant. -> (M,).

    This is the N=1 / straight-decode lower bound that PoGS path length should sit at or above.
    """
    device = z_parent.device
    with torch.no_grad():
        a0 = hyd.decoder_forward(z_parent.reshape(1, -1), softmax=True, flatten=True)  # (1, LA)
        out = np.empty(z_mutants.shape[0], dtype=np.float64)
        for s in range(0, z_mutants.shape[0], chunk):
            a1 = hyd.decoder_forward(z_mutants[s:s + chunk], softmax=True, flatten=True)
            out[s:s + chunk] = torch.linalg.norm(a1 - a0, dim=1).cpu().numpy()
    return out
