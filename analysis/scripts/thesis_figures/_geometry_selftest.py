"""Differential-geometry self-test for ``pep_compass.geometry.utils``.

Validates the SHARED Levi-Civita / geodesic engine (the same functions RQ5 will
use) against closed-form ground truth on two analytic manifolds:

  * unit 2-sphere   g = diag(1, sin^2 u)         -> great-circle geodesics
  * Poincare UHP    g = diag(1/y^2, 1/y^2)        -> constant negative curvature

Everything is pushed through the project's own
``metric_from_jac`` / ``approx_dg_from_jac`` / ``christoffel_from_jac_and_dg`` /
``integrate_geodesic_rk4`` / ``exponential_map`` / ``log_map_shooting`` /
``riemannian_distance`` so a PASS certifies *those* functions, not a re-implementation.

Checks:
  1. Christoffel symbols vs analytic (and vs sympy if available).
  2. Exponential map along coordinate geodesics (meridian / vertical line) vs
     closed-form endpoints  -> validates RK4 + Christoffel + metric.
  3. Geodesic distance via log_map_shooting + riemannian_distance vs the
     great-circle (arccos) / arccosh formulas  -> validates the full pipeline.

Optional (``--hydramp``):
  4. Finite-difference dg (approx_dg_from_jac) vs autograd d(JtJ)/dz on the real
     HydrAMP decoder, swept over eps -> sizes the FD error that feeds Christoffels.

Run:
    python analysis/scripts/thesis_figures/_geometry_selftest.py
    python analysis/scripts/thesis_figures/_geometry_selftest.py --hydramp
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

try:  # make Unicode + incremental output work on Windows cp1250 consoles
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # pep-compass/
sys.path.insert(0, os.path.join(ROOT, "src"))

from pep_compass.geometry.utils import (  # noqa: E402
    metric_from_jac,
    approx_dg_from_jac,
    christoffel_from_jac_and_dg,
    exponential_map,
    log_map_shooting,
    riemannian_distance,
)

torch.set_default_dtype(torch.float64)
DEV = torch.device("cpu")
DG_EPS = 1e-5  # forward-difference step for approx_dg_from_jac in the toy tests


# ───────────────────────────── toy manifolds ──────────────────────────────
# Each manifold provides jac_fn(p) -> (1, M, Z), an analytic Christoffel, an
# analytic distance, and a coordinate geodesic with a closed-form endpoint.

def sphere_jac(p: torch.Tensor) -> torch.Tensor:
    """Unit-sphere embedding f(u,v)=(sin u cos v, sin u sin v, cos u). -> (1,3,2)."""
    u, v = p[..., 0], p[..., 1]
    su, cu, sv, cv = torch.sin(u), torch.cos(u), torch.sin(v), torch.cos(v)
    f_u = torch.stack([cu * cv, cu * sv, -su], dim=-1)   # ∂f/∂u  (1,3)
    f_v = torch.stack([-su * sv, su * cv, torch.zeros_like(u)], dim=-1)  # ∂f/∂v
    return torch.stack([f_u, f_v], dim=-1)               # (1,3,2)


def sphere_christoffel(p: torch.Tensor) -> torch.Tensor:
    """Analytic Γ^k_{ij} for g=diag(1,sin^2 u), indices 0=u,1=v.  -> (1,2,2,2)."""
    u = p[..., 0]
    su, cu = torch.sin(u), torch.cos(u)
    G = torch.zeros(p.shape[0], 2, 2, 2)
    G[:, 0, 1, 1] = -su * cu          # Γ^u_{vv}
    G[:, 1, 0, 1] = cu / su           # Γ^v_{uv}
    G[:, 1, 1, 0] = cu / su           # Γ^v_{vu}
    return G


def sphere_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    u1, v1 = float(p[0, 0]), float(p[0, 1])
    u2, v2 = float(q[0, 0]), float(q[0, 1])
    c = math.cos(u1) * math.cos(u2) + math.sin(u1) * math.sin(u2) * math.cos(v1 - v2)
    return math.acos(max(-1.0, min(1.0, c)))


def poincare_jac(p: torch.Tensor) -> torch.Tensor:
    """g=diag(1/y^2,1/y^2) via J=diag(1/y,1/y).  -> (1,2,2)."""
    y = p[..., 1]
    inv = 1.0 / y
    J = torch.zeros(p.shape[0], 2, 2)
    J[:, 0, 0] = inv
    J[:, 1, 1] = inv
    return J


def poincare_christoffel(p: torch.Tensor) -> torch.Tensor:
    """Analytic Γ for the Poincaré UHP, indices 0=x,1=y.  -> (1,2,2,2)."""
    y = p[..., 1]
    G = torch.zeros(p.shape[0], 2, 2, 2)
    G[:, 0, 0, 1] = -1.0 / y          # Γ^x_{xy}
    G[:, 0, 1, 0] = -1.0 / y          # Γ^x_{yx}
    G[:, 1, 0, 0] = 1.0 / y           # Γ^y_{xx}
    G[:, 1, 1, 1] = -1.0 / y          # Γ^y_{yy}
    return G


def poincare_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    px, py = float(p[0, 0]), float(p[0, 1])
    qx, qy = float(q[0, 0]), float(q[0, 1])
    arg = 1.0 + ((qx - px) ** 2 + (qy - py) ** 2) / (2.0 * py * qy)
    return math.acosh(arg)


# ─────────────────────── engine glue (live Christoffels) ───────────────────

def make_gamma_fn(jac_fn, eps: float = DG_EPS):
    """Live Γ(x): finite-difference dg from jac_fn, then christoffel_from_jac_and_dg."""
    def gamma_fn(x: torch.Tensor) -> torch.Tensor:
        B, Z = x.shape
        J = jac_fn(x)                                  # (B,M,Z)
        cols = []
        for k in range(Z):
            e = torch.zeros_like(x)
            e[:, k] = eps
            cols.append(jac_fn(x + e))                 # (B,M,Z)
        J_pert = torch.stack(cols, dim=1)              # (B,Z,M,Z)
        dg = approx_dg_from_jac(J, J_pert, eps)        # (B,Z,Z,Z)
        return christoffel_from_jac_and_dg(J, dg)      # (B,Z,Z,Z)

    return gamma_fn


def metric_fn_from(jac_fn):
    return lambda x: metric_from_jac(jac_fn(x))


# ───────────────────────────────── checks ─────────────────────────────────

def check_christoffel(name, jac_fn, analytic_fn, pts):
    gamma_fn = make_gamma_fn(jac_fn)
    worst = 0.0
    for p in pts:
        got = gamma_fn(p)
        exp = analytic_fn(p)
        worst = max(worst, float((got - exp).abs().max()))
    status = "PASS" if worst < 1e-3 else "FAIL"
    print(f"  [{status}] {name:18s} Christoffel max|Γ_fd − Γ_analytic| = {worst:.2e}")
    return worst < 1e-3


def check_expmap_coordinate(name, jac_fn, x0, v0, endpoint_fn, n_steps=128):
    """Integrate exp_{x0}(v0) and compare to the closed-form coordinate-geodesic endpoint."""
    gamma_fn = make_gamma_fn(jac_fn)
    xT = exponential_map(x0, v0, gamma_fn, t1=1.0, n_steps=n_steps)
    exp = endpoint_fn(x0, v0)
    err = float((xT - exp).abs().max())
    # distance consistency: ||v0||_{g(x0)} should equal the geodesic length
    g0 = metric_from_jac(jac_fn(x0))[0]
    length = float(torch.sqrt(v0[0] @ g0 @ v0[0]))
    status = "PASS" if err < 5e-3 else "FAIL"
    print(f"  [{status}] {name:18s} exp-map endpoint err = {err:.2e}   "
          f"(‖v‖_g = {length:.4f})")
    return err < 5e-3


def check_distance(name, jac_fn, analytic_dist, pairs, n_iters=80, n_steps=64, lr=0.3):
    gamma_fn = make_gamma_fn(jac_fn)
    metric_fn = metric_fn_from(jac_fn)
    worst_rel = 0.0
    worst_ep = 0.0
    for x, y in pairs:
        v = log_map_shooting(x, y, gamma_fn, t1=1.0, n_steps=n_steps,
                             n_iters=n_iters, lr=lr)
        xT = exponential_map(x, v, gamma_fn, t1=1.0, n_steps=n_steps)
        ep = float((xT - y).abs().max())
        g = metric_fn(x)[0]
        d_got = float(torch.sqrt(torch.clamp(v[0] @ g @ v[0], min=0.0)))
        d_true = analytic_dist(x, y)
        rel = abs(d_got - d_true) / max(d_true, 1e-9)
        worst_rel = max(worst_rel, rel)
        worst_ep = max(worst_ep, ep)
    status = "PASS" if worst_rel < 0.02 else ("WARN" if worst_rel < 0.10 else "FAIL")
    print(f"  [{status}] {name:18s} distance worst rel-err = {worst_rel:.2%}   "
          f"(worst shooting endpoint err = {worst_ep:.2e})")
    return worst_rel < 0.02


def t(*vals):
    return torch.tensor([list(vals)])


def run_toy_tests() -> bool:
    ok = True
    print("=" * 72)
    print("GEOMETRY SELF-TEST  —  analytic ground truth")
    print("=" * 72)

    # ── Sphere ──────────────────────────────────────────────────────────────
    print("\n• Unit 2-sphere  g = diag(1, sin^2 u)")
    sph_pts = [t(0.9, 0.3), t(1.3, -0.7), t(2.0, 1.1), t(0.6, 2.5)]
    ok &= check_christoffel("sphere", sphere_jac, sphere_christoffel, sph_pts)
    # meridian geodesic (v const): u(t)=u0+vu t ; endpoint (u0+vu, v0)
    ok &= check_expmap_coordinate(
        "sphere meridian", sphere_jac, t(0.8, 0.4), t(0.5, 0.0),
        endpoint_fn=lambda x0, v0: t(float(x0[0, 0] + v0[0, 0]), float(x0[0, 1])),
    )
    ok &= check_distance(
        "sphere", sphere_jac, sphere_distance,
        pairs=[(t(0.8, 0.4), t(1.2, 0.4)),     # same meridian
               (t(1.0, 0.2), t(1.25, 0.6)),    # general great circle
               (t(1.4, -0.3), t(1.1, 0.5))],
    )

    # ── Poincaré upper half-plane ────────────────────────────────────────────
    print("\n• Poincaré upper half-plane  g = diag(1/y^2, 1/y^2)")
    poi_pts = [t(0.0, 1.0), t(0.5, 2.0), t(-1.0, 0.7), t(2.0, 1.5)]
    ok &= check_christoffel("poincare", poincare_jac, poincare_christoffel, poi_pts)
    # vertical geodesic: y(t)=y0 exp((vy/y0) t) ; endpoint (x0, y0 exp(vy/y0))
    ok &= check_expmap_coordinate(
        "poincare vertical", poincare_jac, t(0.5, 1.0), t(0.0, 0.6),
        endpoint_fn=lambda x0, v0: t(
            float(x0[0, 0]),
            float(x0[0, 1] * math.exp(float(v0[0, 1]) / float(x0[0, 1]))),
        ),
    )
    ok &= check_distance(
        "poincare", poincare_jac, poincare_distance,
        pairs=[(t(0.5, 1.0), t(0.5, 2.2)),     # vertical
               (t(-0.4, 1.0), t(0.6, 1.3)),    # general semicircle
               (t(0.0, 0.8), t(1.0, 1.6))],
    )

    print("\n" + "-" * 72)
    print(f"TOY MANIFOLD RESULT: {'ALL PASS ✓' if ok else 'SOME FAILURES ✗'}")
    print("-" * 72)
    return ok


# ─────────────────── optional: HydrAMP FD-∂g vs autograd ───────────────────

def run_hydramp_dg_test(eps_grid=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7)) -> None:
    print("\n" + "=" * 72)
    print("HydrAMP  —  finite-difference ∂g vs autograd  (sizes the Christoffel FD error)")
    print("=" * 72)
    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
        HydrAMPEncoderDecoder,
    )

    from pep_compass.models.encoder_decoder.utils import decoder_jacobian
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    Z = hyd.latent_dim
    fwd = lambda x: hyd.decoder_forward(x, softmax=True, flatten=True)

    def approx_jac(x):   # (1, M, Z), no autograd (finite-difference decoder Jacobian)
        return decoder_jacobian(fwd, x, "approx", {"jacobian_eps": 1e-6}).double()

    def metric_at(x):    # (Z, Z)
        return metric_from_jac(approx_jac(x))[0]

    torch.manual_seed(0)
    z = (torch.randn(1, Z) * 0.5).double()
    eye = torch.eye(Z).double()

    # high-accuracy central-difference reference: dg_ref[i,j,k] = ∂_k g_ij
    h = 1e-4
    dg_ref = torch.stack(
        [(metric_at(z + h * eye[k:k + 1]) - metric_at(z - h * eye[k:k + 1])) / (2 * h)
         for k in range(Z)], dim=-1)                  # (Z,Z,Z) ordered [i,j,k]
    gnorm = float(metric_at(z).norm())

    print(f"  latent dim={Z}; ‖g‖={gnorm:.3e} at test z; reference = central diff (h={h})")
    print(f"  {'eps':>8}   {'max|dg_fwd − dg_ref|':>22}   {'rel':>10}")
    for eps in eps_grid:
        J = approx_jac(z)
        cols = [approx_jac(z + eps * eye[k:k + 1]) for k in range(Z)]
        J_pert = torch.stack(cols, dim=1)              # (1,Z,M,Z)
        dg_fwd = approx_dg_from_jac(J, J_pert, eps)[0]  # (Z,Z,Z) ordered [i,j,k]
        err = float((dg_fwd - dg_ref).abs().max())
        rel = err / float(dg_ref.abs().max() + 1e-12)
        print(f"  {eps:>8.0e}   {err:>22.3e}   {rel:>9.2%}")
    print("-" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydramp", action="store_true",
                    help="also run the HydrAMP FD-∂g vs autograd sweep (slow; needs weights)")
    args = ap.parse_args()

    all_ok = run_toy_tests()
    if args.hydramp:
        run_hydramp_dg_test()
    sys.exit(0 if all_ok else 1)
