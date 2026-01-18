import torch
from einops import rearrange, einsum


# ============================================================================
#    Metric from Jacobian
# ============================================================================
def metric_from_jac(jac):
    """
    jac: (B, M, Z)
    returns g: (B, Z, Z)
    """
    return einsum(jac, jac, "b m i, b m j -> b i j")


# ============================================================================
#    Approximate ∂g/∂x from perturbed Jacobians
# ============================================================================
def approx_dg_from_jac(jac, jac_perturbed, eps):
    """
    Approximate dg[b,i,j,k] = ∂_k g[b,i,j] from:
        jac:          (B, M, Z)
        jac_perturbed (B, Z, M, Z) where jac_perturbed[b,k] = J(x + eps * e_k)
    eps: float

    Returns:
        dg: (B, Z, Z, Z)
    """
    # metric at base point
    g0 = einsum(jac, jac, "b m i, b m j -> b i j")  # (B, Z, Z)

    # metric at perturbed points
    gk = einsum(
        jac_perturbed, jac_perturbed, "b k m i, b k m j -> b k i j"
    )  # (B, Z, Z, Z)

    # finite difference
    dg = (gk - g0[:, None]) / eps  # (B, k, i, j)

    return rearrange(dg, "b k i j -> b i j k")  # reorder axes


# ============================================================================
#    Christoffel symbols Γ^k_{ij}
# ============================================================================
def christoffel_from_jac_and_dg(jac, dg):
    """
    jac: (B, M, Z)
    dg:  (B, Z, Z, Z)  with dg[b,i,j,k] = ∂_k g[b,i,j]
    returns Γ: (B, Z, Z, Z)
    """
    g = metric_from_jac(jac)
    g_inv = torch.linalg.inv(g)

    # Build T_{l i j}
    term1 = rearrange(dg, "b l i j -> b l i j")  # ∂_j g_{l i}
    term2 = rearrange(dg, "b l j i -> b l i j")  # ∂_i g_{l j}
    term3 = rearrange(dg, "b i j l -> b l i j")  # ∂_l g_{ij}
    T = term1 + term2 - term3

    # Γ^k_{ij} = 1/2 g^{kl} T_{l i j}
    Gamma = 0.5 * einsum(g_inv, T, "b k l, b l i j -> b k i j")
    return Gamma  # (B, k, i, j)


# ============================================================================
#    Approximate ∂Γ/∂x from perturbed Christoffels
# ============================================================================
def approx_dGamma_from_Gamma(Gamma, Gamma_pert, eps):
    """
    Approximate dGamma[b,l,i,j,k] = ∂_k Γ^l_{ij} from:
        Gamma:       (B, Z, Z, Z)
        Gamma_pert:  (B, Z, Z, Z, Z), where Gamma_pert[b,k] = Γ(x + eps e_k)
    Returns:
        dGamma: (B, Z, Z, Z, Z)
    """
    # finite difference
    dG = (Gamma_pert - Gamma[:, None]) / eps  # (B, k, l, i, j)

    return rearrange(dG, "b k l i j -> b l i j k")


# ============================================================================
#    Riemann curvature R^l_{ijk}
# ============================================================================
def riemann_from_Gamma_and_dGamma(Gamma, dGamma):
    """
    Gamma:  (B, Z, Z, Z)
    dGamma: (B, Z, Z, Z, Z) with ∂_k Γ^l_{ij}
    returns R: (B, Z, Z, Z, Z)
    """
    # ∂_j Γ^l_{ik}
    term_d1 = rearrange(dGamma, "b l i k j -> b l i j k")
    # ∂_k Γ^l_{ij}
    term_d2 = rearrange(dGamma, "b l i j k -> b l i j k")

    # Γ^m_{ik} Γ^l_{mj}
    prod1 = einsum(Gamma, Gamma, "b m i k, b l m j -> b l i j k")
    # Γ^m_{ij} Γ^l_{mk}
    prod2 = einsum(Gamma, Gamma, "b m i j, b l m k -> b l i j k")

    return term_d1 - term_d2 + prod1 - prod2


# ============================================================================
#    Ricci and scalar curvature
# ============================================================================
def ricci_from_riemann(R):
    """Ric_{ij} = R^k_{ikj}"""
    return einsum(R, "b k i k j -> b i j")


def scalar_from_ricci_and_metric(Ric, g):
    """S = g^{ij} Ric_{ij}"""
    g_inv = torch.linalg.inv(g)
    return einsum(g_inv, Ric, "b i j, b i j -> b")


# ============================================================================
#    Geodesic RHS
# ============================================================================
def geodesic_rhs(x, v, Gamma):
    """
    x is ignored (functional geometry)
    v: (B, Z)
    Gamma: (B, Z, Z, Z)
    """
    q = einsum(v, v, "b i, b j -> b i j")
    dv = -einsum(Gamma, q, "b k i j, b i j -> b k")
    dx = v
    return dx, dv


# ============================================================================
#    Geodesic RK4 Integration
# ============================================================================
def integrate_geodesic_rk4(x0, v0, Gamma_fn, t1=1.0, n_steps=32):
    """
    Gamma_fn(x): returns Γ(x) of shape (B, Z, Z, Z)
    """
    x = x0
    v = v0
    h = t1 / n_steps

    for _ in range(n_steps):
        Gamma = Gamma_fn(x)
        k1_x, k1_v = geodesic_rhs(x, v, Gamma)

        Gamma2 = Gamma_fn(x + 0.5 * h * k1_x)
        k2_x, k2_v = geodesic_rhs(x + 0.5 * h * k1_x, v + 0.5 * h * k1_v, Gamma2)

        Gamma3 = Gamma_fn(x + 0.5 * h * k2_x)
        k3_x, k3_v = geodesic_rhs(x + 0.5 * h * k2_x, v + 0.5 * h * k2_v, Gamma3)

        Gamma4 = Gamma_fn(x + h * k3_x)
        k4_x, k4_v = geodesic_rhs(x + h * k3_x, v + h * k3_v, Gamma4)

        x = x + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v = v + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x, v


# ============================================================================
#    Exponential and log maps
# ============================================================================
def exponential_map(x, v, Gamma_fn, t1=1.0, n_steps=32):
    xT, _ = integrate_geodesic_rk4(x, v, Gamma_fn, t1, n_steps)
    return xT


def log_map_shooting(x, y, Gamma_fn, t1=1.0, n_steps=32, n_iters=10, lr=0.5):
    """
    Solve for v such that exp_x(v) ≈ y.
    """
    v = (y - x).detach()

    for _ in range(n_iters):
        v = v.detach().requires_grad_(True)
        xT = exponential_map(x, v, Gamma_fn, t1, n_steps)
        loss = 0.5 * ((xT - y) ** 2).sum(dim=1).mean()
        (grad_v,) = torch.autograd.grad(loss, v)
        v = v - lr * grad_v

    return v.detach()
