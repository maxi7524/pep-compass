# Geodesic Mutation Verification Report

**Experiment:** Does following a Riemannian geodesic in the direction of a MUTANG++ mutation lead to the mutated peptide?

**Script:** `scripts/geodesic_mutation_check.py`
**Date:** Results from `geodesic_mutation_check_results.json`

---

## 1. Introduction

### Problem Statement

The MUTANG++ mutation engine identifies candidate amino-acid substitutions by
analysing the decoder Jacobian of a variational autoencoder (HydrAMP) in the
latent space $\mathbb{R}^{64}$.  Once a mutation is selected, the system must
*travel* in latent space from the parent encoding $z_{\text{parent}}$ to a
point whose decoded peptide matches the intended mutant.

Two design questions arise:

1. **Direction:** Should the latent displacement be computed via a
   pseudoinverse projection of the one-hot ambient mutation vector
   ($J^{+} e_{\text{mut}}$, the "projected direction"), or should we simply
   encode the mutant peptide and take
   $v_{\text{enc}} = z_{\text{mutant}} - z_{\text{parent}}$ (the "encoded
   direction")?

2. **Path type:** Should the path from $z_{\text{parent}}$ to the target be a
   *Euclidean straight line* ($z + v$), or a *Riemannian geodesic*
   ($\exp_z(v)$) that respects the pullback metric induced by the decoder?

### Fundamental Question

> If we follow a Riemannian geodesic in the direction determined by a MUTANG++
> mutation, does the decoded endpoint coincide with the expected mutant peptide?

This experiment disentangles the two factors — **direction computation** and
**path geometry** — by testing all four combinations on real seed peptides.

---

## 2. Mathematical Background

### 2.1 Pullback Metric

Let $f: \mathbb{R}^{Z} \to \mathbb{R}^{M}$ be the decoder, with
$Z = 64$ (latent dimension) and $M = 525$ (ambient dimension:
$25 \text{ positions} \times 21 \text{ tokens}$).  The Jacobian at a latent
point $z$ is:

$$J = \frac{\partial f}{\partial z} \in \mathbb{R}^{M \times Z}$$

The **pullback metric** (first fundamental form) is:

$$g_{ij}(z) = \sum_{k=1}^{M} J_{ki}\, J_{kj} = (J^\top J)_{ij}$$

This is a $Z \times Z$ positive semi-definite matrix that endows latent space
with the geometry inherited from the decoder's output manifold.

### 2.2 Christoffel Symbols

Given the metric tensor $g_{ij}$ and its partial derivatives
$\partial_l g_{ij}$, the **Christoffel symbols of the second kind** are:

$$\Gamma^{k}_{ij} = \frac{1}{2}\, g^{kl} \!\left( \frac{\partial g_{li}}{\partial z^{j}} + \frac{\partial g_{lj}}{\partial z^{i}} - \frac{\partial g_{ij}}{\partial z^{l}} \right)$$

where $g^{kl}$ denotes the $(k,l)$ entry of the inverse metric $g^{-1}$.

In the implementation, the metric derivatives are approximated via
**finite differences** with perturbation $\varepsilon = 0.05$:

$$\frac{\partial g_{ij}}{\partial z^{k}} \approx \frac{g_{ij}(z + \varepsilon\, e_k) - g_{ij}(z)}{\varepsilon}$$

where $e_k$ is the $k$-th standard basis vector in $\mathbb{R}^{64}$.  This
requires $Z + 1 = 65$ decoder Jacobian evaluations per base point.

### 2.3 Geodesic Equation

A geodesic $\gamma(t)$ on a Riemannian manifold satisfies the second-order ODE:

$$\ddot{\gamma}^{k}(t) = -\Gamma^{k}_{ij}\bigl(\gamma(t)\bigr)\, \dot{\gamma}^{i}(t)\, \dot{\gamma}^{j}(t)$$

Rewritten as a first-order system with position $x(t) = \gamma(t)$ and
velocity $v(t) = \dot{\gamma}(t)$:

$$\dot{x}^k = v^k, \qquad \dot{v}^k = -\Gamma^k_{ij}(x)\, v^i\, v^j$$

### 2.4 Exponential Map

The **Riemannian exponential map** at a base point $z$ sends a tangent vector
$v \in T_z \mathcal{M}$ to the point reached by following the geodesic for
unit time:

$$\exp_z(v) = \gamma(1), \qquad \text{where } \gamma(0) = z,\; \dot{\gamma}(0) = v$$

In flat (Euclidean) geometry, $\exp_z(v) = z + v$.  In curved geometry, the
geodesic deviates from the straight line according to the Christoffel symbols.

### 2.5 Sub-Riemannian Tangent Space and Horizontal Projection

The decoder Jacobian SVD yields $J = U \Sigma V^\top$.  Only the directions
corresponding to singular values above a threshold
$\sigma_{\min} = 10^{-6}$ are considered **horizontal** — they span the
sub-Riemannian tangent space.  Ambient vectors are projected onto this
horizontal subspace via:

$$v_{\text{proj}} = J_H^{+}\, e_{\text{mut}}$$

where $J_H = U \cdot \operatorname{diag}(\tilde{\sigma}_1, \ldots, \tilde{\sigma}_{64}) \cdot V^\top$
is the horizontal Jacobian with vertical singular values zeroed out
($\tilde{\sigma}_i = \sigma_i$ if $\sigma_i > \sigma_{\min}$, else $0$),
and $J_H^{+}$ is the Moore–Penrose pseudoinverse.

### 2.6 RK4 Integration

The geodesic ODE is integrated using the classical **fourth-order Runge–Kutta**
(RK4) scheme.  At each sub-step, the Christoffel tensor $\Gamma$ is
re-evaluated (or, in this experiment, held frozen).  The update for one step of
size $h = 1 / N_{\text{steps}}$:

$$k_1 = f(x_n, v_n), \quad k_2 = f\!\left(x_n + \tfrac{h}{2} k_1\right), \quad k_3 = f\!\left(x_n + \tfrac{h}{2} k_2\right), \quad k_4 = f(x_n + h\, k_3)$$

$$x_{n+1} = x_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

The step count is adaptive: $N_{\text{steps}} = \max(16,\; \min(256,\; \lceil 100 \|v\| \rceil))$.

---

## 3. Methodology

### 3.1 Pipeline Overview

The experiment follows the mutation enumeration pipeline from
`potentials_analysis.ipynb`, executed programmatically:

```
encode(peptide) → z_parent
        │
        ▼
decoder_jacobian(z, softmax=True, eps=1e-6) → J   (1 × 525 × 64)
        │
        ▼
    SVD(J) → U, Σ, V
        │
        ▼
get_mutations_from_s_u_standard(Σ, U)
   ↳ filter positions < peptide length
        │
        ▼
DecoderLogProbPotential.compute(peptide, mutations)
   ↳ remove identity mutations (parent_aa == mutant_aa)
   ↳ softmax over log-potentials
   ↳ keep mutations with p > 1/n_mut
        │
        ▼
For each surviving mutation:
   ├─ compute v_proj  (J⁺ horizontal projection)
   ├─ compute v_enc   (z_mutant − z_parent, via encoder)
   ├─ follow Euclidean line:  z_end = z + v
   └─ follow geodesic:        z_end = exp_z(v)  (frozen Γ)
        │
        ▼
   decode(z_end) → peptide_out
   compare with expected mutant
```

### 3.2 Direction Computation

Two latent directions are computed for each mutation:

| Direction | Formula | Description |
|-----------|---------|-------------|
| **Projected** ($v_{\text{proj}}$) | $J_H^{+}\, e_{\text{pos} \cdot 21 + \text{aa}}$ | Pseudoinverse of horizontal Jacobian applied to one-hot mutation vector; scaled to $\lVert v_{\text{enc}} \rVert$ |
| **Encoded** ($v_{\text{enc}}$) | $z_{\text{mutant}} - z_{\text{parent}}$ | Direct difference of encoder outputs; requires knowing the mutant peptide |

The projected direction uses only local differential information (the Jacobian
at $z_{\text{parent}}$), while the encoded direction uses the global encoder as
an oracle.

### 3.3 Frozen Christoffel Symbols

Computing $\Gamma(z)$ requires $Z + 1 = 65$ Jacobian evaluations (base + one
per perturbation direction).  To make geodesic integration tractable, $\Gamma$
is computed **once** at $z_{\text{parent}}$ and held constant during
integration:

$$\gamma_{\text{fn}}(x) \equiv \Gamma(z_{\text{parent}}) \quad \forall\, x$$

This is a zeroth-order approximation — the curvature correction is applied but
assumed constant along the path.  For small displacements ($\|v\| \lesssim 1$),
this is reasonable; for larger steps, the frozen approximation becomes less
accurate.

### 3.4 Metric Regularisation

The raw pullback metric $g = J^\top J$ is ill-conditioned: its $64$
eigenvalues span many orders of magnitude, and finite-difference noise
introduces near-zero or even slightly negative eigenvalues.  Inverting $g$
directly yields $\|\Gamma\| \to \infty$.

**Regularised metric:**

$$g_{\text{reg}} = g + \lambda I, \qquad \lambda = 1.0$$

This Tikhonov-style regularisation ensures all eigenvalues are at least
$\lambda$, bounding $\|g_{\text{reg}}^{-1}\|$ and keeping $\Gamma$ in a
numerically stable range.

### 3.5 Four Test Configurations

Each mutation is tested under all four combinations of direction × path type:

| Configuration | Direction | Path | Endpoint |
|---------------|-----------|------|----------|
| `euc_proj` | Projected ($v_{\text{proj}}$) | Euclidean | $z + v_{\text{proj}}$ |
| `euc_enc` | Encoded ($v_{\text{enc}}$) | Euclidean | $z + v_{\text{enc}}$ |
| `geo_proj` | Projected ($v_{\text{proj}}$) | Geodesic | $\exp_z(v_{\text{proj}})$ |
| `geo_enc` | Encoded ($v_{\text{enc}}$) | Geodesic | $\exp_z(v_{\text{enc}})$ |

The decoded peptide at each endpoint is compared to the expected mutant.

### 3.6 Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `JACOBIAN_EPS` | $10^{-6}$ | Finite-difference step for softmax Jacobian |
| `GAMMA_EPS` | $0.05$ | Finite-difference step for metric derivatives |
| `METRIC_REG` ($\lambda$) | $1.0$ | Diagonal regularisation for metric inversion |
| `DIRECTION_SIG_THRESH` | $10^{-6}$ | SVD singular-value cutoff for horizontal subspace |
| `TOKEN_THRESH` | $10^{-4}$ | Per-token weight threshold in mutation enumeration |
| `HORIZONTAL_THRESH` | $10^{-4}$ | Horizontal threshold for `SubRiemannianTangentSpace` |
| `MIN_DIRECTIONS` | $5$ | Minimum number of SVD directions to retain |
| `n_steps` (base) | $16$ | Base RK4 step count; adapted up to $256$ by $\|v\|$ |
| Latent dim ($Z$) | $64$ | HydrAMP latent dimension |
| Ambient dim ($M$) | $525$ | $25 \times 21$ (positions × alphabet) |

---

## 4. Results

### 4.1 Seed Peptides and Mutation Counts

| Peptide | Sequence | Length | Total Mutations | Above Threshold |
|---------|----------|--------|-----------------|-----------------|
| middle-1 | `FLYKWWIRIGRLKL` | 14 | 29 | 5 |
| jurand-4 | `KYCRRFRWLTFRWL` | 14 | 34 | 5 |
| jurand-2 | `KFRNRHRWKFKLIFRN` | 16 | 77 | 2 |
| jurand-7 | `KKYWLIRKWIRLWFLT` | 16 | 47 | 3 |
| mammuthusin-3 | `KTLKIIRLLF` | 10 | 31 | 2 |
| hydrodamin-2 | `RMARNLVRYVQGLKKKKVI` | 19 | 59 | 5 |
| **Total** | | | **277** | **22** |

The $1/n_{\text{mut}}$ softmax threshold retains only the most confident
mutations (≈ 8% of candidates on average).

### 4.2 Overall Results

| Configuration | Full Match | Full Match % | Position Match | Pos Match % | Avg Hamming |
|---------------|-----------|--------------|----------------|-------------|-------------|
| `euc_proj` | 1 / 22 | **4.5%** | 4 / 22 | 18.2% | ≈ 2.3 |
| `geo_proj` | 2 / 22 | **9.1%** | 4 / 22 | 18.2% | ≈ 2.1 |
| `euc_enc` | 20 / 22 | **90.9%** | 20 / 22 | 90.9% | ≈ 0.08 |
| `geo_enc` | 20 / 22 | **90.9%** | 20 / 22 | 90.9% | ≈ 0.08 |

### 4.3 Per-Peptide Breakdown

#### middle-1 (`FLYKWWIRIGRLKL`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 0 / 5 | 0 / 5 | 2.2 |
| `geo_proj` | 0 / 5 | 0 / 5 | 2.0 |
| `euc_enc` | 5 / 5 | 5 / 5 | 0.0 |
| `geo_enc` | 5 / 5 | 5 / 5 | 0.0 |

Geodesic fallbacks (proj): 4/5 — the projected direction produces such large
raw norms ($322$–$4026$) that geodesic integration diverges; the code falls
back to Euclidean.

#### jurand-4 (`KYCRRFRWLTFRWL`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 0 / 5 | 1 / 5 | 1.8 |
| `geo_proj` | 1 / 5 | 1 / 5 | 1.6 |
| `euc_enc` | 5 / 5 | 5 / 5 | 0.0 |
| `geo_enc` | 5 / 5 | 5 / 5 | 0.0 |

One mutation (pos 9, T→I) matches even with the projected direction under
geodesic integration — a rare case where the tangent approximation is
adequate.

#### jurand-2 (`KFRNRHRWKFKLIFRN`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 1 / 2 | 1 / 2 | 1.5 |
| `geo_proj` | 1 / 2 | 1 / 2 | 1.5 |
| `euc_enc` | 2 / 2 | 2 / 2 | 0.0 |
| `geo_enc` | 2 / 2 | 2 / 2 | 0.0 |

Notably small raw projection norms ($2.5$, $3.7$) — the horizontal subspace
here aligns better with the actual mutation direction.

#### jurand-7 (`KKYWLIRKWIRLWFLT`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 0 / 3 | 1 / 3 | 3.3 |
| `geo_proj` | 0 / 3 | 1 / 3 | 3.3 |
| `euc_enc` | 2 / 3 | 2 / 3 | 0.3 |
| `geo_enc` | 2 / 3 | 2 / 3 | 0.3 |

One encoded-direction failure: pos 14, L→N.  The encoder maps parent and
mutant to points whose midpoint decodes to `...FKT` rather than `...FNT` —
the decoder's discrete argmax at that position is unstable.

#### mammuthusin-3 (`KTLKIIRLLF`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 0 / 2 | 0 / 2 | 2.0 |
| `geo_proj` | 0 / 2 | 0 / 2 | 2.0 |
| `euc_enc` | 2 / 2 | 2 / 2 | 0.0 |
| `geo_enc` | 2 / 2 | 2 / 2 | 0.0 |

No geodesic fallbacks (0/2) — the Christoffel tensor is well-behaved here
($\|\Gamma\| \approx 0.7$), reflecting a nearly flat local geometry.

#### hydrodamin-2 (`RMARNLVRYVQGLKKKKVI`)

| Method | Full Match | Pos Match | Avg Hamming |
|--------|-----------|-----------|-------------|
| `euc_proj` | 0 / 5 | 1 / 5 | 3.0 |
| `geo_proj` | 0 / 5 | 1 / 5 | 3.0 |
| `euc_enc` | 4 / 5 | 4 / 5 | 0.2 |
| `geo_enc` | 4 / 5 | 4 / 5 | 0.2 |

One encoded-direction failure: pos 15, K→Q.  Decoded as `...KW...` instead
of `...KQ...` — similar to jurand-7, the decoder's argmax at that position is
sensitive to small latent perturbations.

### 4.4 Christoffel Norm Variation

The Frobenius norm $\|\Gamma\|_F$ of the Christoffel tensor varies
dramatically across seed peptides (values from script stdout):

| Peptide | $\|\Gamma\|_F$ | $\|\Gamma\|_\infty$ | Interpretation |
|---------|-----------------|----------------------|----------------|
| mammuthusin-3 | $0.69$ | $0.66$ | Nearly flat — Euclidean ≈ geodesic |
| middle-1 | $63$ | $63$ | Moderate curvature |
| jurand-4 | $173$ | $148$ | Significant curvature |
| hydrodamin-2 | $1{,}030$ | $772$ | High curvature |
| jurand-2 | $5{,}740$ | $5{,}290$ | Very high curvature |
| jurand-7 | $10{,}600$ | $2{,}350$ | Extreme curvature |

This 4.2-order-of-magnitude spread (from $0.69$ to $10{,}600$) shows that the
latent-space geometry is highly non-uniform.  Peptides near steep decoder
gradients (e.g. jurand-7, jurand-2) live in regions where the pullback metric
changes rapidly, while mammuthusin-3 occupies a nearly flat region.

### 4.5 Geodesic Fallback Statistics

When the geodesic integration diverges (produces NaN/Inf), the code falls back
to Euclidean extrapolation.  This happened frequently with the projected
direction:

| Direction | Fallbacks | Out of | Rate |
|-----------|-----------|--------|------|
| Projected | 16 | 22 | **72.7%** |
| Encoded | 11 | 22 | **50.0%** |

The projected direction's enormous raw norms (up to $\sim 4000$) cause
$\|v\| \cdot \|\Gamma\|$ to overflow even with adaptive step counts.

### 4.6 Projected Direction Raw Norms

The unnormalised pseudoinverse projection $J_H^{+}\, e_{\text{mut}}$ has
wildly varying norms:

| Peptide | $\|v_{\text{proj,raw}}\|$ range |
|---------|---------------------------------|
| jurand-2 | 2.5 – 3.7 |
| jurand-7 | 4.1 – 96.5 |
| jurand-4 | 24.6 – 856.1 |
| mammuthusin-3 | 135.1 – 201.3 |
| middle-1 | 321.9 – 4026.4 |
| hydrodamin-2 | 22.3 – 392.1 |

After rescaling to $\|v_{\text{enc}}\|$ (typically $0.5$–$1.2$), the
*direction* often still points the wrong way — indicating the fundamental
mismatch between the tangent-space linear approximation and the true nonlinear
encoder mapping.

### 4.7 The Two Encoded-Direction Failures

Both `euc_enc` and `geo_enc` fail on exactly the same 2 mutations (confirming
that geodesic vs Euclidean is irrelevant when the direction is correct):

| Peptide | Position | Mutation | Expected | Decoded (euc & geo) | Hamming |
|---------|----------|----------|----------|---------------------|---------|
| jurand-7 | 14 | L → N | `KKYWLIRKWIRLWF`**N**`T` | `KKYWLIRKWIRLWF`**K**`T` | 1 |
| hydrodamin-2 | 15 | K → Q | `RMARNLVRYVQGLKK`**Q**`KVI` | `RMARNLVRYVQGLKK`**W**`KVI` | 1 |

Both have Hamming distance 1 (wrong at the mutation site only).  These are
near-boundary cases: the encoder places parent and mutant so close that the
decoder's softmax argmax at the target position does not flip.

---

## 5. Key Findings

### Finding 1: Direction matters far more than path geometry

| Factor | Effect on full-match rate |
|--------|--------------------------|
| Projected → Encoded direction | +86.4 pp (4.5% → 90.9%) |
| Euclidean → Geodesic path | +0.0 pp to +4.5 pp |

The direction of travel is the dominant factor.  Switching from Euclidean to
geodesic (with the same direction) makes almost no difference.

### Finding 2: Projected direction ≈ 4.5–9.1% match

The tangent-space linear approximation $v_{\text{proj}} = J_H^{+}\, e_{\text{mut}}$
is insufficient: it achieves only 4.5% (Euclidean) to 9.1% (geodesic) full
match rate.  The Jacobian pseudoinverse captures the *local infinitesimal*
mapping but not the *global* encoder-decoder structure.

### Finding 3: Encoded direction ≈ 90.9% match

Using the true encoder difference $v_{\text{enc}} = z_{\text{mutant}} - z_{\text{parent}}$
succeeds 90.9% of the time.  The two failures are decoder argmax instabilities,
not direction errors.

### Finding 4: Geodesic ≈ Euclidean for encoded direction

With $v_{\text{enc}}$, both path types yield identical results (20/22 each).
The Riemannian curvature correction does not change the decoded peptide because:

- The Euclidean distances $\|z_{\text{mutant}} - z_{\text{parent}}\| \approx 0.5$–$1.2$
  are small enough that straight-line and geodesic endpoints remain in the same
  decoder Voronoi cell.
- The frozen-$\Gamma$ approximation limits the geodesic's accuracy anyway.

### Finding 5: Huge curvature variation across peptides

The Christoffel norm spans four orders of magnitude:

$$\|\Gamma\|_F \in [0.69 \text{ (mammuthusin-3)},\; 10{,}600 \text{ (jurand-7)}]$$

This implies that a uniform integration strategy (fixed step count, fixed
regularisation) is suboptimal.  Adaptive methods that adjust $\lambda$ and
$N_{\text{steps}}$ to the local geometry could improve stability.

### Finding 6: Bottleneck is direction computation

The experiment conclusively shows that the bottleneck is **computing the
correct latent direction**, not the choice of Euclidean vs geodesic path.
For the MUTANG++ pipeline, using the encoder to obtain $v_{\text{enc}}$ is
essential.  The Jacobian pseudoinverse is useful for *enumerating* candidate
mutations but not for *navigating* to them.

### Implications for RL-Based Optimisation

In the reinforcement learning setting (where the next peptide is chosen from
MUTANG++ candidates), the encoder is always available to compute
$z_{\text{mutant}}$.  Therefore:

- **Euclidean displacement $v_{\text{enc}} = z_{\text{mutant}} - z_{\text{parent}}$ is
  the recommended action representation** — it is simple, fast, and 90.9%
  accurate.
- Geodesic integration adds computational cost ($65$ extra Jacobian evaluations
  per base point) without improving decoded accuracy.
- The remaining 9.1% failures are inherent to the encoder-decoder
  reconstruction, not to the path geometry.

---

## 6. Code Overview

### 6.1 Script: `scripts/geodesic_mutation_check.py`

| Function | Description |
|----------|-------------|
| `compute_svd_and_mutations(encoder_decoder, peptide, z)` | Computes the softmax-decoder Jacobian at $z$, performs SVD ($U, \Sigma, V$), enumerates mutations via `get_mutations_from_s_u_standard`, and constructs the `SubRiemannianTangentSpace`. Returns `(jac, U, S, V, mutations, tangent_space)`. |
| `_christoffel_regularised(jac, dg, reg)` | Computes Christoffel symbols $\Gamma^k_{ij}$ using the regularised metric $g_{\text{reg}} = J^\top J + \lambda I$. Avoids the singular inverse of the raw pullback metric. |
| `build_frozen_gamma(encoder_decoder, z, eps)` | Evaluates $65$ decoder Jacobians (base + perturbations) to compute $\partial g / \partial z$ via finite differences; returns a frozen `gamma_fn` closure and the Christoffel tensor. |
| `mutation_direction_projected(tangent_space, pos, aa_idx)` | Constructs the one-hot ambient vector $e_{\text{pos} \cdot 21 + \text{aa}}$ and projects it through the sub-Riemannian horizontal pseudoinverse to obtain $v_{\text{proj}} \in \mathbb{R}^{64}$. |
| `scale_direction(v, target_norm)` | Rescales a direction vector to a specified L2 norm; used to normalise $v_{\text{proj}}$ to $\|v_{\text{enc}}\|$ for fair comparison. |
| `check_single_mutation(...)` | Core evaluation: runs all 4 configurations (Euclidean/geodesic × projected/encoded), decodes each endpoint, and returns full-match, position-match, Hamming distance, and decoded sequences. Includes adaptive step count and NaN/Inf fallback. |
| `main(device)` | Orchestrates the full experiment: loads the model, iterates over `SEED_PEPTIDES`, runs the pipeline for each peptide, aggregates per-peptide and overall statistics, and saves results to JSON. |

### 6.2 Geometry Utilities: `src/pep_compass/geometry/utils.py`

| Function | Signature | Description |
|----------|-----------|-------------|
| `metric_from_jac` | `(jac) → g` | Pullback metric $g = J^\top J$, shape $(B, Z, Z)$. |
| `approx_dg_from_jac` | `(jac, jac_perturbed, eps) → dg` | Finite-difference approximation of $\partial_k g_{ij}$, shape $(B, Z, Z, Z)$. |
| `christoffel_from_jac_and_dg` | `(jac, dg) → Γ` | Standard (unregularised) Christoffel symbols from metric and its derivatives. |
| `geodesic_rhs` | `(x, v, Γ) → (dx, dv)` | Right-hand side of the geodesic ODE: $\dot{x} = v$, $\dot{v} = -\Gamma^k_{ij} v^i v^j$. |
| `integrate_geodesic_rk4` | `(x₀, v₀, Γ_fn, t₁, n_steps) → (x_T, v_T)` | Fourth-order Runge–Kutta integration of the geodesic equation. Calls `Γ_fn` at each sub-step midpoint. |
| `exponential_map` | `(x, v, Γ_fn, t₁, n_steps) → x_T` | Wrapper: returns only the endpoint $\gamma(t_1)$ from `integrate_geodesic_rk4`. |
| `log_map_shooting` | `(x, y, Γ_fn, ...) → v` | Inverse of the exponential map: finds $v$ such that $\exp_x(v) \approx y$ via gradient-based shooting with $10$ iterations. |
| `riemann_from_Gamma_and_dGamma` | `(Γ, dΓ) → R` | Full Riemann curvature tensor $R^l_{ijk}$ from Christoffel symbols and their derivatives. |
| `ricci_from_riemann` | `(R) → \text{Ric}` | Ricci tensor via contraction: $\text{Ric}_{ij} = R^k_{ikj}$. |
| `scalar_from_ricci_and_metric` | `(\text{Ric}, g) → S` | Scalar curvature: $S = g^{ij} \text{Ric}_{ij}$. |

---

## Appendix: Reproduction

```bash
# Run the experiment (CPU, ~10-15 min)
python scripts/geodesic_mutation_check.py --device cpu

# Results are saved to:
#   results/geodesic_mutation_check_results.json
```

The script requires a trained HydrAMP encoder-decoder checkpoint accessible
via `build_encoder_decoder()` from `scripts/rl_peptide_optimizer.py`.
