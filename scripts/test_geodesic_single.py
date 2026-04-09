"""
Focused geodesic test: one peptide, one mutation from MUTANG++.

Tests whether following a geodesic in the SVD-projected direction
(J_H+ @ e_mut) actually decodes to the expected mutant peptide.

Methods tested:
  1. euc_proj         — Euclidean line in projected direction (J_H+ @ e_mut)
  2. euc_enc          — Euclidean line in encoder-difference direction
  3. geo_frozen_proj  — Geodesic with Γ frozen at z_parent, projected dir
  4. geo_frozen_enc   — Geodesic with Γ frozen at z_parent, encoded dir
  5. geo_live_proj    — Geodesic with Γ recomputed at each major step
  6. geo_reproj       — Geodesic with Γ AND direction recomputed each step
  7. euc_grad1        — Euclidean in gradient direction ∂ log p(mut|z)/∂z
  8. grad_ascent      — Multi-step gradient ascent on decoder log-prob
  9. euler_reproj_N   — Euler re-projection flow (N steps)

Extra analysis:
  --distances         — Compute mean ‖z_mut − z_parent‖ for ALL mutations
                        and compare it against mean distance of mutations with
                        MUTANG++ potential above multiple score thresholds,
                        across all six seed peptides. Saves plot + JSON.

Run:
    python scripts/test_geodesic_single.py [--peptide jurand-2] [--device cpu]
    python scripts/test_geodesic_single.py --distances [--device cpu]
"""

from __future__ import annotations
import argparse, os, sys, time, json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from rl_peptide_optimizer import (
    LATENT_DIM,
    AMBIENT_DIM,
    SEED_PEPTIDES,
    ALPHABET,
    MAX_PEPTIDE_LEN,
    build_encoder_decoder,
)
from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.local_enumeration.mutation.utils import get_mutations_from_s_u_standard
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    ProjectedDirectionPairwiseSimilarityPotential,
)
from pep_compass.geometry.utils import integrate_geodesic_rk4

from geodesic_mutation_check import (
    compute_svd_and_mutations,
    build_frozen_gamma,
    mutation_direction_projected,
    scale_direction,
    exponential_map,
    geodesic_live_gamma,
    geodesic_reprojection,
    euler_reprojection,
    decoder_logprob_gradient,
    gradient_ascent_mutation,
    JACOBIAN_EPS,
    DIRECTION_SIG_THRESH,
    TOKEN_THRESH,
    HORIZONTAL_THRESH,
    MIN_DIRECTIONS,
    METRIC_REG,
)


def decode_and_compare(encoder_decoder, z_end, mutant_peptide, pos, aa_idx, label):
    z_2d = z_end.unsqueeze(0) if z_end.ndim == 1 else z_end  # type: ignore[assignment]
    with torch.no_grad():
        decoded = encoder_decoder.decode_peptides(z_2d)[0]
    d = decoded.strip()
    m = mutant_peptide.strip()
    hamming = sum(a != b for a, b in zip(d, m)) + abs(len(d) - len(m))
    full_match = d == m
    pos_ok = len(d) > pos and d[pos] == ALPHABET[aa_idx]
    status = (
        "✓ FULL MATCH"
        if full_match
        else (f"△ pos_ok={pos_ok}" if pos_ok else "✗ mismatch")
    )
    print(f"  [{label:<22}]  decoded={d!r:25s}  hamming={hamming}  {status}")
    return full_match, hamming, decoded


def _parse_thresholds(thresholds: str) -> list[float]:
    """Parse comma-separated numeric thresholds."""
    if not thresholds.strip():
        return []
    parsed: list[float] = []
    for token in thresholds.split(","):
        t = token.strip()
        if not t:
            continue
        parsed.append(float(t))
    return sorted(set(parsed))


def _normalise_mutations(
    mutations: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Deduplicate and sort mutation options per position."""
    normalised: dict[int, list[int]] = {}
    for pos, aa_indices in mutations.items():
        uniq = sorted({int(a) for a in aa_indices})
        if uniq:
            normalised[int(pos)] = uniq
    return normalised


def _single_mutation_similarity_scores(
    peptide: str,
    mutations: dict[int, list[int]],
    tangent_space: SubRiemannianTangentSpace,
) -> list[tuple[int, int, float]]:
    """Return (pos, aa_idx, potential) using similarity-matrix potential.

    Uses ``ProjectedDirectionPairwiseSimilarityPotential`` with the class default
    scoring function and extracts potentials of single-mutation tuples from
    ``compute_with_identities``.
    """
    padded = peptide.ljust(MAX_PEPTIDE_LEN)
    muts = _normalise_mutations(mutations)
    if not muts:
        return []

    sim_potential = ProjectedDirectionPairwiseSimilarityPotential(
        tangent_space=tangent_space,
        alphabet=ALPHABET,
    )
    positions = sorted(muts.keys())
    parent_tuple = [ALPHABET.index(padded[pos]) for pos in positions]
    pos_to_dim = {pos: i for i, pos in enumerate(positions)}

    # Evaluate each single mutation in full parent context without exploding the
    # full Cartesian product: for non-target positions keep only parent residue.
    parent_by_pos = {pos: parent_tuple[pos_to_dim[pos]] for pos in positions}
    flat: list[tuple[int, int, float]] = []
    for pos in positions:
        dim = pos_to_dim[pos]
        parent_aa_idx = parent_tuple[dim]
        pos_candidates = [aa for aa in muts[pos] if aa != parent_aa_idx]
        if not pos_candidates:
            continue

        local_mutations: dict[int, list[int]] = {
            p: ([aa for aa in pos_candidates] if p == pos else [parent_by_pos[p]])
            for p in positions
        }
        local_scores = sim_potential.compute_with_identities(peptide, local_mutations)
        if not local_scores:
            continue

        for aa_idx in muts[pos]:
            if aa_idx == parent_aa_idx:
                continue
            aa_tuple = list(parent_tuple)
            aa_tuple[dim] = aa_idx
            score = local_scores.get(tuple(aa_tuple))
            if score is None:
                continue
            flat.append((pos, aa_idx, float(score)))
    return flat


def run_distance_analysis(
    device: str = "cpu",
    potential_thresholds: str = "",
    auto_threshold_count: int = 9,
    out_dir: str = "results",
) -> None:
    """Compare mean mutation distance vs potential-score thresholds.

    For each seed peptide:
    - Compute mean distance across all mutations: mean ||z_mut - z_parent||
    - For each potential score threshold τ, compute mean distance for mutations
      with log-potential >= τ.
    - Save a plot of threshold-vs-mean-distance with horizontal baseline at
      mean distance of all mutations.
    """
    print("=" * 72)
    print("DISTANCE ANALYSIS: all mutations  vs  similarity-potential thresholds")
    print("=" * 72)

    os.makedirs(out_dir, exist_ok=True)
    encoder_decoder = build_encoder_decoder(device)

    per_peptide_data: list[dict] = []
    all_potentials_global: list[float] = []

    # ── collect mutation distances + potentials per peptide ─────────────────
    for pep_name, peptide in SEED_PEPTIDES.items():
        padded = peptide.ljust(MAX_PEPTIDE_LEN)
        with torch.no_grad():
            z_parent = encoder_decoder.encode_peptides([peptide])[0]

        _, _U, _S, _V, mutations, tangent_space = compute_svd_and_mutations(
            encoder_decoder, peptide, z_parent
        )
        if not mutations:
            print(f"  {pep_name}: no mutations — skipping")
            continue

        # Use similarity-matrix potential with default scoring transform.
        flat = _single_mutation_similarity_scores(peptide, mutations, tangent_space)
        mut_distances: list[float] = []
        mut_potentials: list[float] = []

        for pos, aa_idx, pot in flat:
            mut_list = list(peptide)
            mut_list[pos] = ALPHABET[aa_idx]
            mutant_peptide = "".join(mut_list)
            with torch.no_grad():
                z_mut = encoder_decoder.encode_peptides([mutant_peptide])[0]
            d = torch.norm(z_mut - z_parent).item()
            mut_distances.append(float(d))
            mut_potentials.append(float(pot))

        if not mut_distances:
            print(f"  {pep_name}: no non-identity mutations — skipping")
            continue

        dists_np = np.array(mut_distances, dtype=np.float64)
        pots_np = np.array(mut_potentials, dtype=np.float64)

        mean_all = float(np.mean(dists_np))
        std_all = float(np.std(dists_np))
        all_potentials_global.extend(mut_potentials)

        per_peptide_data.append(
            {
                "name": pep_name,
                "peptide": peptide,
                "distances": dists_np,
                "potentials": pots_np,
                "mean_all": mean_all,
                "std_all": std_all,
            }
        )

        print(f"\n  {pep_name} ({peptide})")
        print(
            f"    All mutations ({len(dists_np):>3}): "
            f"mean={mean_all:.4f}  std={std_all:.4f}  "
            f"[{dists_np.min():.4f}, {dists_np.max():.4f}]"
        )
        print(
            f"    Similarity potential range: [{pots_np.min():.4f}, {pots_np.max():.4f}]"
        )

    if not per_peptide_data:
        print("\nNo peptide data collected; aborting.")
        print(f"\n{'='*72}")
        return

    # ── choose thresholds ────────────────────────────────────────────────────
    manual_thresholds = _parse_thresholds(potential_thresholds)
    if manual_thresholds:
        thresholds = np.array(manual_thresholds, dtype=np.float64)
        threshold_mode = "manual"
    else:
        if auto_threshold_count < 2:
            auto_threshold_count = 2
        qs = np.linspace(0.05, 0.95, auto_threshold_count)
        thresholds = np.quantile(np.array(all_potentials_global), qs)
        thresholds = np.unique(np.round(thresholds, 8))
        threshold_mode = "auto_global_quantile"

    print(f"\nThreshold mode: {threshold_mode}")
    print(f"Thresholds ({len(thresholds)}): {np.round(thresholds, 4).tolist()}")

    # ── compute threshold curves ─────────────────────────────────────────────
    for item in per_peptide_data:
        dists = item["distances"]
        pots_np = item["potentials"]
        mean_by_thr: list[float] = []
        n_by_thr: list[int] = []
        for thr in thresholds:
            mask = pots_np >= thr
            n_kept = int(mask.sum())
            n_by_thr.append(n_kept)
            if n_kept > 0:
                mean_by_thr.append(float(np.mean(dists[mask])))
            else:
                mean_by_thr.append(float("nan"))
        item["thresholds"] = thresholds.copy()
        item["mean_by_threshold"] = np.array(mean_by_thr, dtype=np.float64)
        item["n_by_threshold"] = np.array(n_by_thr, dtype=np.int64)

    # ── plot 2x3 panels (one per seed peptide) ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, item in zip(axes_flat, per_peptide_data):
        thr = item["thresholds"]
        mean_thr = item["mean_by_threshold"]
        n_thr = item["n_by_threshold"]
        mean_all = item["mean_all"]

        ax.plot(
            thr,
            mean_thr,
            marker="o",
            linewidth=1.8,
            label="mean dist (potential >= threshold)",
        )
        ax.axhline(
            y=mean_all,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label="mean dist (all mutations)",
        )
        ax.set_title(f"{item['name']}  (n={len(item['distances'])})")
        ax.set_xlabel("MutAgg+++ similarity potential threshold")
        ax.set_ylabel("Mean ||z_mut - z_parent||")
        ax.grid(alpha=0.25)

        # annotate right-most kept-count to track support at strict thresholds
        right_idx = len(thr) - 1
        if right_idx >= 0:
            ax.text(
                0.98,
                0.04,
                f"kept@max_thr={int(n_thr[right_idx])}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )

    # hide unused panels if any
    for ax in axes_flat[len(per_peptide_data):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Mutation distance vs MutAgg+++ similarity potential threshold (6 seed peptides)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(out_dir, "geodesic_distance_thresholds.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # ── save numeric data ────────────────────────────────────────────────────
    out_json = {
        "potential_type": "ProjectedDirectionPairwiseSimilarityPotential",
        "potential_scoring": "default similarity_transform in class",
        "threshold_mode": threshold_mode,
        "thresholds": [float(x) for x in thresholds.tolist()],
        "peptides": [],
    }
    for item in per_peptide_data:
        out_json["peptides"].append(
            {
                "name": item["name"],
                "peptide": item["peptide"],
                "n_all": int(len(item["distances"])),
                "mean_all": float(item["mean_all"]),
                "std_all": float(item["std_all"]),
                "mean_by_threshold": [
                    (None if np.isnan(v) else float(v))
                    for v in item["mean_by_threshold"].tolist()
                ],
                "n_by_threshold": [int(v) for v in item["n_by_threshold"].tolist()],
            }
        )
    json_path = os.path.join(out_dir, "geodesic_distance_thresholds.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=2)

    print(f"\nSaved plot : {plot_path}")
    print(f"Saved data : {json_path}")
    print(f"\n{'='*72}")


def run_test(
    peptide_name: str,
    device: str,
    n_major_steps: int = 5,
    n_euler_steps: int = 20,
    grad_steps: int = 150,
):
    peptide = SEED_PEPTIDES[peptide_name]
    print("=" * 72)
    print(f"Geodesic SVD-direction test  |  peptide={peptide_name}  ({peptide})")
    print("=" * 72)

    encoder_decoder = build_encoder_decoder(device)
    padded = peptide.ljust(MAX_PEPTIDE_LEN)

    # ── Encode parent ──────────────────────────────────────────────────────
    with torch.no_grad():
        z_parent = encoder_decoder.encode_peptides([peptide])[0]
    print(f"\nz_parent norm: {torch.norm(z_parent).item():.4f}")

    # ── SVD + mutations ────────────────────────────────────────────────────
    print("\n[1/4] Computing Jacobian SVD and MUTANG++ mutations …")
    t0 = time.time()
    jac, U, S, V, mutations, tangent_space = compute_svd_and_mutations(
        encoder_decoder, peptide, z_parent
    )
    print(
        f"      Done in {time.time()-t0:.2f}s | {len(mutations)} positions | "
        f"horizontal_dim={tangent_space.horizontal_dim}"
    )
    print(
        f"      Singular values (top 8): "
        f"{S[0].detach().cpu().numpy()[:8].round(4).tolist()}"
    )

    # ── Potentials ─────────────────────────────────────────────────────────
    flat = _single_mutation_similarity_scores(peptide, mutations, tangent_space)
    log_pots = np.array([m[2] for m in flat], dtype=np.float64)
    shifted = log_pots - log_pots.max()
    softmax_probs = np.exp(shifted) / np.exp(shifted).sum()
    n_mut = len(flat)
    above = [(m, p) for m, p in zip(flat, softmax_probs) if p > 1.0 / n_mut]

    print(
        f"\n      Total mutations: {n_mut}  |  above 1/{n_mut} threshold: {len(above)}"
    )
    if not above:
        print("No above-threshold mutations — aborting.")
        return

    # ── Christoffel at parent (frozen) ─────────────────────────────────────
    print("\n[2/4] Computing Christoffel symbols at z_parent (frozen Γ) …")
    t0 = time.time()
    gamma_fn_frozen, Gamma_frozen = build_frozen_gamma(encoder_decoder, z_parent)
    gamma_norm = torch.norm(Gamma_frozen).item()
    gamma_max = torch.abs(Gamma_frozen).max().item()
    print(
        f"      Done in {time.time()-t0:.2f}s | "
        f"‖Γ‖_F={gamma_norm:.2e}  ‖Γ‖_∞={gamma_max:.2e}"
    )

    # ── Test each above-threshold mutation ─────────────────────────────────
    print("\n[3/4] Testing mutations …")
    for idx, ((pos, aa_idx, log_pot), softmax_p) in enumerate(above):
        mut_list = list(peptide)
        mut_list[pos] = ALPHABET[aa_idx]
        mutant_peptide = "".join(mut_list)

        with torch.no_grad():
            z_mutant = encoder_decoder.encode_peptides([mutant_peptide])[0]

        v_proj_raw = mutation_direction_projected(tangent_space, pos, aa_idx)
        eucl_dist = torch.norm(z_mutant - z_parent).item()
        v_proj = scale_direction(v_proj_raw, eucl_dist)
        v_enc = z_mutant - z_parent

        cos_proj_enc = (
            torch.dot(v_proj_raw, v_enc)
            / (torch.norm(v_proj_raw) * torch.norm(v_enc) + 1e-12)
        ).item()

        print(f"\n{'─'*72}")
        print(
            f"Mutation [{idx+1}/{len(above)}]:  pos={pos}  "
            f"{padded[pos]}→{ALPHABET[aa_idx]}  "
            f"log_pot={log_pot:.4f}  p_softmax={softmax_p:.4f}"
        )
        print(f"  Expected mutant : {mutant_peptide!r}")
        print(f"  ‖v_enc‖          = {eucl_dist:.4f}   (encoder difference)")
        print(
            f"  ‖v_proj_raw‖     = {torch.norm(v_proj_raw).item():.4f}  (J_H+ @ e_mut, unscaled)"
        )
        print(
            f"  cos(v_proj, v_enc)  = {cos_proj_enc:+.4f}  "
            f"({'aligned' if cos_proj_enc > 0.7 else 'misaligned'})"
        )

        # Gradient direction ∂ log p(mut_aa | z, pos) / ∂z
        v_grad_raw = decoder_logprob_gradient(encoder_decoder, z_parent, pos, aa_idx)
        v_grad = scale_direction(v_grad_raw, eucl_dist)
        cos_grad_enc = (
            torch.dot(v_grad_raw, v_enc)
            / (torch.norm(v_grad_raw) * torch.norm(v_enc) + 1e-12)
        ).item()
        cos_grad_proj = (
            torch.dot(v_grad_raw, v_proj_raw)
            / (torch.norm(v_grad_raw) * torch.norm(v_proj_raw) + 1e-12)
        ).item()
        print(
            f"  ‖v_grad‖         = {torch.norm(v_grad_raw).item():.4f}  (∂ log p/∂z, unscaled)"
        )
        print(
            f"  cos(v_grad, v_enc)  = {cos_grad_enc:+.4f}  "
            f"({'aligned' if cos_grad_enc > 0.7 else 'misaligned'})"
        )
        print(f"  cos(v_grad, v_proj) = {cos_grad_proj:+.4f}")
        print()

        # ── 1. Euclidean in projected direction ────────────────────────────
        z_euc_proj = z_parent.unsqueeze(0) + v_proj.unsqueeze(0)
        decode_and_compare(
            encoder_decoder, z_euc_proj, mutant_peptide, pos, aa_idx, "euc_proj"
        )

        # ── 2. Euclidean in encoded direction ──────────────────────────────
        z_euc_enc = z_parent.unsqueeze(0) + v_enc.unsqueeze(0)
        decode_and_compare(
            encoder_decoder, z_euc_enc, mutant_peptide, pos, aa_idx, "euc_enc"
        )

        # ── 3. Euclidean in gradient direction ─────────────────────────────
        z_euc_grad = z_parent.unsqueeze(0) + v_grad.unsqueeze(0)
        decode_and_compare(
            encoder_decoder, z_euc_grad, mutant_peptide, pos, aa_idx, "euc_grad1"
        )

        # ── 4. Geodesic, frozen Γ, projected direction ────────────────────
        t0 = time.time()
        v_norm = torch.norm(v_proj).item()
        adaptive_steps = max(16, min(256, int(v_norm * 100)))
        z_geo_frozen = exponential_map(
            z_parent.unsqueeze(0),
            v_proj.unsqueeze(0),
            gamma_fn_frozen,
            t1=1.0,
            n_steps=adaptive_steps,
        )
        if torch.isnan(z_geo_frozen).any() or torch.isinf(z_geo_frozen).any():
            print(
                f"  [geo_frozen_proj       ]  *** DIVERGED — fallback to Euclidean ***"
            )
            z_geo_frozen = z_parent.unsqueeze(0) + v_proj.unsqueeze(0)
        decode_and_compare(
            encoder_decoder,
            z_geo_frozen,
            mutant_peptide,
            pos,
            aa_idx,
            f"geo_frozen_proj (n={adaptive_steps})",
        )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

        # ── 4. Geodesic, frozen Γ, encoded direction ──────────────────────
        t0 = time.time()
        v_enc_norm = torch.norm(v_enc).item()
        enc_steps = max(16, min(256, int(v_enc_norm * 100)))
        z_geo_enc_frozen = exponential_map(
            z_parent.unsqueeze(0),
            v_enc.unsqueeze(0),
            gamma_fn_frozen,
            t1=1.0,
            n_steps=enc_steps,
        )
        if torch.isnan(z_geo_enc_frozen).any() or torch.isinf(z_geo_enc_frozen).any():
            print(f"  [geo_frozen_enc        ]  *** DIVERGED — fallback ***")
            z_geo_enc_frozen = z_parent.unsqueeze(0) + v_enc.unsqueeze(0)
        decode_and_compare(
            encoder_decoder,
            z_geo_enc_frozen,
            mutant_peptide,
            pos,
            aa_idx,
            f"geo_frozen_enc (n={enc_steps})",
        )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

        # ── 5. Geodesic with live Γ, projected direction ──────────────────
        print(f"  Computing geodesic with live Γ ({n_major_steps} steps) …")
        t0 = time.time()
        v_proj_for_live = scale_direction(v_proj_raw, eucl_dist)
        z_geo_live = geodesic_live_gamma(
            encoder_decoder,
            z_parent,
            v_proj_for_live,
            n_major_steps=n_major_steps,
            rk4_substeps=4,
        )
        if torch.isnan(z_geo_live).any() or torch.isinf(z_geo_live).any():
            print(f"  [geo_live_proj         ]  *** DIVERGED ***")
        else:
            decode_and_compare(
                encoder_decoder,
                z_geo_live,
                mutant_peptide,
                pos,
                aa_idx,
                f"geo_live_proj (k={n_major_steps})",
            )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

        # ── 6. Geodesic with live Γ AND re-projected direction ─────────────
        print(f"  Computing geodesic with live Γ + reproj ({n_major_steps} steps) …")
        t0 = time.time()
        z_geo_reproj = geodesic_reprojection(
            encoder_decoder,
            z_parent,
            pos,
            aa_idx,
            total_dist=eucl_dist,
            n_major_steps=n_major_steps,
            rk4_substeps=4,
        )
        if torch.isnan(z_geo_reproj).any() or torch.isinf(z_geo_reproj).any():
            print(f"  [geo_reproj            ]  *** DIVERGED ***")
        else:
            decode_and_compare(
                encoder_decoder,
                z_geo_reproj,
                mutant_peptide,
                pos,
                aa_idx,
                f"geo_reproj (k={n_major_steps})",
            )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

        # ── 7. Euler re-projection flow ────────────────────────────────────
        print(f"  Computing Euler re-projection flow ({n_euler_steps} steps) …")
        t0 = time.time()
        z_euler, _ = euler_reprojection(
            encoder_decoder,
            z_parent,
            pos,
            aa_idx,
            total_dist=eucl_dist,
            n_steps=n_euler_steps,
        )
        decode_and_compare(
            encoder_decoder,
            z_euler,
            mutant_peptide,
            pos,
            aa_idx,
            f"euler_reproj (n={n_euler_steps})",
        )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

        # ── 8. Gradient ascent (multi-step) ────────────────────────────────
        print(f"  Computing gradient ascent ({grad_steps} steps) …")
        t0 = time.time()
        z_grd = gradient_ascent_mutation(
            encoder_decoder,
            z_parent,
            pos,
            aa_idx,
            padded,
            n_steps=grad_steps,
        )
        decode_and_compare(
            encoder_decoder,
            z_grd,
            mutant_peptide,
            pos,
            aa_idx,
            f"grad_ascent ({grad_steps})",
        )
        print(f"    └─ elapsed: {time.time()-t0:.2f}s")

    print(f"\n{'='*72}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peptide",
        type=str,
        default="jurand-2",
        choices=list(SEED_PEPTIDES.keys()),
        help="Seed peptide to test",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--major-steps",
        type=int,
        default=5,
        help="Number of major steps for live-Γ geodesics",
    )
    parser.add_argument(
        "--euler-steps",
        type=int,
        default=20,
        help="Number of Euler re-projection steps (default 20)",
    )
    parser.add_argument(
        "--grad-steps",
        type=int,
        default=150,
        help="Number of gradient ascent steps (default 150)",
    )
    parser.add_argument(
        "--distances",
        action="store_true",
        help="Run distance analysis across all peptides instead",
    )
    parser.add_argument(
        "--potential-thresholds",
        type=str,
        default="",
        help="Comma-separated MUTANG++ potential thresholds, e.g. '-8,-7,-6'. "
        "If empty, global quantile thresholds are used.",
    )
    parser.add_argument(
        "--auto-threshold-count",
        type=int,
        default=9,
        help="Number of automatic global thresholds when --potential-thresholds is empty.",
    )
    parser.add_argument(
        "--distance-out-dir",
        type=str,
        default="results",
        help="Output directory for threshold distance plot and JSON.",
    )
    args = parser.parse_args()

    if args.distances:
        run_distance_analysis(
            device=args.device,
            potential_thresholds=args.potential_thresholds,
            auto_threshold_count=args.auto_threshold_count,
            out_dir=args.distance_out_dir,
        )
    else:
        run_test(
            args.peptide,
            args.device,
            n_major_steps=args.major_steps,
            n_euler_steps=args.euler_steps,
            grad_steps=args.grad_steps,
        )
