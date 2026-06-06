"""RESCUE -- Riemannian Energy-based Selection of Candidates Under Enumeration.

RQ4/RQ5 showed that the pairwise-coherence TANDEM potentials do not align with the geodesic
feasibility distance, and the cache check shows *why*: within a fixed mutation count the first-order
pullback distance d_G barely tracks d_geo (Spearman ~0.05, the parent metric is degenerate), whereas
the plain Euclidean latent distance tracks it well (~0.4). RESCUE turns this into an analytic,
prune-on-the-Cartesian-product potential.

For a candidate that applies the substitution set S, the first-order latent displacement is the sum
of the per-mutation latent pull-backs of the *diff* directions,
    Dz(S) = sum_{i in S} J_h^+ (e_{l_i,a_i} - e_{l_i,p_{l_i}}),
and RESCUE scores the candidate by its squared latent displacement energy
    E(S) = || Dz(S) ||^2 = sum_i G_ii + 2 sum_{i<j} G_ij ,   G_ij = <v_i, v_j>.
Smaller E = smaller latent move = closer / more feasible.  Two metric choices for v_i:
    whitened  (J_h^+ e, i.e. the columns of the projection matrix)  -> predicts Euclidean latent dist,
    pullback  (rows of U_h)                                         -> predicts d_G (ablation).
The whitened energy is the recommended RESCUE; both are O(1) per candidate from a precomputed set of
per-mutation latent vectors, so the whole Cartesian product is scored without re-encoding.

Reuses the cached candidates + geodesic distances of the RQ4 set (``_thesis_rq5_distances.parquet``);
no geodesic is recomputed.  Writes ``rq5_rescue.pdf`` and ``_thesis_rq5_rescue.csv``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, save  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fig_rq56_potentials import HydrAMPEncoderDecoder, setup, ALPHABET, MAXLEN  # noqa: E402
from pep_compass.local_enumeration.mutation.mutation_potentials import (  # noqa: E402
    ProjectedDirectionPairwiseSimilarityPotential,
    AmbientMetricPairwiseSimilarityPotential,
)

NMUTS = (2, 3, 4)
MIN_GROUP = 12
A = len(ALPHABET)
# methods to compare: name -> (column in merged df, +1 if higher=closer i.e. a potential, -1 if a distance)
# RESCUE energies are distances (smaller = closer); cached distances likewise; potentials higher=better.
rng = np.random.default_rng(0)


def rescue_energies(hyd, pep, z, ts, positions, parent_aa, cand, muts):
    """Squared net latent-displacement energy ||sum_i v_i||^2 per candidate, for three choices of the
    per-mutation latent vector v_i:
      whitened  : first-order J_h^+ (diff)   -- columns of the projection matrix (analytic),
      pullback  : first-order rows of U_h    -- the (degenerate) pullback metric (analytic ablation),
      additive  : the *true* encoded single-mutant displacement z_i - z (K encodes/peptide).
    All three are O(1) per candidate from precomputed per-position lookups."""
    potW = ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode="diff")
    potB = AmbientMetricPairwiseSimilarityPotential(ts, direction_mode="diff")
    potW._ensure_projection()
    z0 = z.detach().cpu().numpy()[0]
    latent = z0.shape[0]
    padded = pep.ljust(MAXLEN)

    # --- precompute the true encoded single-mutant displacements (one batch of K encodes) ---
    single_seqs, single_key = [], []
    for k, p in enumerate(positions):
        for a in muts[p]:
            if a == parent_aa[k]:
                continue
            s = list(padded); s[p] = ALPHABET[a]
            single_seqs.append("".join(s[:len(pep)]))
            single_key.append((k, int(a)))
    add_lut = [np.zeros((A, latent)) for _ in positions]
    if single_seqs:
        with torch.no_grad():
            zz = hyd.encode_peptides(single_seqs).detach().cpu().numpy()
        for idx, (k, a) in enumerate(single_key):
            add_lut[k][a] = zz[idx] - z0

    sumW = sumB = sumA = None
    for k, p in enumerate(positions):
        flat = torch.arange(A, dtype=torch.long) + p * A
        parent_flat = torch.tensor([p * A + parent_aa[k]], dtype=torch.long)
        rawW = potW._raw_directions(flat).detach().cpu().numpy()
        rawW = rawW - potW._raw_directions(parent_flat).detach().cpu().numpy()   # (A, latent)
        rawB = potB._raw_directions(flat).detach().cpu().numpy()
        rawB = rawB - potB._raw_directions(parent_flat).detach().cpu().numpy()   # (A, horiz)
        cW = rawW[cand[:, k]]; cB = rawB[cand[:, k]]; cA = add_lut[k][cand[:, k]]
        sumW = cW if sumW is None else sumW + cW
        sumB = cB if sumB is None else sumB + cB
        sumA = cA if sumA is None else sumA + cA
    return (sumW ** 2).sum(1), (sumB ** 2).sum(1), (sumA ** 2).sum(1)


def main():
    dev = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=dev)
    hyd.eval()

    cache = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    peps = list(dict.fromkeys(cache["pep"].tolist()))
    print(f"[rescue] {len(peps)} peptides, {len(cache)} cached candidates")

    dfs = []
    for i, pep in enumerate(peps):
        sub = cache[cache.pep == pep]
        z, ts, muts, G = setup(hyd, pep)
        if len(muts) < 2:
            continue
        positions = sorted(muts.keys())
        pos_set = set(positions)
        parent_aa = [ALPHABET.index(pep.ljust(MAXLEN)[p]) for p in positions]
        seqs = sub["seq"].astype(str).tolist()
        keep, rows = [], []
        for s in seqs:
            diff = {q for q in range(min(len(s), len(pep))) if s[q] != pep[q]}
            ok = diff <= pos_set
            keep.append(ok)
            if ok:
                rows.append([ALPHABET.index(s[p]) for p in positions])
        keep = np.asarray(keep)
        if keep.sum() < 30:
            continue
        cand = np.asarray(rows)
        Ew, Ep, Ea = rescue_energies(hyd, pep, z, ts, positions, parent_aa, cand, muts)
        g = sub.loc[keep, ["pep", "n_mut", "dist_geo", "dist_eucl", "dist_maha",
                           "score_mutangplus", "score_A_onehot", "score_B_onehot"]].copy()
        g["rescue_white"] = Ew
        g["rescue_pull"] = Ep
        g["rescue_add"] = Ea
        dfs.append(g)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(peps)}")
    big = pd.concat(dfs, ignore_index=True)
    big.to_parquet(CACHE / "_thesis_rq5_rescue_scores.parquet")

    # method -> (column, orientation): +1 potential (higher=closer), -1 distance/energy (lower=closer)
    METHODS = {
        "TANDEM-A": ("score_A_onehot", +1), "TANDEM-B": ("score_B_onehot", +1),
        "MUTANG+": ("score_mutangplus", +1),
        "d_G (pullback)": ("dist_maha", -1), "d_eucl (encoded)": ("dist_eucl", -1),
        "RESCUE-pullback": ("rescue_pull", -1), "RESCUE-whitened": ("rescue_white", -1),
        "RESCUE-additive": ("rescue_add", -1),
    }

    def align(col, sign):
        out = {nm: [] for nm in NMUTS}
        for (_p, nm), gg in big.groupby(["pep", "n_mut"]):
            if nm not in NMUTS or len(gg) < MIN_GROUP:
                continue
            s = gg[col].values
            dg = gg.dist_geo.values
            if np.std(s) == 0 or np.std(dg) == 0:
                continue
            # want correlation of "feasibility" with -d_geo; feasibility = sign*(-? ) ...
            # potentials (sign+1): high=closer -> corr(score, -dg). distances (sign-1): low=closer -> corr(-col,-dg)=corr(col,dg)
            r = spearmanr(s, -dg).correlation if sign > 0 else spearmanr(s, dg).correlation
            out[nm].append(r)
        return {nm: float(np.median(v)) if v else np.nan for nm, v in out.items()}

    def sel_ratio(col, sign, p=0.5):
        out = []
        for (_p, nm), gg in big.groupby(["pep", "n_mut"]):
            if nm not in NMUTS or len(gg) < MIN_GROUP:
                continue
            s = gg[col].values * (1 if sign > 0 else -1)   # higher = closer
            dg = gg.dist_geo.values
            thr = np.quantile(s, 1 - p)
            sel = s >= thr
            if sel.sum() < 5 or dg.mean() <= 0:
                continue
            out.append(dg[sel].mean() / dg.mean())
        return float(np.median(out)) if out else np.nan

    print(f"\n[tab:rescue] median per-peptide Spearman(feasibility, -d_geo) within n_mut, "
          f"and top-50% selection ratio")
    print(f"  {'method':18s} | {'nmut2':>7} {'nmut3':>7} {'nmut4':>7} | {'sel50':>6}")
    rows = []
    for name, (col, sign) in METHODS.items():
        a = align(col, sign)
        sr = sel_ratio(col, sign)
        rows.append({"method": name, **{f"nmut{k}": a[k] for k in NMUTS}, "sel50": sr})
        print(f"  {name:18s} | {a[2]:+7.3f} {a[3]:+7.3f} {a[4]:+7.3f} | {sr:6.3f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(CACHE / "_thesis_rq5_rescue.csv", index=False)

    # ---------------- figure ----------------
    order = list(METHODS)
    colors = ["#bdbdbd", "#bdbdbd", "#6baed6", "#9e9ac8", "#74c476", "#fdd0a2", "#fdae6b", "#e6550d"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # (a) alignment, averaged over n_mut
    ax = axes[0]
    vals = [np.nanmean([r[f"nmut{k}"] for k in NMUTS]) for r in rows]
    ax.barh(np.arange(len(order)), vals, color=colors, edgecolor="white")
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel(r"median Spearman(feasibility, $-d_{\mathrm{geo}}$), mean over $n_{\mathrm{mut}}\in\{2,3,4\}$")
    ax.set_title("(a) alignment with geodesic feasibility", fontweight="bold", fontsize=11)
    for y, v in enumerate(vals):
        ax.text(v + (0.005 if v >= 0 else -0.005), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    # (b) selection ratio
    ax = axes[1]
    srv = [r["sel50"] for r in rows]
    ax.barh(np.arange(len(order)), srv, color=colors, edgecolor="white")
    ax.axvline(1.0, color="#d7191c", ls="--", lw=1.0)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlim(0.6, 1.05)
    ax.set_xlabel(r"top-50%-by-feasibility $d_{\mathrm{geo}}$ ratio (selected/full)")
    ax.set_title("(b) does selection bring candidates closer?", fontweight="bold", fontsize=11)
    for y, v in enumerate(srv):
        ax.text(v - 0.005, y, f"{v:.2f}", va="center", ha="right", fontsize=8)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("RESCUE vs.\\ existing potentials on geodesic feasibility", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "rq5_rescue.pdf")


if __name__ == "__main__":
    main()
