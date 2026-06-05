"""RQ7 -- can a TANDEM-style pairwise score be made to align with geodesic feasibility?

The TANDEM potential aggregates, over pairs (i,j) of single-position mutations in a candidate,
a transform of the direction similarity x_ij = CS(v_i, v_j):
  * both taken      -> T_tt(x)   (reward aligned, penalise OPPOSED pair taken together),
  * exactly one     -> T_tnt(x)  (penalise taking only one of a CONSISTENT/parallel pair).
RQ5 showed the default (log transform, mean aggregation, similarity A/B) does not align with the
geodesic distance d_geo. Within a fixed Hamming distance, any LINEAR aggregation (sum or mean) over
pairs gives identical per-peptide rankings, so the only levers are the per-pair transform, the
similarity, and a NON-LINEAR aggregation. We sweep:
  similarity  : A (whitened latent cosine) | B (pullback-metric cosine)
  transform   : log | linear | hinge | quad | count   (all encode the two desired penalties)
  aggregation : mean (dilutes) | min  (worst single pair -> the "enormous penalty")
and measure the per-peptide, Hamming-controlled Spearman(score, -d_geo) for n_mut in {2,3,4}.
A transform-shape figure shows where each transform places its penalty.

Env: RQ7_LIMIT (cap sample peptides), RQ7_BENCH_ONLY=1 (skip parquet), THESIS_FIG_DIR.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, save  # noqa: E402
from _geodesic import geodesic_distance_to_parent  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fig_rq56_potentials import (  # noqa: E402  (reuse the exact RQ5 setup + candidate logic)
    HydrAMPEncoderDecoder, setup, stratified_keys, mahalanobis, ALPHABET, MAXLEN, GEO_K,
)
from pep_compass.local_enumeration.mutation.mutation_potentials import (  # noqa: E402
    ProjectedDirectionPairwiseSimilarityPotential,
    AmbientMetricPairwiseSimilarityPotential,
)

N_SAMPLE_PEPS = 400
rng = np.random.default_rng(0)
BENCH = {
    "middle-1": "FLYKWWIRIGRLKL", "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN", "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF", "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}


# ---- per-pair transforms (tt = both taken, tnt = exactly one taken); all <= 0, higher=better ----
def _clog(a):
    return np.log(np.clip(a, 1e-12, 1.0))

TRANSFORMS = {
    # name : (T_tt(x), T_tnt(x))
    "log":    (lambda x: _clog((1 + x) / 2), lambda x: _clog((1 - x) / 2)),
    "linear": (lambda x: (x - 1) / 2,        lambda x: -(x + 1) / 2),
    "hinge":  (lambda x: np.minimum(0.0, x), lambda x: np.minimum(0.0, -x)),   # penalise only wrong sign
    "quad":   (lambda x: -np.maximum(0.0, -x) ** 2, lambda x: -np.maximum(0.0, x) ** 2),  # steeper
    "count":  (lambda x: -(x < 0).astype(float),     lambda x: -(x > 0).astype(float)),    # # violations
}
AGGS = ["mean", "min"]
SIMS = ["A", "B"]

# Hard feasibility filter (no score, just accept/reject): reject a candidate if ANY
# one-taken pair has cosine > tau_high (consistent pair left half-done) OR any both-taken
# pair has cosine < tau_low (opposing pair taken together).  name -> (tau_low, tau_high).
REJECTS = {"reject.3": (-0.3, 0.3), "reject.5": (-0.5, 0.5)}


def battery_scores(positions, parent_aa, cand_aa, dirs):
    """cand_aa: (N, P) chosen aa-idx per position. dirs[sim][p]: dict aa_idx -> unit vector (np).
    Returns dict 'sim/transform/agg' -> (N,) score, plus n_mut (N,)."""
    N, P = cand_aa.shape
    mut = cand_aa != np.asarray(parent_aa)[None, :]          # (N,P) taken mask
    n_mut = mut.sum(1)
    iu, ju = np.triu_indices(P, k=1)                          # (n_pairs,)
    both = mut[:, iu] & mut[:, ju]                            # (N, n_pairs)
    one = (mut[:, iu] ^ mut[:, ju])
    involved = both | one
    out = {}
    for sim in SIMS:
        # chosen direction vectors per position -> (N, P, dim)
        V = np.stack([np.stack([dirs[sim][p][int(a)] for a in cand_aa[:, k]])
                      for k, p in enumerate(positions)], axis=1)   # (N, P, dim)
        cos = np.einsum("npd,nqd->npq", V, V)                 # (N, P, P)
        x = cos[:, iu, ju]                                    # (N, n_pairs)
        for tname, (ftt, ftnt) in TRANSFORMS.items():
            contrib = np.where(both, ftt(x), np.where(one, ftnt(x), 0.0))  # (N, n_pairs)
            masked = np.where(involved, contrib, np.nan)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(masked, axis=1)
                mn = np.nanmin(masked, axis=1)
            out[f"{sim}/{tname}/mean"] = np.nan_to_num(mean, nan=0.0)
            out[f"{sim}/{tname}/min"] = np.nan_to_num(mn, nan=0.0)
        # hard feasibility filter: 1 = accepted (no violating pair), 0 = rejected
        for rname, (tlow, thigh) in REJECTS.items():
            violate = (one & (x > thigh)).any(1) | (both & (x < tlow)).any(1)
            out[f"{sim}/{rname}"] = (~violate).astype(float)
    return out, n_mut


def eval_peptide(hyd, pep, n_sample):
    z, ts, muts, G = setup(hyd, pep)
    if len(muts) < 2:
        return None
    positions = sorted(muts.keys())
    padded = pep.ljust(MAXLEN)
    parent_aa = [ALPHABET.index(padded[p]) for p in positions]

    potA = ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode="onehot")
    potB = AmbientMetricPairwiseSimilarityPotential(ts, direction_mode="onehot")
    dirs = {"A": {}, "B": {}}
    for k, p in enumerate(positions):
        augmented = sorted(set(muts[p]) | {parent_aa[k]})
        aa_t = torch.tensor(augmented, dtype=torch.long)
        vA = potA._position_vectors(p, aa_t, parent_aa[k]).detach().cpu().numpy()
        vB = potB._position_vectors(p, aa_t, parent_aa[k]).detach().cpu().numpy()
        dirs["A"][p] = {int(a): vA[i] for i, a in enumerate(augmented)}
        dirs["B"][p] = {int(a): vB[i] for i, a in enumerate(augmented)}

    # enumerate Cartesian product of {muts[p] + parent}, drop all-parent, stratify
    per_pos = [sorted(set(muts[p]) | {parent_aa[k]}) for k, p in enumerate(positions)]
    grids = np.meshgrid(*per_pos, indexing="ij")
    cand = np.stack([g.ravel() for g in grids], axis=1)       # (prod, P)
    nmut_all = (cand != np.asarray(parent_aa)[None, :]).sum(1)
    cand = cand[nmut_all >= 1]
    if len(cand) == 0:
        return None
    keys = [tuple(int(a) for a in row) for row in cand]
    nmut_of = {k: int((np.asarray(k) != np.asarray(parent_aa)).sum()) for k in keys}
    sel = set(stratified_keys(keys, nmut_of, n_sample))
    cand = np.array([k for k in keys if k in sel])

    scores, n_mut = battery_scores(positions, parent_aa, cand, dirs)

    # build sequences, re-encode, distances
    seqs = []
    for row in cand:
        s = list(padded)
        for k, p in enumerate(positions):
            s[p] = ALPHABET[int(row[k])]
        seqs.append("".join(s[:len(pep)]))
    with torch.no_grad():
        zz_t = hyd.encode_peptides(seqs).detach()
    zz = zz_t.cpu().numpy()
    z_np = z.detach().cpu().numpy()[0]

    df = pd.DataFrame({"pep": pep, "n_mut": n_mut, **scores})
    df["dist_geo"] = geodesic_distance_to_parent(hyd, z, zz_t, k=GEO_K)
    df["dist_maha"] = mahalanobis(zz, z_np, G)
    return df


def main():
    # HydrAMP runs on CPU (multithreaded), matching the RQ5 pipeline that produced the
    # d_geo values the tangent space / setup() are built for; avoids cpu/cuda mixing.
    dev = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=dev)
    hyd.eval()
    print(f"HydrAMP loaded on {dev}")

    peps = [("bench", p) for p in BENCH.values()]
    if os.environ.get("RQ7_BENCH_ONLY", "0") != "1":
        pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")["sequence"]
        pool = pool.dropna().astype(str)
        pool = pool[(pool.str.len() <= MAXLEN) & (pool.str.len() > 3)
                    & pool.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates()
        n = int(os.environ.get("RQ7_LIMIT", N_SAMPLE_PEPS))
        peps += [("sample", p) for p in pool.sample(n=min(n, len(pool)), random_state=0)]

    dfs = []
    for tag, pep in peps:
        df = eval_peptide(hyd, pep, 3000 if tag == "bench" else 500)
        if df is not None and len(df) >= 30:
            dfs.append(df)
    print(f"[rq7] {len(dfs)} peptides")

    score_cols = ([f"{s}/{t}/{a}" for s in SIMS for t in TRANSFORMS for a in AGGS]
                  + [f"{s}/{r}" for s in SIMS for r in REJECTS])

    # acceptance rate of the hard filters (per mutation count), for interpretability
    allc = pd.concat(dfs, ignore_index=True)
    print("\n[reject] acceptance rate (fraction kept) by n_mut:")
    for col in [f"{s}/{r}" for s in SIMS for r in REJECTS]:
        rates = {nm: float(allc.loc[allc.n_mut == nm, col].mean()) for nm in (2, 3, 4)}
        print(f"  {col:14s} nmut2={rates[2]:.2f} nmut3={rates[3]:.2f} nmut4={rates[4]:.2f}")

    # Hamming-controlled median Spearman(score, -d_geo) per n_mut
    def med(col, nm):
        v = []
        for df in dfs:
            s = df[df.n_mut == nm]
            if len(s) >= 8 and s[col].std() > 0 and s.dist_geo.std() > 0:
                v.append(spearmanr(s[col], -s.dist_geo).correlation)
        return float(np.median(v)) if v else np.nan

    print(f"\n[tab:rq7] median per-peptide Spearman(score, -d_geo), n={len(dfs)} peptides")
    print(f"  {'variant':22s} | {'nmut2':>7} {'nmut3':>7} {'nmut4':>7}")
    rows = []
    for col in score_cols:
        r = {nm: med(col, nm) for nm in (2, 3, 4)}
        rows.append({"variant": col, **{f"nmut{k}": v for k, v in r.items()}})
        print(f"  {col:22s} | {r[2]:+7.3f} {r[3]:+7.3f} {r[4]:+7.3f}")
    tab = pd.DataFrame(rows)
    tab.to_csv(CACHE / "_thesis_rq7_scorings.csv", index=False)
    pd.concat(dfs, keys=range(len(dfs))).to_parquet(CACHE / "_thesis_rq7_scores.parquet")

    # ---- Fig 1: transform shapes (where the penalty lives) ----
    xs = np.linspace(-1, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for tname, (ftt, ftnt) in TRANSFORMS.items():
        axes[0].plot(xs, ftt(xs), label=tname)
        axes[1].plot(xs, ftnt(xs), label=tname)
    axes[0].set_title("both taken: $T_{tt}(x)$  (penalise opposed, $x\\to-1$)", fontweight="bold")
    axes[1].set_title("one taken: $T_{tnt}(x)$  (penalise consistent, $x\\to+1$)", fontweight="bold")
    for ax in axes:
        ax.axhline(0, color="#999", lw=0.6); ax.set_xlabel("pair similarity $x$")
        ax.set_ylim(-4, 0.3); ax.legend(fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("per-pair contribution")
    fig.suptitle("pairwise penalty transforms", fontweight="bold")
    fig.tight_layout(); save(fig, "rq5_scorefamily_transforms.pdf")

    # ---- Fig 2: alignment heatmap (variant x n_mut) ----
    H = tab.set_index("variant")[["nmut2", "nmut3", "nmut4"]]
    fig, ax = plt.subplots(figsize=(6.5, 9))
    sns.heatmap(H, annot=True, fmt="+.2f", center=0, cmap="vlag", vmin=-0.4, vmax=0.4,
                cbar_kws={"label": r"median Spearman(score, $-d_{\mathrm{geo}}$)"}, ax=ax)
    ax.set_title(f"scoring alignment with geodesic feasibility (n={len(dfs)})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("within mutation count"); ax.set_ylabel("similarity / transform / aggregation")
    fig.tight_layout(); save(fig, "rq5_scorefamily_alignment.pdf")


if __name__ == "__main__":
    main()
