"""RQ5 & RQ6 -- the TANDEM mutation-similarity potential.

Potentials over the MUTANG candidate set:
  * MUTANG+                 -- decoder log-prob baseline (metric-free; sum over mutated positions).
  * TANDEM-A/B {onehot,diff} -- pairwise-similarity potential, Euclidean-latent (A) or pullback
                                metric G=J^T J (B); direction = e_{l,a} (onehot) or
                                e_{l,a}-e_{l,parent} (diff). Transform log((1+-x)/2).
  * TANDEM-A/B onehot (lin)  -- same but with the bounded linear transform
                                T_tt(x)=(x-1)/2, T_tnt(x)=-(x+1)/2.

Distance to parent = pullback (Mahalanobis) d_G(z,z')=||J(z)(z'-z)||, G=metric_from_jac (once
per peptide); Euclidean kept as a baseline. feasibility = -d_G.

IMPORTANT: d_G and the *sum*-type MUTANG+ both scale with the number of mutations (Hamming
distance), so raw comparisons are confounded. We therefore report (i) the requested top-p%
selection figure (raw) and (ii) a Hamming-controlled view (per-mutation-count Spearman).

Outputs (vector PDF) + CSVs.
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
from Bio.Align import substitution_matrices

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, ALL_AA, save, SOFT_DIVERGING

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder
from pep_compass.models.encoder_decoder.utils import decoder_jacobian
from pep_compass.local_enumeration.mutation.utils import get_mutations_from_s_u_standard
from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace
from pep_compass.local_enumeration.mutation.mutation_potentials import (
    ProjectedDirectionPairwiseSimilarityPotential,
    AmbientMetricPairwiseSimilarityPotential,
    linear_taken_taken_transform,
    linear_taken_not_taken_transform,
)
from pep_compass.geometry.utils import metric_from_jac
from _geodesic import geodesic_distance_to_parent

# Which distance feeds the RQ5 feasibility analysis: graph geodesic (default),
# first-order pullback Mahalanobis, or Euclidean (both kept as baselines).
_RQ5_FEAS = os.environ.get("RQ5_FEAS", "geo")
_DIST = {"geo": "dist_geo", "maha": "dist_maha", "eucl": "dist_eucl"}
_DLAB = {"geo": "geodesic distance", "maha": "pullback distance", "eucl": "Euclidean distance"}
_DSYM = {"geo": r"-d_{\mathrm{geo}}", "maha": r"-d_G", "eucl": r"-d_{\mathrm{euc}}"}
DIST_COL = _DIST[_RQ5_FEAS]
DIST_LABEL = _DLAB[_RQ5_FEAS]
DIST_SYM = _DSYM[_RQ5_FEAS]
GEO_K = int(os.environ.get("RQ5_GEO_K", "12"))

BENCHMARK_PEPTIDES = {
    "middle-1": "FLYKWWIRIGRLKL", "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN", "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF", "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}
ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
MAXLEN = 25
KAPPA, THETA_MUT, MIN_DIR = 1e-3, 0.1, 5
PRODUCT_CAP = 300_000
N_SAMPLE_PEPS = 500
N_BLOSUM_PEPS = 300
P_GRID = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
SEL_LEVELS = [25, 50, 75, 90, 100]
SEL9 = [95, 80, 70, 50]                      # selection levels for the 9-series figure
rng = np.random.default_rng(0)

_lin = dict(similarity_transform=linear_taken_taken_transform,
            taken_not_taken_transform=linear_taken_not_taken_transform)
POTS = {
    "A_onehot":     lambda ts: ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode="onehot"),
    "A_diff":       lambda ts: ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode="diff"),
    "B_onehot":     lambda ts: AmbientMetricPairwiseSimilarityPotential(ts, direction_mode="onehot"),
    "B_diff":       lambda ts: AmbientMetricPairwiseSimilarityPotential(ts, direction_mode="diff"),
    "A_onehot_lin": lambda ts: ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode="onehot", **_lin),
    "B_onehot_lin": lambda ts: AmbientMetricPairwiseSimilarityPotential(ts, direction_mode="onehot", **_lin),
}
PLABEL = {"mutangplus": "MUTANG+", "A_onehot": "A onehot", "A_diff": "A diff",
          "B_onehot": "B onehot", "B_diff": "B diff",
          "A_onehot_lin": "A onehot (lin)", "B_onehot_lin": "B onehot (lin)"}


def cartesian_size(muts):
    prod = 1
    for m in muts.values():
        prod *= (len(m) + 1)
    return prod


def bound_mutations(muts, cap):
    muts = dict(muts)
    while len(muts) > 1 and cartesian_size(muts) > cap:
        del muts[max(muts, key=lambda k: len(muts[k]))]
    return muts


def setup(hydramp, pep):
    z = hydramp.encode_peptides([pep])
    jac = decoder_jacobian(
        lambda x: hydramp.decoder_forward(x, softmax=True, flatten=True),
        z, "approx", {"jacobian_eps": 1e-6})
    U, S, V = torch.linalg.svd(jac, full_matrices=False)
    S0 = S[0]; n_sv = S0.shape[0]
    K = min(max(int((S0 > KAPPA).sum().item()), MIN_DIR), n_sv)
    hi = float(S0[K - 1].item()); lo = float(S0[K].item()) if K < n_sv else 0.0
    ts = SubRiemannianTangentSpace(U=U[0], S=S[0], V=V[0],
                                   horizontal_threshold=0.5 * (hi + lo), device="cpu")
    G = metric_from_jac(jac)[0].detach().cpu().numpy()
    muts = get_mutations_from_s_u_standard(
        s=S0.detach().cpu().numpy(), u=U[0].detach().cpu().numpy(),
        max_len=MAXLEN, alphabet_size=len(ALPHABET),
        direction_significance_threshold=KAPPA, min_number_of_directions=MIN_DIR,
        token_threshold=THETA_MUT)
    muts = {p: m for p, m in muts.items() if p < len(pep) and len(m) > 0}
    return z, ts, bound_mutations(muts, PRODUCT_CAP), G


def mahalanobis(zz, z0, G):
    d = zz - z0[None, :]
    return np.sqrt(np.clip(np.einsum("ni,ij,nj->n", d, G, d), 0.0, None))


def parent_logprob_table(hydramp, z):
    return hydramp.decoder_forward(z, softmax=False, log_softmax=True,
                                   flatten=False)[0].detach().cpu().numpy()


def stratified_keys(keys, nmut_of, n_sample):
    """Keep all single mutants, then doubles, then higher-order, up to n_sample."""
    by_h = defaultdict(list)
    for k in keys:
        by_h[nmut_of[k]].append(k)
    singles = by_h.get(1, [])
    doubles = by_h.get(2, []); rng.shuffle(doubles)
    higher = [k for h, ks in by_h.items() if h >= 3 for k in ks]; rng.shuffle(higher)
    sel = list(singles)
    sel += doubles[:max(0, int((n_sample - len(sel)) * 0.6))]
    sel += higher[:max(0, n_sample - len(sel))]
    return sel if sel else keys[:n_sample]


def evaluate_peptide(hydramp, pep, n_sample):
    z, ts, muts, G = setup(hydramp, pep)
    if len(muts) < 2:
        return None
    z_np = z.detach().cpu().numpy()[0]
    lp = parent_logprob_table(hydramp, z)
    scores = {name: POTS[name](ts).compute_with_identities(pep, muts) for name in POTS}
    keys = list(scores["A_onehot"].keys())
    if not keys:
        return None
    positions = sorted(muts.keys())
    padded = pep.ljust(MAXLEN)
    parent_aa = [ALPHABET.index(padded[p]) for p in positions]
    nmut_of = {k: sum(1 for i in range(len(positions)) if k[i] != parent_aa[i]) for k in keys}
    keys = stratified_keys(keys, nmut_of, n_sample)

    rows, seqs = [], []
    for key in keys:
        seq_arr = list(padded); lp_mut = 0.0
        for i, p in enumerate(positions):
            seq_arr[p] = ALPHABET[key[i]]
            if key[i] != parent_aa[i]:
                lp_mut += float(lp[p, key[i]])
        seqs.append("".join(seq_arr[:len(pep)]))
        row = {"pep": pep, "n_mut": nmut_of[key], "seq": seqs[-1], "score_mutangplus": lp_mut}
        for name in POTS:
            row[f"score_{name}"] = scores[name][key]
        rows.append(row)
    df = pd.DataFrame(rows)
    with torch.no_grad():
        zz_t = hydramp.encode_peptides(seqs).detach()      # (N, Z) on device
    zz = zz_t.cpu().numpy()
    df["dist_eucl"] = np.linalg.norm(zz - z_np[None, :], axis=1)
    df["dist_maha"] = mahalanobis(zz, z_np, G)             # first-order pullback (baseline)
    df["dist_geo"] = geodesic_distance_to_parent(hydramp, z, zz_t, k=GEO_K)
    df["feas"] = -df[DIST_COL]
    return df


def cross_position_similarity(pot, pep, muts):
    padded = pep.ljust(MAXLEN)
    vecs, labels, posarr = [], [], []
    for p in sorted(muts.keys()):
        parent_aa = ALPHABET.index(padded[p])
        v = pot._position_vectors(p, torch.tensor(list(muts[p]), dtype=torch.long), parent_aa)
        vecs.append(v.detach())
        labels += [f"{p}:{ALPHABET[a]}" for a in muts[p]]
        posarr += [p] * len(muts[p])
    V = torch.cat(vecs, dim=0).cpu().numpy()
    posarr = np.array(posarr)
    return V @ V.T, (posarr[:, None] == posarr[None, :]), labels


# ----------------------------- AMP-BLOSUM ------------------------------------
def discrete_blosum(pair_counts):
    F = pair_counts + pair_counts.T
    np.fill_diagonal(F, np.diag(pair_counts))
    T = F.sum()
    if T == 0:
        return np.full((20, 20), np.nan)
    q = F / T
    p = np.diag(q) + 0.5 * (q.sum(1) - np.diag(q))
    E = np.outer(p, p); iu = ~np.eye(20, dtype=bool); E[iu] *= 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 2.0 * np.log2(q / E)
    s[~np.isfinite(s)] = np.nan
    return np.round(s)


def build_amp_blosum(hydramp, peps):
    idx = {a: i for i, a in enumerate(ALL_AA)}
    counts = {p: np.zeros((20, 20)) for p in SEL_LEVELS}
    n_used = 0
    for pep in peps:
        z, ts, muts, _ = setup(hydramp, pep)
        if len(muts) < 2:
            continue
        sc = AmbientMetricPairwiseSimilarityPotential(
            tangent_space=ts, direction_mode="onehot").compute_with_identities(pep, muts)
        if not sc:
            continue
        positions = sorted(muts.keys()); padded = pep.ljust(MAXLEN)
        items = sorted(sc.items(), key=lambda kv: -kv[1])
        seqs = ["".join([(ALPHABET[k[positions.index(p)]] if p in positions else padded[p])
                         for p in range(len(pep))]) for k, _ in items]
        n = len(seqs)
        for lvl in SEL_LEVELS:
            for s in seqs[:max(1, int(round(lvl / 100 * n)))]:
                for a, b in zip(pep, s):
                    if a in idx and b in idx:
                        counts[lvl][idx[a], idx[b]] += 1
        n_used += 1
    return {lvl: discrete_blosum(counts[lvl]) for lvl in SEL_LEVELS}, n_used


def blosum62_matrix():
    bl = substitution_matrices.load("BLOSUM62")
    return np.array([[float(bl[a, b]) for b in ALL_AA] for a in ALL_AA])


def offdiag_spearman(A, B):
    iu = ~np.eye(20, dtype=bool)
    a, b = A[iu], B[iu]
    m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).correlation if m.sum() >= 10 else np.nan


def main():
    hydramp = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                    field_eps=1e-6, device=torch.device("cpu"))
    hydramp.eval()
    print(f"HydrAMP loaded | latent {hydramp.latent_dim} | ambient {hydramp.ambient_dim}")

    pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")["sequence"]
    pool = pool.dropna().astype(str)
    pool = pool[(pool.str.len() <= MAXLEN) & (pool.str.len() > 3)
                & pool.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates()
    n_sample = int(os.environ.get("RQ5_LIMIT", N_SAMPLE_PEPS))  # cap for gate-checks
    sample_peps = pool.sample(n=min(n_sample, len(pool)), random_state=0).tolist()
    bench = list(BENCHMARK_PEPTIDES.values())

    # ---- Fig 1+2: similarity heatmaps (4 variants) + transform curves ----
    demo = "FLYKWWIRIGRLKL"
    z, ts, muts, _ = setup(hydramp, demo)
    four = ["A_onehot", "A_diff", "B_onehot", "B_diff"]
    sims = {v: cross_position_similarity(POTS[v](ts), demo, muts) for v in four}
    titles = {"A_onehot": "TANDEM-A, one-hot", "A_diff": "TANDEM-A, diff",
              "B_onehot": "TANDEM-B, one-hot", "B_diff": "TANDEM-B, diff"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    for ax, v in zip(axes.ravel(), four):
        sim, _, labels = sims[v]; m = sim.copy(); np.fill_diagonal(m, np.nan)
        sns.heatmap(m, ax=ax, cmap=SOFT_DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                    cbar_kws={"shrink": 0.6, "label": "cosine similarity"},
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(titles[v], fontsize=12, fontweight="bold"); ax.tick_params(labelsize=5)
    fig.suptitle(f"Pairwise single-position mutation similarity, four variants ({demo})",
                 fontsize=14, fontweight="bold"); fig.tight_layout()
    save(fig, "rq5_similarity_variants.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.linspace(-1, 1, 400)
    axes[0].plot(xs, np.log(np.clip((1 + xs) / 2, 1e-12, 1)), label=r"log: taken-taken $\log\frac{1+x}{2}$")
    axes[0].plot(xs, np.log(np.clip((1 - xs) / 2, 1e-12, 1)), label=r"log: taken-not-taken $\log\frac{1-x}{2}$")
    axes[0].plot(xs, (xs - 1) / 2, "--", label=r"linear: taken-taken $\frac{x-1}{2}$")
    axes[0].plot(xs, -(xs + 1) / 2, "--", label=r"linear: taken-not-taken $-\frac{x+1}{2}$")
    axes[0].axhline(0, color="#999", lw=0.6); axes[0].set_ylim(-4, 0.3)
    axes[0].set_xlabel("pairwise similarity x"); axes[0].set_ylabel("transform T(x)")
    axes[0].set_title("Pair transforms (log vs new bounded linear)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=7)
    for v in four:
        sim, samepos, _ = sims[v]
        off = (~samepos) & np.triu(np.ones_like(samepos), 1).astype(bool)
        axes[1].hist(sim[off], bins=40, alpha=0.5, density=True, label=titles[v])
    axes[1].axvline(0, color="#999", lw=0.6)
    axes[1].set_xlabel("cross-position pairwise similarity"); axes[1].set_ylabel("density")
    axes[1].set_title("Similarity distribution by variant", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=7)
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout(); save(fig, "rq5_transforms.pdf")

    # ---- aggregate ----
    dfs, tags = [], []
    for tag, pep in [("bench", p) for p in bench] + [("sample", p) for p in sample_peps]:
        df = evaluate_peptide(hydramp, pep, 3000 if tag == "bench" else 500)
        if df is None or df[DIST_COL].std() == 0 or len(df) < 30:
            continue
        dfs.append(df); tags.append(tag)
    print(f"[aggregate] {len(dfs)} peptides ({tags.count('bench')} bench, {tags.count('sample')} sample)")
    print(f"[feas] using {DIST_COL} for the RQ5 feasibility analysis (geodesic k={GEO_K})")

    # ---- gate check: is the geodesic materially different from the baselines? ----
    def _med_spear(a, b):
        vals = [spearmanr(df[a], df[b]).correlation for df in dfs
                if df[a].std() > 0 and df[b].std() > 0]
        return float(np.median(vals)) if vals else float("nan")
    print("[gate] median per-peptide Spearman between distances:")
    print(f"   geo vs maha  = {_med_spear('dist_geo', 'dist_maha'):+.3f}")
    print(f"   geo vs eucl  = {_med_spear('dist_geo', 'dist_eucl'):+.3f}")
    print(f"   maha vs eucl = {_med_spear('dist_maha', 'dist_eucl'):+.3f}")
    for c in ("dist_geo", "dist_maha", "dist_eucl"):
        allv = np.concatenate([df[c].values for df in dfs])
        sing = np.concatenate([df.loc[df.n_mut == 1, c].values for df in dfs])
        print(f"   {c:9s} all: median={np.median(allv):.4g} mean={allv.mean():.4g}"
              f"  | single-mut median={np.median(sing):.4g}")

    # ---- Fig 3: 9-series selection figure (raw) ----
    nine_vars = ["A_onehot", "B_onehot"]
    agg = {"MUTANG": 1.0}
    for v in nine_vars:
        for p in SEL9:
            ratios = []
            for df in dfs:
                k = max(1, int(round(p / 100 * len(df))))
                full = df[DIST_COL].mean()
                if full > 0:
                    ratios.append(df.nlargest(k, f"score_{v}")[DIST_COL].mean() / full)
            agg[f"{v}@{p}"] = float(np.mean(ratios))
    order = ["MUTANG"] + [f"{v}@{p}" for v in nine_vars for p in SEL9]
    colors = ["#9aa0a6"] + ["C0"] * len(SEL9) + ["C1"] * len(SEL9)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(order)), [agg[o] for o in order], color=colors)
    ax.axhline(1.0, color="#c00", lw=0.9, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["MUTANG\n(all)"] + [o.replace("_onehot", "").replace("@", "\n@") + "%"
                                            for o in order[1:]], fontsize=8)
    ax.set_ylabel(f"mean {DIST_LABEL} to parent\n(relative to full candidate set)")
    ax.set_ylim(0.9, max(1.02, max(agg.values()) + 0.02))
    ax.set_title(f"RQ5: proximity of top-p% selected mutants vs MUTANG (n={len(dfs)} peptides)",
                 fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); save(fig, "rq5_feasibility.pdf")
    print("9-series mean distance ratio (vs MUTANG=1.0):")
    for o in order:
        print(f"  {o:14s} {agg[o]:.3f}")

    # ---- Fig 3b: Hamming-controlled feasibility (per-mutation-count Spearman) ----
    methods = ["mutangplus", "A_onehot", "B_onehot", "A_onehot_lin", "B_onehot_lin", "A_diff", "B_diff"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    summary = {}
    for ax, nm, lab in [(axes[0], 1, "single mutants"), (axes[1], 2, "double mutants")]:
        recs = []
        for m in methods:
            col = "score_mutangplus" if m == "mutangplus" else f"score_{m}"
            for df in dfs:
                s = df[df["n_mut"] == nm]
                if len(s) >= 8 and s[col].std() > 0 and s[DIST_COL].std() > 0:
                    recs.append({"method": PLABEL[m], "rho": spearmanr(s[col], -s[DIST_COL]).correlation})
        rdf = pd.DataFrame(recs)
        summary[nm] = rdf.groupby("method")["rho"].median()
        sns.boxplot(data=rdf, x="method", y="rho", ax=ax,
                    order=[PLABEL[m] for m in methods if PLABEL[m] in set(rdf["method"])], width=0.6)
        ax.axhline(0, color="#c00", lw=0.8, ls="--")
        ax.set_title(f"within {lab} (n_mut={nm})", fontsize=11, fontweight="bold")
        ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(rf"per-peptide Spearman$(\,$potential, ${DIST_SYM}\,)$")
    fig.suptitle("RQ5: Hamming-controlled feasibility alignment (confound removed)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "rq5_feasibility_hamming.pdf")
    print("\nHamming-controlled median Spearman (single / double):")
    for m in methods:
        l = PLABEL[m]
        print(f"  {l:16s} {summary[1].get(l, np.nan):+.3f} / {summary[2].get(l, np.nan):+.3f}")

    # ---- table: Hamming-controlled medians for ALL THREE distances (geo primary,
    #      maha/eucl baselines) -- gives tab:rq5 + the baseline comparison in one run ----
    def _ham_median(dist_col, m, nm):
        col = "score_mutangplus" if m == "mutangplus" else f"score_{m}"
        vals = []
        for df in dfs:
            s = df[df["n_mut"] == nm]
            if len(s) >= 8 and s[col].std() > 0 and s[dist_col].std() > 0:
                vals.append(spearmanr(s[col], -s[dist_col]).correlation)
        return float(np.median(vals)) if vals else float("nan")

    print(f"\n[tab:rq5] median per-peptide Spearman(potential, -distance), n={len(dfs)} peptides")
    print(f"  {'potential':16s} | {'geodesic s/d':>16s} | {'pullback s/d':>16s} | {'euclid s/d':>16s}")
    tab_rows = []
    for m in methods:
        cells = {}
        for dc in ("dist_geo", "dist_maha", "dist_eucl"):
            cells[dc] = (_ham_median(dc, m, 1), _ham_median(dc, m, 2))
        g, ma, eu = cells["dist_geo"], cells["dist_maha"], cells["dist_eucl"]
        print(f"  {PLABEL[m]:16s} | {g[0]:+.3f}/{g[1]:+.3f} | "
              f"{ma[0]:+.3f}/{ma[1]:+.3f} | {eu[0]:+.3f}/{eu[1]:+.3f}")
        tab_rows.append({"potential": PLABEL[m],
                         "geo_single": g[0], "geo_double": g[1],
                         "maha_single": ma[0], "maha_double": ma[1],
                         "eucl_single": eu[0], "eucl_double": eu[1]})
    pd.DataFrame(tab_rows).to_csv(CACHE / "_thesis_rq5_distance_table.csv", index=False)
    pd.concat(dfs, keys=range(len(dfs))).to_parquet(CACHE / "_thesis_rq5_distances.parquet")

    # ---- Fig 4: threshold-distance curves (many peptides; log vs linear transform) ----
    curve_vars = ["A_onehot", "B_onehot", "A_onehot_lin", "B_onehot_lin"]
    clab = {"A_onehot": "A (log)", "B_onehot": "B (log)", "A_onehot_lin": "A (linear)", "B_onehot_lin": "B (linear)"}

    def curves(group):
        out = {}
        for v in curve_vars:
            ys = []
            for p in P_GRID:
                rr = []
                for df in group:
                    k = max(1, int(round(p / 100 * len(df))))
                    full = df[DIST_COL].mean()
                    if full > 0:
                        rr.append(df.nlargest(k, f"score_{v}")[DIST_COL].mean() / full)
                ys.append(np.mean(rr))
            out[v] = ys
        return out
    bdfs = [d for d, t in zip(dfs, tags) if t == "bench"]
    sdfs = [d for d, t in zip(dfs, tags) if t == "sample"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, g, ttl in [(axes[0], bdfs, f"{len(bdfs)} benchmark peptides"),
                       (axes[1], sdfs, f"{len(sdfs)} sampled peptides")]:
        cur = curves(g)
        for v in curve_vars:
            ax.plot(P_GRID, cur[v], marker="o", label=clab[v])
        ax.axhline(1.0, color="#999", lw=0.8, ls=":")
        ax.set_xlabel("top-% of candidates kept (by potential)"); ax.invert_xaxis()
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(f"mean {DIST_LABEL} to parent\n(relative to full candidate set)")
    axes[1].legend(fontsize=8)
    fig.suptitle("RQ5/RQ6: does keeping high-potential candidates select more proximal mutants?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "rq5_threshold_distance.pdf")

    # ---- Fig 5: AMP-BLOSUM (TANDEM-B) vs BLOSUM62 ----
    amp_bl, n_used = build_amp_blosum(hydramp, pool.sample(n=min(N_BLOSUM_PEPS, len(pool)),
                                                           random_state=1).tolist())
    b62 = blosum62_matrix()
    corr = {lvl: offdiag_spearman(amp_bl[lvl], b62) for lvl in SEL_LEVELS}
    print(f"\n[RQ6 a] AMP-BLOSUM from TANDEM-B over {n_used} Veltri-positive peptides")
    for lvl in SEL_LEVELS:
        print(f"  top-{lvl:3d}%  Spearman vs BLOSUM62 = {corr[lvl]:+.3f}")
    order_aa = list("AVLIMFWPGSTYCNQKRHDE")
    perm = [ALL_AA.index(a) for a in order_aa]
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9])

    def show(ax, M, title, vmin, vmax, cmap):
        sns.heatmap(M[np.ix_(perm, perm)], ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
                    square=True, xticklabels=order_aa, yticklabels=order_aa, cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.tick_params(labelsize=6)
    show(fig.add_subplot(gs[0]), b62, "BLOSUM62", -4, 11, "viridis")
    show(fig.add_subplot(gs[1]), amp_bl[25], "AMP-BLOSUM (TANDEM-B, top-25%)", -6, 6, SOFT_DIVERGING)
    show(fig.add_subplot(gs[2]), amp_bl[100], "AMP-BLOSUM (top-100% = MUTANG)", -6, 6, SOFT_DIVERGING)
    axc = fig.add_subplot(gs[3])
    axc.plot(SEL_LEVELS, [corr[l] for l in SEL_LEVELS], marker="o", color="C1")
    axc.set_xlabel("TANDEM-B selection (top-%)"); axc.set_ylabel("Spearman vs BLOSUM62")
    axc.set_title("agreement vs selection", fontsize=11, fontweight="bold"); axc.invert_xaxis()
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    fig.suptitle("RQ6(a): discrete log-odds AMP-BLOSUM from TANDEM-B mutants vs BLOSUM62",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "rq6_blosum.pdf")
    pd.DataFrame({"level": SEL_LEVELS, "spearman_vs_blosum62": [corr[l] for l in SEL_LEVELS]}).to_csv(
        CACHE / "_thesis_rq6_blosum_corr.csv", index=False)


if __name__ == "__main__":
    main()
