"""MUTANG+ viability filter --- sweep over similarity type and selection rule.

MUTANG+ keeps, for each single-position mutation (the anchor), the cross-position partner mutations it
deems *viable* (cosine similarity to the anchor above a threshold tau) and drops the *improper* ones.
We compare FOUR options:
  similarity : A = whitened latent cosine (TANDEM-A)  |  B = pullback-metric cosine (TANDEM-B)
  selection  : argmax  = one best residue per viable position (one filtered candidate per anchor)
               product = the Cartesian product of all viable mutations (incl. identity per position)
Distances are the cached geodesics (`_thesis_rq5_distances.parquet`, seq -> dist_geo); NO geodesic is
recomputed.  Per tau we report the mean d_geo relative to the full MUTANG set (MUTANG=1), the mean
number of positions changed, and the cache coverage, for all four options.  We also report the
distribution of cross-position cosine similarities for A vs B (the B similarities are near-orthogonal).

CPU; the model is used only for the cheap per-peptide SVD/similarity, never for distances.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, save  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fig_rq56_potentials import HydrAMPEncoderDecoder, setup, ALPHABET, MAXLEN  # noqa: E402
from pep_compass.local_enumeration.mutation.mutation_potentials import (  # noqa: E402
    ProjectedDirectionPairwiseSimilarityPotential,
    AmbientMetricPairwiseSimilarityPotential,
)

THRESHOLDS = [round(t, 3) for t in np.arange(-0.9, 0.9001, 0.025)]   # dense grid
SIMS = ["A", "B"]
COL = {("A", "argmax"): "#e6550d", ("A", "product"): "#fdae6b",
       ("B", "argmax"): "#3182bd", ("B", "product"): "#9ecae1"}
STY = {"argmax": "-", "product": "--"}
ENC_COL = {("A", "onehot"): "#e6550d", ("A", "diff"): "#fd8d3c",
           ("B", "onehot"): "#3182bd", ("B", "diff"): "#6baed6"}


def sim_matrices(ts, muts, positions, parent_aa, mode="onehot"):
    """labels=[(pos,aa)], and the A (whitened) and B (pullback) cross-mutation cosine matrices
    under the given direction encoding (`onehot` = e_{l,a}, `diff` = e_{l,a}-e_{l,parent})."""
    potA = ProjectedDirectionPairwiseSimilarityPotential(ts, direction_mode=mode)
    potB = AmbientMetricPairwiseSimilarityPotential(ts, direction_mode=mode)
    labels, vA, vB = [], [], []
    for k, p in enumerate(positions):
        aa = list(muts[p])
        if not aa:
            continue
        t = torch.tensor(aa, dtype=torch.long)
        vA.append(potA._position_vectors(p, t, parent_aa[k]).detach().cpu().numpy())
        vB.append(potB._position_vectors(p, t, parent_aa[k]).detach().cpu().numpy())
        labels += [(p, int(a)) for a in aa]
    VA, VB = np.concatenate(vA), np.concatenate(vB)
    return labels, VA @ VA.T, VB @ VB.T


def cross_cosines(S, labels):
    pos = np.array([l for l, _ in labels])
    iu, ju = np.triu_indices(len(labels), 1)
    m = pos[iu] != pos[ju]
    return S[iu, ju][m]


def seq_of(pep, chosen):
    s = list(pep.ljust(MAXLEN))
    for p, a in chosen.items():
        s[p] = ALPHABET[a]
    return "".join(s[:len(pep)])


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=torch.device("cpu"))
    hyd.eval()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))

    cache = pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet").reset_index(drop=True)
    peps = list(dict.fromkeys(cache["pep"].tolist()))
    print(f"[mutangp] {len(peps)} cached peptides; thresholds {THRESHOLDS[0]}..{THRESHOLDS[-1]}")

    cos_pool = {(s, e): [] for s in SIMS for e in ("onehot", "diff")}
    rec_p = {s: [] for s in SIMS}    # argmax  (per anchor, per tau)
    rec_pm = {s: [] for s in SIMS}   # product (per cached candidate, per anchor)
    rec_pr = {s: [] for s in SIMS}   # product sizes / expected #changed (per anchor, per tau)

    for ip, pep in enumerate(peps):
        sub = cache[cache.pep == pep]
        seq2d = dict(zip(sub["seq"], sub["dist_geo"]))
        full_mean = float(sub["dist_geo"].mean())
        if not (full_mean > 0):
            continue
        z, ts, muts, G = setup(hyd, pep)
        if len(muts) < 2:
            continue
        positions = sorted(muts.keys())
        parent_aa = [ALPHABET.index(pep.ljust(MAXLEN)[p]) for p in positions]
        labels, SA, SB = sim_matrices(ts, muts, positions, parent_aa, "onehot")
        M = len(labels)
        if M < 3:
            continue
        Smats = {"A": SA, "B": SB}                            # filter uses the one-hot similarities
        cos_pool[("A", "onehot")].append(cross_cosines(SA, labels))
        cos_pool[("B", "onehot")].append(cross_cosines(SB, labels))
        _, SAd, SBd = sim_matrices(ts, muts, positions, parent_aa, "diff")   # diff: diagnostic only
        cos_pool[("A", "diff")].append(cross_cosines(SAd, labels))
        cos_pool[("B", "diff")].append(cross_cosines(SBd, labels))

        label_of = {la: k for k, la in enumerate(labels)}
        pos_items = defaultdict(list)
        for k, (l, a) in enumerate(labels):
            pos_items[l].append((k, a))
        cand_idx = []
        for seq, dg in seq2d.items():
            try:
                idxs = [label_of[(p, ALPHABET.index(seq[p]))] for p in range(len(pep)) if seq[p] != pep[p]]
            except KeyError:
                continue
            if idxs:
                cand_idx.append((idxs, dg))

        for s in SIMS:
            S = Smats[s]
            for idxs, dg in cand_idx:                         # product-membership records
                for i in idxs:
                    others = [j for j in idxs if j != i]
                    mins = float(min(S[i, j] for j in others)) if others else np.inf
                    rec_pm[s].append({"pep": pep, "nmut": len(idxs), "dgeo": dg,
                                      "minS": mins, "full_mean": full_mean})
            for i, (li, ai) in enumerate(labels):             # argmax + product sizes
                best = {}
                for l, items in pos_items.items():
                    if l == li:
                        continue
                    sims = [(float(S[i, k]), a) for k, a in items]
                    bs, ba = max(sims, key=lambda x: x[0])
                    best[l] = (bs, ba, [v for v, _ in sims])
                for tau in THRESHOLDS:
                    chosen = {li: ai}
                    for l in best:
                        if best[l][0] > tau:
                            chosen[l] = best[l][1]
                    seq = seq_of(pep, chosen)
                    dg = seq2d.get(seq, np.nan)
                    rec_p[s].append({"pep": pep, "tau": tau, "size": len(chosen),
                                     "dgeo": float(dg) if dg == dg else np.nan,
                                     "in_cache": seq in seq2d, "full_mean": full_mean})
                    prod, exp_changed = 1.0, 1.0
                    for l in best:
                        if best[l][0] > tau:
                            k = sum(1 for v in best[l][2] if v > tau)
                            prod *= (k + 1); exp_changed += k / (k + 1)
                    rec_pr[s].append({"pep": pep, "tau": tau, "prod": prod, "exp_changed": exp_changed})
        if (ip + 1) % 50 == 0:
            print(f"  ...{ip + 1}/{len(peps)}")

    # ---------------- cosine-distribution diagnostic (A/B x onehot/diff) ----------------
    print("\n[cosine diagnostic] cross-position cosine similarity (pooled):")
    print(f"  {'sim/enc':12s} {'mean':>7} {'mean|x|':>8} {'min':>7} {'max':>7} {'frac|x|<0.1':>12}")
    cstat = {}
    for s in SIMS:
        for e in ("onehot", "diff"):
            x = np.concatenate(cos_pool[(s, e)])
            cstat[(s, e)] = dict(mean=float(x.mean()), mean_abs=float(np.abs(x).mean()),
                                 mn=float(x.min()), mx=float(x.max()),
                                 lt=float((np.abs(x) < 0.1).mean()), vals=x)
            d = cstat[(s, e)]
            print(f"  {s+'/'+e:12s} {d['mean']:+7.3f} {d['mean_abs']:8.3f} {d['mn']:+7.3f}"
                  f" {d['mx']:+7.3f} {d['lt']:12.3f}")
    pd.DataFrame([{"sim": s, "enc": e, **{k: v for k, v in d.items() if k != 'vals'}}
                  for (s, e), d in cstat.items()]).to_csv(CACHE / "_thesis_simcluster_cosines.csv", index=False)

    # ---------------- per-(sim, method, tau) aggregation ----------------
    def _relmean(g):
        if not len(g):
            return np.nan
        r = g.groupby("pep").dgeo.mean() / g.groupby("pep").full_mean.first()
        return float(np.nanmean(r.replace([np.inf, -np.inf], np.nan)))

    rows = []
    for s in SIMS:
        dfp, dfpm, dfpr = pd.DataFrame(rec_p[s]), pd.DataFrame(rec_pm[s]), pd.DataFrame(rec_pr[s])
        for tau in THRESHOLDS:
            tp = dfp[dfp.tau == tau]; tpc = tp[tp.in_cache]
            tpm = dfpm[dfpm.minS > tau]
            pr = dfpr[dfpr.tau == tau]; prod_total = float(pr["prod"].sum())
            rows.append({"sim": s, "method": "argmax", "tau": tau,
                         "rel": _relmean(tpc), "cov": float(tp.in_cache.mean()),
                         "pos": float(tp.groupby("pep")["size"].mean().mean())})
            rows.append({"sim": s, "method": "product", "tau": tau,
                         "rel": _relmean(tpm),
                         "cov": float(len(tpm) / prod_total) if prod_total > 0 else np.nan,
                         "pos": float(pr.groupby("pep")["exp_changed"].mean().mean())})
    summ = pd.DataFrame(rows).sort_values(["sim", "method", "tau"])
    summ.to_csv(CACHE / "_thesis_simcluster.csv", index=False)

    # ---------------- figure: 2x2 ----------------
    def curve(ax, field):
        for s in SIMS:
            for m in ("argmax", "product"):
                d = summ[(summ.sim == s) & (summ.method == m)]
                ax.plot(d["tau"].to_numpy(), d[field].to_numpy(), STY[m], color=COL[(s, m)],
                        lw=1.8, label=f"sim-{s} / {m}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    curve(ax, "rel"); ax.axhline(1.0, color="#d7191c", ls=":", lw=1.0)
    ax.set_xlabel(r"viability threshold $\tau$"); ax.set_ylabel(r"mean $d_{\mathrm{geo}}$ / full MUTANG set")
    ax.set_title("(a) mean distance vs threshold (4 options)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    bins = np.linspace(-1, 1, 80)
    for (s, e), d in cstat.items():
        ax.hist(d["vals"], bins=bins, density=True, histtype="step", lw=1.6,
                color=ENC_COL[(s, e)], ls="-" if e == "onehot" else "--",
                label=f"{s}/{e}  (mean$|x|$={d['mean_abs']:.2f}, $\\in$[{d['mn']:.2f},{d['mx']:.2f}])")
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("cross-position cosine similarity"); ax.set_ylabel("density")
    ax.set_title("(b) similarity distribution (A/B, one-hot/diff)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    curve(ax, "pos")
    ax.set_xlabel(r"viability threshold $\tau$"); ax.set_ylabel("mean # positions changed")
    ax.set_title("(c) positions changed vs threshold", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    curve(ax, "cov"); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"viability threshold $\tau$")
    ax.set_ylabel("fraction of candidates found in cache")
    ax.set_title("(d) fraction scored (cache hit rate)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7)
    for a in axes.ravel():
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("MUTANG+ viability filter: similarity (A/B) x selection (argmax/product)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "rq5_simcluster.pdf")

    sel = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    summ[summ.tau.isin(sel)].pivot_table(index="tau", columns=["sim", "method"],
                                         values=["rel", "pos", "cov"]).to_csv(
        CACHE / "_thesis_simcluster_positions.csv")
    print("\n[rel d_geo / full set]")
    print(summ.pivot_table(index="tau", columns=["sim", "method"], values="rel").round(3).to_string())


if __name__ == "__main__":
    main()
