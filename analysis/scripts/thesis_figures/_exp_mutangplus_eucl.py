"""How MUTANG+ does on the Euclidean latent distance (run on bury; no geodesics -> cheap).

Reuses the whitened-similarity selection of _exp_mutangplus_fulldist.py but scores candidates
by Euclidean latent distance d_eucl = ||enc(mutant) - enc(parent)|| instead of the graph geodesic.
For each (encoding in {onehot,diff}) x (selection in {argmax,product}) and viability threshold tau
it reports the mean d_eucl of the retained set relative to the full MUTANG (product-sample) set,
and the mean number of positions changed. Saves a summary CSV and the plot rq5_mutangplus_eucl.pdf.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PROJECT_ROOT, CACHE, save  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from _exp_mutangplus_fulldist import (  # noqa: E402
    parent_geometry, seq_of, THRESHOLDS, ENCS, PROD_CAP, COL, STY, rng,
)
from fig_rq56_potentials import HydrAMPEncoderDecoder  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIM = int(os.environ.get("MPE_LIMIT", 0))     # cap #peptides per dataset (0 = all), for smoke tests


def main():
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6,
                                field_eps=1e-6, device=DEV)
    hyd.eval()
    pool = pd.read_parquet(CACHE / "parents_hydramp_veltri_positive.parquet")["sequence"]
    veltri = pool.dropna().astype(str)
    veltri = veltri[(veltri.str.len() <= 25) & (veltri.str.len() > 3)
                    & veltri.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates().tolist()
    if LIM:
        veltri = veltri[:LIM]
    print(f"[mpe] {len(veltri)} veltri-positive parents | device={DEV}")

    rec_arg = {e: [] for e in ENCS}
    rec_prod = {e: [] for e in ENCS}
    base_rows = []
    for ip, pep in enumerate(veltri):
        try:
            z, G, muts, positions, parent_aa, labels, Voh, Vdf = parent_geometry(hyd, pep, DEV)
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {pep[:12]} ({ex})"); continue
        if len(labels) < 3:
            continue
        Smats = {"onehot": Voh @ Voh.T, "diff": Vdf @ Vdf.T}
        label_of = {la: k for k, la in enumerate(labels)}
        pos_items = defaultdict(list)
        for k, (l, a) in enumerate(labels):
            pos_items[l].append((k, a))

        cand = {}
        def add(chosen):
            cand.setdefault(seq_of(pep, chosen), dict(chosen))
        for p in positions:
            for a in muts[p]:
                add({p: a})
        for e in ENCS:
            S = Smats[e]
            for i, (li, ai) in enumerate(labels):
                best = {}
                for l, items in pos_items.items():
                    if l == li:
                        continue
                    bs, ba = max(((float(S[i, k]), a) for k, a in items), key=lambda x: x[0])
                    best[l] = (bs, ba)
                chosen = {li: ai}; add(dict(chosen))
                for l in sorted(best, key=lambda l: -best[l][0]):
                    chosen[l] = best[l][1]; add(dict(chosen))
        prod_keys = []
        for _ in range(PROD_CAP):
            chosen = {}
            for p in positions:
                a = int(rng.choice([parent_aa[p]] + list(muts[p])))
                if a != parent_aa[p]:
                    chosen[p] = a
            if chosen:
                s = seq_of(pep, chosen)
                cand.setdefault(s, dict(chosen)); prod_keys.append(s)
        prod_keys = list(dict.fromkeys(prod_keys))

        seqs = list(cand.keys())
        with torch.no_grad():
            zz = hyd.encode_peptides(seqs).detach().cpu().numpy()
            z0 = hyd.encode_peptides([pep]).detach().cpu().numpy()[0]
        d = np.linalg.norm(zz - z0[None, :], axis=1)          # EUCLIDEAN, not geodesic
        seq2d = {s: float(dd) for s, dd in zip(seqs, d)}
        base = np.mean([seq2d[s] for s in prod_keys]) if prod_keys else np.nan
        base_rows.append({"pep": pep, "base_mean": float(base)})

        for e in ENCS:
            S = Smats[e]
            for i, (li, ai) in enumerate(labels):
                best = {}
                for l, items in pos_items.items():
                    if l == li:
                        continue
                    bs, ba = max(((float(S[i, k]), a) for k, a in items), key=lambda x: x[0])
                    best[l] = (bs, ba)
                for tau in THRESHOLDS:
                    chosen = {li: ai}
                    for l in best:
                        if best[l][0] > tau:
                            chosen[l] = best[l][1]
                    rec_arg[e].append({"pep": pep, "tau": tau, "size": len(chosen),
                                       "d": seq2d[seq_of(pep, chosen)]})
            for s in prod_keys:
                idxs = [label_of[(p, a)] for p, a in cand[s].items()]
                for i in idxs:
                    others = [j for j in idxs if j != i]
                    mins = float(min(S[i, j] for j in others)) if others else np.inf
                    rec_prod[e].append({"pep": pep, "nmut": len(idxs), "d": seq2d[s], "minS": mins})
        if (ip + 1) % 25 == 0:
            print(f"  ...{ip + 1}/{len(veltri)}", flush=True)

    base = pd.DataFrame(base_rows).drop_duplicates("pep").set_index("pep")["base_mean"]

    def relmean(df_):
        df_ = df_.copy(); df_["base"] = df_["pep"].map(base)
        return (df_["d"] / df_["base"]).groupby(df_["tau"] if "tau" in df_ else None)

    rows = []
    for e in ENCS:
        da = pd.DataFrame(rec_arg[e]); dp = pd.DataFrame(rec_prod[e])
        da["base"] = da["pep"].map(base)
        for tau in THRESHOLDS:
            ta = da[da.tau == tau]
            tp = dp[dp.minS > tau].copy(); tp["base"] = tp["pep"].map(base)
            rows.append({"enc": e, "method": "argmax", "tau": tau,
                         "rel": float((ta["d"] / ta["base"]).mean()), "pos": float(ta["size"].mean() - 1)})
            rows.append({"enc": e, "method": "product", "tau": tau,
                         "rel": float((tp["d"] / tp["base"]).mean()) if len(tp) else np.nan,
                         "pos": float(tp["nmut"].mean()) if len(tp) else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(CACHE / "_thesis_mutangplus_eucl_summary.csv", index=False)
    print(f"[mpe] saved summary ({len(summ)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for e in ENCS:
        for m in ("argmax", "product"):
            d_ = summ[(summ.enc == e) & (summ.method == m)].sort_values("tau")
            axes[0].plot(d_["tau"], d_["rel"], STY[m], color=COL[(e, m)], lw=1.8, label=f"{e}/{m}")
            axes[1].plot(d_["tau"], d_["pos"], STY[m], color=COL[(e, m)], lw=1.8, label=f"{e}/{m}")
    axes[0].axhline(1.0, color="0.6", ls=":", lw=1)
    axes[0].set_xlabel(r"viability threshold $\tau$"); axes[0].set_ylabel(r"mean $d_{\mathrm{eucl}}$ / full set")
    axes[0].set_title("(a) MUTANG+ Euclidean proximity vs $\\tau$", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"viability threshold $\tau$"); axes[1].set_ylabel("mean # positions changed")
    axes[1].set_title("(b) positions changed vs $\\tau$", fontsize=11, fontweight="bold")
    fig.suptitle("MUTANG+ whitened viability filter vs Euclidean latent distance",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq5_mutangplus_eucl.pdf")


if __name__ == "__main__":
    main()
