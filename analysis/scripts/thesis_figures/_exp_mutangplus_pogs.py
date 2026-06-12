"""How MUTANG+ does on the PoGS geodesic distance (run on bury).

Identical whitened argmax/product viability selection as _exp_mutangplus_eucl.py, but each
candidate is scored by the PoGS geodesic distance d_pogs (lambda=0; ambient chord length of the
ADAM-optimised path, no metric G -- see _pogs.py) instead of the Euclidean latent distance.
For each (encoding in {onehot,diff}) x (selection in {argmax,product}) and threshold tau it
reports the mean d_pogs of the retained set relative to the full MUTANG (product-sample) set,
and the mean number of positions changed. Saves a summary CSV and the plot rq5_mutangplus_pogs.pdf.
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
from _pogs import pogs_distance_to_parent  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LIM = int(os.environ.get("MPE_LIMIT", 0))          # cap #peptides (0 = all), for smoke tests
PGS_SEG = int(os.environ.get("PGS_SEG", 8))
PGS_STEPS = int(os.environ.get("PGS_STEPS", 400))  # rank-stable from ~300; 400 for a little margin
PGS_MU = float(os.environ.get("PGS_MU", 1e-2))
PGS_LR = float(os.environ.get("PGS_LR", 1e-3))
DEC_BATCH = int(os.environ.get("DEC_BATCH", 2048))   # batch size for the recovery (decode) test


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
    print(f"[mpp] {len(veltri)} veltri-positive parents | device={DEV} "
          f"| PoGS seg={PGS_SEG} steps={PGS_STEPS}")

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
            zz = hyd.encode_peptides(seqs).detach()
            z0 = hyd.encode_peptides([pep]).detach()[0]
            # recovery (self-consistency) test: a candidate passes if HydrAMP round-trips it
            # exactly, argmax(Dec(enc(seq))) == seq. The user-requested recon variant only counts a
            # candidate's distance once the chosen peptide passes this test.
            dec = hyd.decode_peptides(zz, batch_size=DEC_BATCH)
        recon = {s: bool(ds == s) for s, ds in zip(seqs, dec)}
        d = pogs_distance_to_parent(hyd, z0, zz, lam=0.0, n_seg=PGS_SEG,
                                    steps=PGS_STEPS, mu=PGS_MU, lr=PGS_LR)   # PoGS, not Euclidean
        seq2d = {s: float(dd) for s, dd in zip(seqs, d)}
        base = np.mean([seq2d[s] for s in prod_keys]) if prod_keys else np.nan
        base_rec = [seq2d[s] for s in prod_keys if recon[s]]
        base_rows.append({"pep": pep, "base_mean": float(base),
                          "base_mean_recon": float(np.mean(base_rec)) if base_rec else np.nan})

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
                    cseq = seq_of(pep, chosen)
                    rec_arg[e].append({"pep": pep, "tau": tau, "size": len(chosen),
                                       "d": seq2d[cseq], "recon": recon[cseq]})
            for s in prod_keys:
                idxs = [label_of[(p, a)] for p, a in cand[s].items()]
                for i in idxs:
                    others = [j for j in idxs if j != i]
                    mins = float(min(S[i, j] for j in others)) if others else np.inf
                    rec_prod[e].append({"pep": pep, "nmut": len(idxs), "d": seq2d[s],
                                        "minS": mins, "recon": recon[s]})
        if (ip + 1) % 25 == 0:
            print(f"  ...{ip + 1}/{len(veltri)}", flush=True)

    # PoGS is an ABSOLUTE distance (ambient chord length), comparable across peptides -- unlike the
    # cloud-relative Isomap graph geodesic it does NOT need per-peptide normalisation to MUTANG=1.
    # We report the absolute mean d_pogs of the retained set, aggregated per-peptide-first (each
    # peptide weighted equally) and the full-MUTANG (product-sample) absolute mean for reference.
    base_df = pd.DataFrame(base_rows).drop_duplicates("pep").set_index("pep")
    full_dabs = float(base_df["base_mean"].mean())
    full_dabs_recon = float(base_df["base_mean_recon"].mean())

    def per_pep_mean(frame, recon_only=False):
        f = frame[frame["recon"]] if recon_only else frame
        return float(f.groupby("pep")["d"].mean().mean()) if len(f) else np.nan

    def _tail(a, top):
        m = max(1, int(np.ceil(0.05 * len(a))))
        s = np.sort(a)
        return s[-m:].mean() if top else s[:m].mean()

    def dist_stats(frame):
        """Median-over-peptides of the retained set's [mean, median, closest-5%, farthest-5%] d_pogs."""
        if not len(frame):
            return dict(mean=np.nan, median=np.nan, close5=np.nan, far5=np.nan)
        mm, md, lo, hi = [], [], [], []
        for _, g in frame.groupby("pep"):
            d = g["d"].to_numpy()
            mm.append(d.mean()); md.append(np.median(d))
            lo.append(_tail(d, top=False)); hi.append(_tail(d, top=True))
        return dict(mean=np.median(mm), median=np.median(md),
                    close5=np.median(lo), far5=np.median(hi))

    # FULL summary (all candidates) and RECON summary (only candidates that pass the recovery test);
    # for the recon variant the retained mean and the position count are computed over reconstructing
    # candidates only -- "check the distance only after the chosen peptide passes recovery". Each row
    # also carries the kept-set distribution (mean/median/closest-5%/farthest-5%) for the tables.
    def build_summary(recon_only):
        rows = []
        for e in ENCS:
            da = pd.DataFrame(rec_arg[e]); dp = pd.DataFrame(rec_prod[e])
            for tau in THRESHOLDS:
                ta = da[da.tau == tau]; tp = dp[dp.minS > tau]
                if recon_only:
                    ta = ta[ta["recon"]]; tp = tp[tp["recon"]]
                rows.append({"enc": e, "method": "argmax", "tau": tau,
                             "dabs": per_pep_mean(ta),
                             "pos": float(ta.groupby("pep")["size"].mean().mean() - 1) if len(ta) else np.nan,
                             **dist_stats(ta)})
                rows.append({"enc": e, "method": "product", "tau": tau,
                             "dabs": per_pep_mean(tp),
                             "pos": float(tp.groupby("pep")["nmut"].mean().mean()) if len(tp) else np.nan,
                             **dist_stats(tp)})
        return pd.DataFrame(rows)

    summ = build_summary(recon_only=False)
    summ_recon = build_summary(recon_only=True)
    n_rec = sum(int(v) for e in ENCS for v in pd.DataFrame(rec_prod[e])["recon"]) if rec_prod[ENCS[0]] else 0
    print(f"[mpp] full-MUTANG absolute mean d_pogs (reference) = {full_dabs:.3f}  "
          f"| recon-only full-pool mean = {full_dabs_recon:.3f}")
    summ.to_csv(CACHE / "_thesis_mutangplus_pogs_summary.csv", index=False)
    summ_recon.to_csv(CACHE / "_thesis_mutangplus_pogs_recon_summary.csv", index=False)
    print(f"[mpp] saved recon summary ({len(summ_recon)} rows)")
    print(f"[mpp] saved summary ({len(summ)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for e in ENCS:
        for m in ("argmax", "product"):
            d_ = summ[(summ.enc == e) & (summ.method == m)].sort_values("tau")
            axes[0].plot(d_["tau"], d_["dabs"], STY[m], color=COL[(e, m)], lw=1.8, label=f"{e}/{m}")
            axes[1].plot(d_["tau"], d_["pos"], STY[m], color=COL[(e, m)], lw=1.8, label=f"{e}/{m}")
    axes[0].axhline(full_dabs, color="0.6", ls=":", lw=1, label="full MUTANG")
    axes[0].set_xlabel(r"viability threshold $\tau$")
    axes[0].set_ylabel(r"mean $d_{\mathrm{PoGS}}$ (absolute)")
    axes[0].set_title("(a) MUTANG+ PoGS distance vs $\\tau$", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"viability threshold $\tau$"); axes[1].set_ylabel("mean # positions changed")
    axes[1].set_title("(b) positions changed vs $\\tau$", fontsize=11, fontweight="bold")
    fig.suptitle("MUTANG+ whitened viability filter vs PoGS geodesic distance",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "rq5_mutangplus_pogs.pdf")


if __name__ == "__main__":
    main()
