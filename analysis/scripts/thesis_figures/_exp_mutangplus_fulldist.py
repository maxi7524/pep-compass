"""MUTANG+ viability filter on FRESHLY computed geodesic distances (no cache; full coverage).

Similarity = whitened latent cosine (TANDEM-A) ONLY, in two mutation encodings:
  onehot : direction J_kappa^+ e_{l,a}      ("whitened")
  diff   : direction J_kappa^+ (e_{l,a}-e_{l,parent})   ("e_i - e_j")
Selection = argmax (best residue per viable position) | product (Cartesian product of viable mutations).
=> 4 options total (2 encodings x 2 selections), no pullback-B.

For each parent (385 Veltri-positive + 500 sampled DBAASP) we:
  1. encode it, take the decoder Jacobian (approx, eps=1e-6) and its SVD; build J_kappa^+ and the MUTANG
     single-position mutation set (get_mutations_from_s_u_standard);
  2. assemble a per-peptide candidate UNION = {single mutations} U {argmax candidates over a dense tau
     grid, both encodings} U {a capped uniform sample of the full MUTANG Cartesian product}
     (the product sample doubles as the "full MUTANG set" baseline and as the product-variant pool);
  3. encode the union and compute the graph geodesic distance to the parent (plus Euclidean and
     first-order pullback for verification) -- so every plotted candidate is scored (coverage = 100%).
Distances are saved to results/data/all_in/_cache/_thesis_mutangplus_fulldist.parquet (kept on bury).

Aggregation: per (encoding, selection, tau) the mean d_geo relative to the full-product baseline
(MUTANG=1) and to the single-mutation mean, and the mean number of positions changed.

Env: MPFD_LIMIT (cap #peptides per dataset, for smoke tests), MPFD_PROD_CAP (product sample cap).
GPU is used for encode+geodesic; the similarity/SVD is done in numpy (no tangent-space object), which
avoids the cpu/cuda device mismatch.
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
from fig_rq56_potentials import (  # noqa: E402
    HydrAMPEncoderDecoder, ALPHABET, MAXLEN, GEO_K, KAPPA, THETA_MUT, MIN_DIR,
    PRODUCT_CAP, bound_mutations,
)
from pep_compass.models.encoder_decoder.utils import decoder_jacobian  # noqa: E402
from pep_compass.local_enumeration.mutation.utils import get_mutations_from_s_u_standard  # noqa: E402
from pep_compass.geometry.utils import metric_from_jac  # noqa: E402
from _geodesic import geodesic_distance_to_parent  # noqa: E402

A = len(ALPHABET)
THRESHOLDS = [round(t, 3) for t in np.arange(-0.9, 0.9001, 0.025)]
ENCS = ["onehot", "diff"]
PROD_CAP = int(os.environ.get("MPFD_PROD_CAP", 1800))     # uniform product-sample size per peptide
COL = {("onehot", "argmax"): "#e6550d", ("onehot", "product"): "#fdae6b",
       ("diff", "argmax"): "#3182bd", ("diff", "product"): "#9ecae1"}
STY = {"argmax": "-", "product": "--"}
rng = np.random.default_rng(0)


def parent_geometry(hyd, pep, dev):
    """Return z (1,latent tensor on dev), G (latent,latent np), muts dict, and the whitened unit
    direction matrices V_onehot, V_diff (M,latent np) + labels [(pos,aa)] for the single mutations."""
    z = hyd.encode_peptides([pep])
    jac = decoder_jacobian(lambda x: hyd.decoder_forward(x, softmax=True, flatten=True),
                           z, "approx", {"jacobian_eps": 1e-6})           # (1, ambient, latent)
    U, S, Vh = torch.linalg.svd(jac, full_matrices=False)
    U0 = U[0].detach().cpu().numpy(); S0 = S[0].detach().cpu().numpy(); Vh0 = Vh[0].detach().cpu().numpy()
    G = metric_from_jac(jac)[0].detach().cpu().numpy()
    nsv = S0.shape[0]
    K = min(max(int((S0 > KAPPA).sum()), MIN_DIR), nsv)
    # J_kappa^+ = Vh0[:K].T @ diag(1/S0[:K]) @ U0[:, :K].T   -> (latent, ambient)
    Jpinv = (Vh0[:K].T * (1.0 / S0[:K])) @ U0[:, :K].T
    muts = get_mutations_from_s_u_standard(
        s=S0, u=U0, max_len=MAXLEN, alphabet_size=A,
        direction_significance_threshold=KAPPA, min_number_of_directions=MIN_DIR,
        token_threshold=THETA_MUT)
    muts = {p: m for p, m in muts.items() if p < len(pep) and len(m) > 0}
    muts = bound_mutations(muts, PRODUCT_CAP)
    positions = sorted(muts.keys())
    padded = pep.ljust(MAXLEN)
    parent_aa = {p: ALPHABET.index(padded[p]) for p in positions}
    labels, v_oh, v_df = [], [], []
    for p in positions:
        for a in muts[p]:
            oh = Jpinv[:, p * A + a]
            df = oh - Jpinv[:, p * A + parent_aa[p]]
            v_oh.append(oh / (np.linalg.norm(oh) + 1e-12))
            v_df.append(df / (np.linalg.norm(df) + 1e-12))
            labels.append((p, int(a)))
    return z, G, muts, positions, parent_aa, labels, np.array(v_oh), np.array(v_df)


def seq_of(pep, chosen):
    s = list(pep.ljust(MAXLEN))
    for p, a in chosen.items():
        s[p] = ALPHABET[a]
    return "".join(s[:len(pep)])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyd = HydrAMPEncoderDecoder(jacobian_mode="approx", jacobian_eps=1e-6, field_eps=1e-6, device=dev)
    hyd.eval()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
    print(f"HydrAMP on {dev}")

    # ---- parents: 385 Veltri-positive (the RQ5 set) + 500 sampled DBAASP ----
    veltri = list(dict.fromkeys(
        pd.read_parquet(CACHE / "_thesis_rq5_distances.parquet")["pep"].tolist()))
    ext = pd.read_parquet(CACHE / "parents_ext_veltri_negative_veltri_positive_dbaasp_clean_mic_data.parquet")
    db = ext[ext["dataset"].astype(str).str.contains("dbaasp_clean", na=False)]["sequence"].dropna().astype(str)
    db = db[(db.str.len() <= MAXLEN) & (db.str.len() > 3)
            & db.apply(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].drop_duplicates()
    dbaasp = db.sample(n=min(500, len(db)), random_state=0).tolist()
    lim = int(os.environ.get("MPFD_LIMIT", 0))
    if lim:
        veltri, dbaasp = veltri[:lim], dbaasp[:lim]
    peps = [("veltri_positive", p) for p in veltri] + [("dbaasp", p) for p in dbaasp]
    seen = set()                                              # dedup shared sequences (keep first)
    peps = [(d, p) for d, p in peps if not (p in seen or seen.add(p))]
    print(f"[mpfd] {len(veltri)} veltri-positive + {len(dbaasp)} dbaasp = {len(peps)} unique peptides")

    dist_rows = []                                            # per-candidate distances (saved)
    rec_arg = {e: [] for e in ENCS}                           # argmax: per anchor, per tau
    rec_prod = {e: [] for e in ENCS}                          # product-sample: per candidate, per anchor minS
    base_rows = []                                            # per-peptide baselines

    for ip, (dset, pep) in enumerate(peps):
        try:
            z, G, muts, positions, parent_aa, labels, Voh, Vdf = parent_geometry(hyd, pep, dev)
        except Exception as ex:                               # noqa: BLE001
            print(f"  skip {pep[:12]} ({ex})"); continue
        M = len(labels)
        if M < 3:
            continue
        Smats = {"onehot": Voh @ Voh.T, "diff": Vdf @ Vdf.T}
        label_of = {la: k for k, la in enumerate(labels)}
        pos_items = defaultdict(list)
        for k, (l, a) in enumerate(labels):
            pos_items[l].append((k, a))

        cand = {}                                            # seq -> chosen dict (dedup)

        def add(chosen):
            cand.setdefault(seq_of(pep, chosen), dict(chosen))

        for p in positions:                                  # (1) single mutations
            for a in muts[p]:
                add({p: a})
        for e in ENCS:                                       # (2) argmax union (nested over tau)
            S = Smats[e]
            for i, (li, ai) in enumerate(labels):
                best = {}
                for l, items in pos_items.items():
                    if l == li:
                        continue
                    bs, ba = max(((float(S[i, k]), a) for k, a in items), key=lambda x: x[0])
                    best[l] = (bs, ba)
                order = sorted(best, key=lambda l: -best[l][0])
                chosen = {li: ai}
                add(dict(chosen))
                for l in order:                              # add each nested prefix candidate
                    chosen[l] = best[l][1]
                    add(dict(chosen))
        prod_keys = []                                       # (3) uniform product sample (= baseline pool)
        for _ in range(PROD_CAP):
            chosen = {}
            for p in positions:
                opts = [parent_aa[p]] + list(muts[p])
                a = int(rng.choice(opts))
                if a != parent_aa[p]:
                    chosen[p] = a
            if chosen:
                s = seq_of(pep, chosen)
                if s not in cand:
                    cand[s] = dict(chosen)
                prod_keys.append(s)
        prod_keys = list(dict.fromkeys(prod_keys))

        seqs = list(cand.keys())
        with torch.no_grad():
            zz = hyd.encode_peptides(seqs).detach()
        dg = geodesic_distance_to_parent(hyd, z, zz, k=GEO_K)
        zz_np = zz.cpu().numpy(); z_np = z.detach().cpu().numpy()[0]
        d_eucl = np.linalg.norm(zz_np - z_np[None, :], axis=1)
        diff = zz_np - z_np[None, :]
        d_maha = np.sqrt(np.clip(np.einsum("ni,ij,nj->n", diff, G, diff), 0.0, None))
        seq2geo = {}
        for s, g_, e_, m_ in zip(seqs, dg, d_eucl, d_maha):
            nm = len(cand[s])
            seq2geo[s] = float(g_)
            dist_rows.append({"pep": pep, "dataset": dset, "seq": s, "n_mut": nm,
                              "dist_geo": float(g_), "dist_eucl": float(e_), "dist_maha": float(m_)})

        single_geo = np.mean([seq2geo[seq_of(pep, {p: a})] for p in positions for a in muts[p]])
        base_geo = np.mean([seq2geo[s] for s in prod_keys]) if prod_keys else np.nan
        base_rows.append({"pep": pep, "single_mean": float(single_geo), "base_mean": float(base_geo)})

        for e in ENCS:                                       # records for aggregation
            S = Smats[e]
            for i, (li, ai) in enumerate(labels):            # argmax candidate per tau
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
                                       "dgeo": seq2geo[seq_of(pep, chosen)]})
            for s in prod_keys:                              # product-sample membership (minS per anchor)
                idxs = [label_of[(p, a)] for p, a in cand[s].items()]
                for i in idxs:
                    others = [j for j in idxs if j != i]
                    mins = float(min(S[i, j] for j in others)) if others else np.inf
                    rec_prod[e].append({"pep": pep, "nmut": len(idxs), "dgeo": seq2geo[s], "minS": mins})
        if (ip + 1) % 25 == 0:
            print(f"  ...{ip + 1}/{len(peps)}  ({pep[:10]}, |cand|={len(seqs)})")

    dist = pd.DataFrame(dist_rows)
    dist.to_parquet(CACHE / "_thesis_mutangplus_fulldist.parquet")
    base = pd.DataFrame(base_rows).drop_duplicates("pep").set_index("pep")
    print(f"[mpfd] saved {len(dist)} candidate distances over {dist.pep.nunique()} peptides")

    def relmean(df, valcol="dgeo", base_col="base_mean"):
        if not len(df):
            return np.nan
        g = df.groupby("pep")[valcol].mean()
        r = g / base[base_col].reindex(g.index)
        return float(np.nanmean(r.replace([np.inf, -np.inf], np.nan)))

    rows = []
    for e in ENCS:
        da = pd.DataFrame(rec_arg[e]); dp = pd.DataFrame(rec_prod[e])
        for tau in THRESHOLDS:
            ta = da[da.tau == tau]
            tp = dp[dp.minS > tau]
            rows.append({"enc": e, "method": "argmax", "tau": tau,
                         "rel_base": relmean(ta), "rel_single": relmean(ta, base_col="single_mean"),
                         "abs": float(ta.groupby("pep").dgeo.mean().mean()) if len(ta) else np.nan,
                         "pos": float(ta.groupby("pep")["size"].mean().mean()) if len(ta) else np.nan})
            rows.append({"enc": e, "method": "product", "tau": tau,
                         "rel_base": relmean(tp), "rel_single": relmean(tp, base_col="single_mean"),
                         "abs": float(tp.groupby("pep").dgeo.mean().mean()) if len(tp) else np.nan,
                         "pos": float(tp.groupby("pep")["nmut"].mean().mean()) if len(tp) else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(CACHE / "_thesis_mutangplus_fulldist_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    def curve(ax, field):
        for e in ENCS:
            for m in ("argmax", "product"):
                d = summ[(summ.enc == e) & (summ.method == m)]
                ax.plot(d["tau"].to_numpy(), d[field].to_numpy(), STY[m], color=COL[(e, m)], lw=1.8,
                        label=f"{e} / {m}")
    ax = axes[0]; curve(ax, "rel_base"); ax.axhline(1.0, color="#d7191c", ls=":", lw=1.0)
    ax.set_xlabel(r"viability threshold $\tau$"); ax.set_ylabel(r"mean $d_{\mathrm{geo}}$ / full MUTANG set")
    ax.set_title("(a) mean distance vs threshold", fontweight="bold", fontsize=11); ax.legend(fontsize=8)
    ax = axes[1]; curve(ax, "pos")
    ax.set_xlabel(r"viability threshold $\tau$"); ax.set_ylabel("mean # positions changed")
    ax.set_title("(b) positions changed vs threshold", fontweight="bold", fontsize=11); ax.legend(fontsize=8)
    ax = axes[2]; curve(ax, "abs")
    ax.set_xlabel(r"viability threshold $\tau$"); ax.set_ylabel(r"mean geodesic distance $d_{\mathrm{geo}}$")
    ax.set_title("(c) mean distance vs threshold (absolute)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("MUTANG+ viability filter (whitened similarity): one-hot vs diff, argmax vs product",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "rq5_simcluster.pdf")

    print("\n[rel to full MUTANG set]")
    print(summ.pivot_table(index="tau", columns=["enc", "method"], values="rel_base").round(3).to_string())


if __name__ == "__main__":
    main()
