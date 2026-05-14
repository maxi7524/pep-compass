"""Build a position-specific BLOSUM-style log-odds substitution matrix from
parent-mutant peptide CSVs.

Formula recap (see plans/create-blosum-matrix-calculator-tidy-bubble.md):

  c'_p(a, b) = #rows with row.position == p, parent[p]=a, mutant[p]=b
  n_p(a, b)  = c'_p(a, b) + c'_p(b, a)             # symmetric
  N_p        = sum_{a,b} n_p(a, b)
  q_p(a, b)  = n_p(a, b) / N_p
  P_p(a)     = sum_b q_p(a, b)                     # marginal at position p
  e_p(a, b)  = P_p(a) * P_p(b)
  s_p(a, b)  = 2 * log2( q_p(a, b) / e_p(a, b) )   # half-bit log-odds

The output matrix M is 500x500 = (n_aa * P_max)^2 with index(a, p) = p*n_aa + AA.index(a).
M is block-diagonal: M[i, j] = s_p(a, b) only when i and j refer to the same position,
otherwise 0. Unobserved entries within a block are NaN in the float file and stored
as the integer sentinel -128 in the labeled CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_AAS = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NAN_SENTINEL = -128

log = logging.getLogger("build_position_blosum")


def load_rows(data_root: Path, max_positions: int) -> tuple[pd.DataFrame, dict]:
    csv_files = sorted(data_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found under {data_root}")

    log.info("Loading %d CSV files from %s", len(csv_files), data_root)
    t0 = time.time()
    frames = []
    for i, f in enumerate(csv_files, 1):
        t_file = time.time()
        df = pd.read_csv(f, usecols=["mutant", "position", "parent"])
        df["__source__"] = str(f.relative_to(data_root))
        frames.append(df)
        log.info("  [%d/%d] %s: %d rows (%.1fs)",
                 i, len(csv_files), f.relative_to(data_root), len(df), time.time() - t_file)
    all_df = pd.concat(frames, ignore_index=True)
    n_total = len(all_df)
    log.info("Concatenated %d total rows in %.1fs; validating...", n_total, time.time() - t0)

    # Validate
    same_len = all_df["parent"].str.len() == all_df["mutant"].str.len()
    pos_in_range = (all_df["position"] >= 0) & (all_df["position"] < max_positions)
    pos_lt_len = all_df["position"] < all_df["parent"].str.len()

    keep = same_len & pos_in_range & pos_lt_len
    dropped = int((~keep).sum())
    all_df = all_df.loc[keep].reset_index(drop=True)
    log.info("Validation: kept %d rows, dropped %d", len(all_df), dropped)

    # Position indexing sanity check: with 0-based indexing, parent[position] should
    # differ from mutant[position] in the vast majority of rows (these are mutations).
    log.info("Detecting position indexing convention...")
    pos = all_df["position"].to_numpy()
    parents = all_df["parent"].to_numpy()
    mutants = all_df["mutant"].to_numpy()
    par_chars = np.array([p[i] for p, i in zip(parents, pos)])
    mut_chars = np.array([m[i] for m, i in zip(mutants, pos)])
    n_diff_0based = int((par_chars != mut_chars).sum())
    frac_0based = n_diff_0based / max(len(all_df), 1)
    indexing = "0-based" if frac_0based >= 0.9 else "1-based-or-unknown"
    log.info("Position indexing: %s (parent[pos]!=mutant[pos] in %.2f%% of rows)",
             indexing, 100 * frac_0based)

    meta = {
        "alphabet": AMINO_ACIDS,
        "max_positions": max_positions,
        "n_rows_total": n_total,
        "n_rows_used": int(len(all_df)),
        "n_rows_dropped": dropped,
        "position_indexing": indexing,
        "frac_position_differs_0based": frac_0based,
        "csv_files": [str(f.relative_to(data_root)) for f in csv_files],
    }
    return all_df, meta


def build_directed_counts(df: pd.DataFrame, max_positions: int) -> np.ndarray:
    """Return C[p, a, b] = #rows at position p with parent[p]=a, mutant[p]=b."""
    log.info("Building directed substitution counts for %d rows...", len(df))
    t0 = time.time()
    pos = df["position"].to_numpy()
    parents = df["parent"].to_numpy()
    mutants = df["mutant"].to_numpy()

    par_aa = np.array([p[i] for p, i in zip(parents, pos)])
    mut_aa = np.array([m[i] for m, i in zip(mutants, pos)])
    log.info("  extracted parent/mutant chars at row position (%.1fs)", time.time() - t0)

    # Drop rows whose parent or mutant char is outside the 20-AA alphabet
    valid = np.array([(a in AA_TO_IDX and b in AA_TO_IDX) for a, b in zip(par_aa, mut_aa)])
    n_invalid = int((~valid).sum())
    if n_invalid:
        log.warning("  dropping %d rows with non-standard AA characters", n_invalid)
    pos = pos[valid]
    par_idx = np.array([AA_TO_IDX[a] for a in par_aa[valid]])
    mut_idx = np.array([AA_TO_IDX[b] for b in mut_aa[valid]])

    C = np.zeros((max_positions, N_AAS, N_AAS), dtype=np.int64)
    np.add.at(C, (pos, par_idx, mut_idx), 1)
    log.info("  built %d x %d x %d count tensor in %.1fs (total %d obs)",
             *C.shape, time.time() - t0, int(C.sum()))
    return C


def compute_blosum_per_position(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (s, n) where s[p, a, b] is the BLOSUM log-odds at position p
    (NaN where undefined), and n[p, a, b] is the symmetric pair count."""
    max_positions = C.shape[0]
    # Symmetric counts: n_p(a, b) = c'_p(a, b) + c'_p(b, a)
    n = C + C.transpose(0, 2, 1)
    # Avoid double-counting the diagonal: c'_p(a, a) is added to itself.
    # Fix: subtract one copy on the diagonal so n_p(a, a) = c'_p(a, a).
    for a in range(N_AAS):
        n[:, a, a] = C[:, a, a]
    n = n.astype(np.float64)

    log.info("Computing per-position log-odds for %d positions...", max_positions)
    t0 = time.time()
    s = np.full_like(n, np.nan)
    for p in range(max_positions):
        N_p = n[p].sum()
        if N_p <= 0:
            log.info("  position %2d: no observations -> entire block NaN", p)
            continue
        q = n[p] / N_p
        P = q.sum(axis=1)  # marginal P_p(a) = row sum of symmetric q
        e = np.outer(P, P)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((q > 0) & (e > 0), q / e, np.nan)
            score = 2.0 * np.log2(ratio)
        score = np.where(np.isfinite(score), score, np.nan)
        s[p] = score
        n_finite = int(np.isfinite(score).sum())
        log.info("  position %2d: rows=%d  finite_entries=%d/400",
                 p, int(N_p // 2), n_finite)
    log.info("Per-position computation done in %.1fs", time.time() - t0)
    return s, n


def assemble_full_matrix(s_per_pos: np.ndarray) -> np.ndarray:
    max_positions = s_per_pos.shape[0]
    size = max_positions * N_AAS
    M = np.zeros((size, size), dtype=np.float64)
    for p in range(max_positions):
        start = p * N_AAS
        M[start:start + N_AAS, start:start + N_AAS] = s_per_pos[p]
    return M


def labels(max_positions: int) -> list[str]:
    return [f"{aa}_{p}" for p in range(max_positions) for aa in AMINO_ACIDS]


def save_outputs(out_dir: Path, M: np.ndarray, n: np.ndarray, meta: dict, max_positions: int) -> None:
    log.info("Saving outputs to %s ...", out_dir)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "position_blosum.npy", M)
    np.save(out_dir / "position_blosum_counts.npy", n)

    # Integer-rounded labeled CSV with NaN sentinel
    M_int = np.where(np.isnan(M), NAN_SENTINEL, np.rint(M)).astype(np.int64)
    lbls = labels(max_positions)
    df_int = pd.DataFrame(M_int, index=lbls, columns=lbls)
    csv_path = out_dir / "position_blosum.csv"
    with open(csv_path, "w") as fh:
        fh.write(f"# Position-specific BLOSUM log-odds (half-bit), rounded to int.\n")
        fh.write(f"# NaN sentinel: {NAN_SENTINEL}. Index/columns: AA_position (0-indexed).\n")
        df_int.to_csv(fh)

    # Add per-position observation counts to meta for diagnostics
    meta = dict(meta)
    meta["rows_per_position"] = [int(n[p].sum() // 2) for p in range(max_positions)]
    with open(out_dir / "position_blosum_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved 4 files (npy + csv + counts + meta) in %.1fs", time.time() - t0)


def _configure_logging(verbose: bool) -> None:
    # Line-buffered stdout so logs appear promptly over ssh / tee / nohup.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build position-specific BLOSUM matrix.")
    parser.add_argument("--data-root", type=Path, default=Path("results/mutants/mutants"))
    parser.add_argument("--out-dir", type=Path, default=Path("blosum"))
    parser.add_argument("--max-positions", type=int, default=25)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    t_start = time.time()
    log.info("=== build_position_blosum starting ===")
    log.info("data_root=%s  out_dir=%s  max_positions=%d",
             args.data_root, args.out_dir, args.max_positions)

    df, meta = load_rows(args.data_root, args.max_positions)
    C = build_directed_counts(df, args.max_positions)
    s, n = compute_blosum_per_position(C)
    M = assemble_full_matrix(s)

    log.info("Matrix shape=%s  finite_entries=%d  NaN_entries=%d",
             M.shape, int(np.isfinite(M).sum()), int(np.isnan(M).sum()))

    save_outputs(args.out_dir, M, n, meta, args.max_positions)
    log.info("=== done in %.1fs ===", time.time() - t_start)


if __name__ == "__main__":
    main()
