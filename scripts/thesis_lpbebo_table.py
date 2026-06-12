"""Build the LPBEBO+ best-score-per-trajectory table for the 6 APEX seed peptides
and compare it to the PepCompass LE-BO / SORBES-SE numbers from the paper.

Reads CSVs produced by ``scripts/lpbebo_plus.py`` (``results/lpbebo_plus/APEX/*.csv``)
and prints, for each seed peptide, the mean and standard deviation of the per-trajectory
minimum score (``score`` column = -log2 mean MIC; negation is reverted via ``--negate``
if needed).

Usage on Bury:
    /home/kjurasz/pep-compass/.venv/bin/python scripts/thesis_lpbebo_table.py \
        --root /home/kjurasz/pep-compass/results/lpbebo_plus/APEX
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from typing import Iterable

ORDER: list[tuple[str, str]] = [
    ("middle-1", "FLYKWWIRIGRLKL"),
    ("jurand-4", "KYCRRFRWLTFRWL"),
    ("jurand-2", "KFRNRHRWKFKLIFRN"),
    ("jurand-7", "KKYWLIRKWIRLWFLT"),
    ("mammuthusin-3", "KTLKIIRLLF"),
    ("hydrodamin-2", "RMARNLVRYVQGLKKKKVI"),
]

# PepCompass paper LE-BO / SORBES-SE means and standard deviations for the same
# seed peptides, in the order above.
PEPCOMPASS_LEBO_SORBES_SE: list[tuple[float, float]] = [
    (0.50, 0.24),
    (0.60, 0.29),
    (0.50, 0.14),
    (0.60, 0.22),
    (0.50, 0.38),
    (0.58, 0.34),
]


def best_score(path: str) -> tuple[float, int]:
    """Return (min score, num scored rows) over a single trajectory CSV."""
    best = math.inf
    n = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            n += 1
            if v < best:
                best = v
    return best, n


def fmt(mu: float, sd: float) -> str:
    return f"{mu:.2f} \u00b1 {sd:.2f}"


def build(root: str, order: Iterable[tuple[str, str]]) -> list[dict]:
    out = []
    for name, seq in order:
        pattern = os.path.join(root, f"{name}_{seq}_traj*.csv")
        files = sorted(glob.glob(pattern))
        mins: list[float] = []
        nrows: list[int] = []
        for fp in files:
            m, n = best_score(fp)
            if math.isfinite(m):
                mins.append(m)
                nrows.append(n)
        mu = statistics.mean(mins) if mins else float("nan")
        sd = statistics.stdev(mins) if len(mins) > 1 else 0.0
        avg_n = sum(nrows) / len(nrows) if nrows else 0.0
        out.append(
            {
                "name": name,
                "seq": seq,
                "n_traj": len(files),
                "avg_evals": avg_n,
                "min_mu": mu,
                "min_sd": sd,
                "trajectory_mins": mins,
                "files": files,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Directory containing trajectory CSVs.")
    args = p.parse_args()

    rows = build(args.root, ORDER)

    print(f"{'peptide':13s} {'seed':24s} {'#traj':>5s} {'avg_evals':>9s}  {'LPBEBO+ min':>15s}")
    for r in rows:
        print(
            f"{r['name']:13s} {r['seq']:24s} {r['n_traj']:5d} {r['avg_evals']:9.0f}  "
            f"{fmt(r['min_mu'], r['min_sd']):>15s}"
        )

    print()
    print("Comparison vs PepCompass LE-BO / SORBES-SE (lower score = better):")
    print(
        f"{'peptide':13s} {'LPBEBO+ (TANDEM filter)':>24s}    "
        f"{'LE-BO / SORBES-SE':>18s}    {'delta':>7s}"
    )
    for r, (pmu, psd) in zip(rows, PEPCOMPASS_LEBO_SORBES_SE):
        delta = r["min_mu"] - pmu
        print(
            f"{r['name']:13s} {fmt(r['min_mu'], r['min_sd']):>24s}    "
            f"{fmt(pmu, psd):>18s}    {delta:+7.3f}"
        )

    print()
    print("LaTeX-friendly table line per peptide (LPBEBO+ vs LE-BO/SORBES-SE):")
    print(r"\begin{tabular}{lcc}")
    print(r"peptide & LPBEBO+ (MUTANG++ / TANDEM filter) & LE-BO / SORBES-SE \\")
    for r, (pmu, psd) in zip(rows, PEPCOMPASS_LEBO_SORBES_SE):
        print(
            rf"{r['name']} & {r['min_mu']:.2f} $\pm$ {r['min_sd']:.2f} "
            rf"& {pmu:.2f} $\pm$ {psd:.2f} \\"
        )
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
