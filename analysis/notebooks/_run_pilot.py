"""Standalone driver — runs the MUTANG++ geodesic sanity-check on the first
N peptides of the 500-peptide list and writes the resulting CSV.

Usage:
    python _run_pilot.py [N=25] [out_path]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _mutang_geodesic_helpers import run_one_parent, summarize_results
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


def main(n: int, out_csv: Path):
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    encoder_decoder = HydrAMPEncoderDecoder(
        jacobian_mode="approx",
        device=device,
        jacobian_eps=0.05,
        field_eps=0.05,
    )
    encoder_decoder.eval()

    peps_file = HERE / "../../basic_eps_greedy_rl/inputs/sampled_500_peptides.txt"
    all_peptides = [
        line.strip()
        for line in peps_file.read_text().splitlines()
        if line.strip()
    ]
    valid = [p for p in all_peptides if 1 <= len(p) <= 25]
    peps = valid[:n]
    print(f"running on {len(peps)} peptides (lengths: {min(len(p) for p in peps)}–{max(len(p) for p in peps)})")

    rng = random.Random(SEED)
    rows: list[dict] = []
    start = time.time()
    for i, p in enumerate(peps):
        t0 = time.time()
        try:
            parent_rows = run_one_parent(
                encoder_decoder, p,
                top_frac=0.20,
                max_mutants=50,
                n_steps=32,
                t1=1.0,
                rng=rng,
            )
        except Exception as e:
            print(f"  ! {p[:25]}: {type(e).__name__}: {e}")
            continue
        rows.extend(parent_rows)
        dt = time.time() - t0
        elapsed = time.time() - start
        print(
            f"  [{i + 1:3d}/{len(peps)}] {p[:25]:25s} "
            f"len={len(p):2d} mutants={len(parent_rows):3d} {dt:5.1f}s "
            f"(elapsed {elapsed / 60:.1f}m, rows so far {len(rows)})"
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nsaved → {out_csv}  ({len(df)} rows)")

    summary, _ = summarize_results(rows)
    summary_path = out_csv.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved → {summary_path}")
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        HERE / "results/mutang_geodesic_sanity/pilot_25_results.csv"
    )
    main(n, out_csv)
