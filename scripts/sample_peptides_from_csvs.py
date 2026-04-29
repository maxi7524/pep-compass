from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PEPTIDE_COLUMNS = {
    "mutant",
    "parent",
    "mutant_sequence",
    "parent_sequence",
    "sequence",
    "seq",
    "peptide",
    "peptides",
}

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def _is_valid_peptide(text: str, max_len: int) -> bool:
    seq = text.strip().upper()
    return 0 < len(seq) <= max_len and set(seq).issubset(VALID_AAS)


def _iter_csv_files(root: Path) -> list[Path]:
    skip = {".git", ".venv", ".env", "__pycache__", "node_modules"}
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        if any(part in skip or part.startswith(".") for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def sample_peptides(
    root: str = ".",
    dataset_subdir: str = "results/mutants/mutants",
    sample_size: int = 500,
    max_len: int = 25,
    seed: int = 42,
    output_file: str = "basic_eps_greedy_rl/inputs/sampled_500_peptides.txt",
    meta_file: str = "basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json",
    max_per_csv: int = 10_000,   # 🔥 KLUCZOWE
) -> dict:

    rng = random.Random(seed)

    root_path = Path(root).resolve()
    out_path = Path(output_file)
    meta_path = Path(meta_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_root = (root_path / Path(dataset_subdir)).resolve()

    if preferred_root.exists():
        csv_files = _iter_csv_files(preferred_root)
        scan_root = preferred_root
        scan_mode = "preferred_subdir_only"
    else:
        csv_files = _iter_csv_files(root_path)
        scan_root = root_path
        scan_mode = "full_root_fallback"

    pool: list[str] = []
    source_hits: dict[str, int] = {}
    read_errors: list[str] = []

    # =========================
    # 🔥 MAIN FIX: per-CSV cap
    # =========================
    for csv_path in csv_files:
        try:
            local: list[str] = []

            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                reader = csv.DictReader(fh)

                if not reader.fieldnames:
                    continue

                name_map = {name.lower().strip(): name for name in reader.fieldnames}
                peptide_fields = [name_map[k] for k in PEPTIDE_COLUMNS if k in name_map]

                if not peptide_fields:
                    continue

                for row in reader:
                    for field in peptide_fields:
                        val = (row.get(field) or "").strip().upper()
                        if _is_valid_peptide(val, max_len=max_len):
                            local.append(val)

            # 🔥 limit per CSV
            if len(local) > max_per_csv:
                local = rng.sample(local, max_per_csv)

            pool.extend(local)

            source_hits[str(csv_path.relative_to(scan_root))] = len(local)

        except Exception as exc:
            read_errors.append(f"{csv_path}: {exc}")
            continue

    # =========================
    # FINAL SAMPLING
    # =========================
    rng.shuffle(pool)
    chosen = pool[: min(sample_size, len(pool))]

    # write output
    out_path.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")

    meta = {
        "root": str(root_path),
        "scan_mode": scan_mode,
        "scan_root": str(scan_root),
        "dataset_subdir_requested": dataset_subdir,
        "sample_size_requested": sample_size,
        "sample_size_obtained": len(chosen),
        "pool_size_after_sampling": len(pool),
        "max_per_csv": max_per_csv,
        "seed": seed,
        "max_len": max_len,
        "output_file": str(out_path),
        "sources_with_hits": source_hits,
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:50],
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if read_errors:
        print(
            f"Warning: {len(read_errors)} CSV files could not be read. "
            f"See metadata at {meta_path}.",
            file=sys.stderr,
        )

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset_subdir", default="results/mutants/mutants")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="basic_eps_greedy_rl/inputs/sampled_500_peptides.txt")
    parser.add_argument("--meta_file", default="basic_eps_greedy_rl/inputs/sampled_500_peptides_meta.json")

    args = parser.parse_args()

    result = sample_peptides(
        root=args.root,
        dataset_subdir=args.dataset_subdir,
        sample_size=args.sample_size,
        max_len=args.max_len,
        seed=args.seed,
        output_file=args.output_file,
        meta_file=args.meta_file,
    )

    print(json.dumps(result, indent=2))