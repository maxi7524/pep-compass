"""Combine CSV trajectories produced by the optimization runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parse the results root and aggregate output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Root output directory of a grid.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimization_results.csv"),
        help="Combined CSV path.",
    )
    return parser.parse_args()


def _read_config(result_file: Path, root: Path) -> tuple[str, str, str, str]:
    """Resolve experiment identity from the nearest variant configuration.

    :param result_file: Observer trajectory being aggregated.
    :param root: Grid output root.
    :return: Grid ID, optimizer, black box, and LE-BO candidate strategy.
    :raises ValueError: If no parent contains ``resolved_config.json``.
    """
    for parent in result_file.parents:
        config_path = parent / "resolved_config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
            optimizer = config["optimizer"]["name"]
            candidate_strategy = config.get("method", "")
            if optimizer == "lebo":
                candidate_strategy = config["optimizer"]["lebo"].get(
                    "candidate_strategy", candidate_strategy
                )
            return (
                parent.relative_to(root).as_posix(),
                optimizer,
                config["black_box"]["name"],
                candidate_strategy if optimizer == "lebo" else "",
            )
        if parent == root:
            break
    raise ValueError(f"No resolved_config.json found for {result_file}")


def main() -> None:
    """Aggregate observer trajectories without modifying source files."""
    args = _parse_args()
    root = args.results.resolve()
    output = args.output.resolve()
    result_files = sorted(
        path
        for path in root.rglob("*.csv")
        if path.name not in {"grid_manifest.csv", "run_manifest.csv"}
        and "tracking" not in path.parts
        and path.resolve() != output
    )
    fieldnames = [
        "grid_id",
        "optimizer",
        "black_box",
        "candidate_strategy",
        "source_file",
        "time",
        "sequence",
        "score",
        "latent_point",
    ]
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result_file in result_files:
            grid_id, optimizer, black_box, candidate_strategy = _read_config(
                result_file, root
            )
            with result_file.open(encoding="utf-8", newline="") as input_file:
                reader = csv.reader(input_file)
                next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        continue
                    writer.writerow(
                        {
                            "grid_id": grid_id,
                            "optimizer": optimizer,
                            "black_box": black_box,
                            "candidate_strategy": candidate_strategy,
                            "source_file": result_file.relative_to(root),
                            "time": row[0],
                            "sequence": row[1],
                            "score": row[2],
                            "latent_point": ",".join(row[3:]),
                        }
                    )


if __name__ == "__main__":
    main()
