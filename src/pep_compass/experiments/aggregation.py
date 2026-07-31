"""Aggregate independently persisted composable experiment runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def aggregate_experiment(output_directory: str | Path) -> dict[str, Any]:
    """Scan atomic run results and write normalized aggregate artifacts.

    :param output_directory: Root output directory of one experiment.
    :type output_directory: str | pathlib.Path
    :return: Aggregate summary including statuses and best results by variant.
    :rtype: dict[str, Any]
    """
    root = Path(output_directory)
    rows = []
    for path in sorted(root.glob("**/runs/run_*/result.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                row = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        row["result_path"] = str(path.relative_to(root))
        rows.append(row)
    statuses = Counter(row.get("status", "unknown") for row in rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant_id", "variant_00000"))].append(row)
    variants = {}
    for variant_id, variant_rows in sorted(grouped.items()):
        completed = [
            row
            for row in variant_rows
            if row.get("status") == "completed" and row.get("best_score") is not None
        ]
        best = None
        if completed:
            direction = completed[0].get("objective_direction", "minimize")
            best = (max if direction == "maximize" else min)(
                completed, key=lambda row: float(row["best_score"])
            )
        variants[variant_id] = {
            "runs": len(variant_rows),
            "statuses": dict(Counter(row.get("status", "unknown") for row in variant_rows)),
            "best_sequence": best.get("best_sequence") if best else None,
            "best_score": best.get("best_score") if best else None,
            "objective_name": best.get("objective_name") if best else None,
            "objective_direction": best.get("objective_direction") if best else None,
            "variant_values": variant_rows[0].get("variant_values", {}),
        }
    summary = {
        "discovered_runs": len(rows),
        "statuses": dict(statuses),
        "variants": variants,
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, root / "aggregate_runs.csv")
    temporary = root / "aggregate_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(root / "aggregate_summary.json")
    return summary


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the union of discovered result fields as a normalized CSV."""
    preferred = [
        "status", "variant_id", "run_id", "source_index", "repetition", "seed",
        "input_sequence", "final_candidate_count", "best_sequence", "best_score",
        "objective_name", "objective_direction", "error", "result_path",
    ]
    fields = [name for name in preferred if any(name in row for row in rows)]
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(fields)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, *extras])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
