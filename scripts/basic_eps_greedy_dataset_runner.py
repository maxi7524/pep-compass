from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .rl_peptide_optimizer import run_basic_epsilon_greedy  # type: ignore
except ImportError:
    from rl_peptide_optimizer import run_basic_epsilon_greedy


def _load_peptides(path: Path) -> list[str]:
    peptides: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        seq = raw.strip().upper()
        if not seq or seq in seen:
            continue
        peptides.append(seq)
        seen.add(seq)
    return peptides


def run_dataset(
    peptides_file: str,
    repeats_per_peptide: int = 10,
    num_workers: int = 6,
    worker_id: int = 0,
    n_epochs: int = 1500,
    max_steps: int = 200,
    device: str = "cuda",
    output_dir: str = "results\\basic_eps_greedy_rl_500x10",
    start_selection: str = "random",
    verbose: bool = False,
) -> dict:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")

    in_path = Path(peptides_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    peptides = _load_peptides(in_path)
    tasks: list[tuple[int, str, int]] = []
    for p_idx, peptide in enumerate(peptides):
        for rep in range(repeats_per_peptide):
            tasks.append((p_idx, peptide, rep))

    assigned = [
        task for task_idx, task in enumerate(tasks) if task_idx % num_workers == worker_id
    ]

    completed = 0
    failed = 0
    manifest_rows: list[dict] = []
    for p_idx, peptide, rep in assigned:
        run_name = f"p{p_idx:04d}_r{rep + 1:02d}_w{worker_id}"
        try:
            result = run_basic_epsilon_greedy(
                start_peptide=peptide,
                start_peptides="",
                start_peptides_file="",
                start_selection=start_selection,
                n_epochs=n_epochs,
                max_steps=max_steps,
                device=device,
                output_dir=str(out_path),
                run_name=run_name,
                seed=1000 + p_idx * 37 + rep,
                verbose=verbose,
            )
            completed += 1
            manifest_rows.append(
                {
                    "worker_id": worker_id,
                    "peptide_index": p_idx,
                    "peptide": peptide,
                    "repeat": rep + 1,
                    "run_name": result.get("run_name", run_name),
                    "run_id": result.get("run_id"),
                    "best_peptide": result.get("best_peptide"),
                    "best_log2mic": result.get("best_log2mic"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            failed += 1
            manifest_rows.append(
                {
                    "worker_id": worker_id,
                    "peptide_index": p_idx,
                    "peptide": peptide,
                    "repeat": rep + 1,
                    "run_name": run_name,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = {
        "worker_id": worker_id,
        "num_workers": num_workers,
        "peptides_file": str(in_path),
        "total_peptides": len(peptides),
        "repeats_per_peptide": repeats_per_peptide,
        "assigned_tasks": len(assigned),
        "completed": completed,
        "failed": failed,
    }
    (out_path / f"worker_{worker_id:02d}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_path / f"worker_{worker_id:02d}_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failed > 0:
        raise RuntimeError(
            f"Worker {worker_id} finished with failures: {failed}/{len(assigned)} tasks failed."
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptides_file", required=True)
    parser.add_argument("--repeats_per_peptide", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=1500)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results\\basic_eps_greedy_rl_500x10")
    parser.add_argument("--start_selection", default="random")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_dataset(
        peptides_file=args.peptides_file,
        repeats_per_peptide=args.repeats_per_peptide,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=args.output_dir,
        start_selection=args.start_selection,
        verbose=args.verbose,
    )
