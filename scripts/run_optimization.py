"""Run configured peptide-optimization experiments locally or through Slurm."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_BLACK_BOXES = {"apex", "battleamp", "hydrophobicity", "toxipep"}
SUPPORTED_METHODS = {
    "lebo",
    "lpbebo",
    "lams",
    "tandem",
    "move",
    "random_walker",
    "random_mutang",
}
SUPPORTED_OPTIMIZERS = {"lebo", "random_mutation", "cmaes", "saasbo", "lambo2"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a child configuration into its parent."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    """Load JSON configuration and resolve relative inheritance and CSV paths."""
    path = path.resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Circular config inheritance detected at {path}")
    visited.add(path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    parent = config.pop("extends", None)
    if parent is not None:
        config = _deep_merge(_load_config(path.parent / parent, visited), config)
    if config.get("input_csv") is not None:
        config["input_csv"] = str((path.parent / config["input_csv"]).resolve())
    return config


def _load_sequences(path: Path) -> list[dict[str, Any]]:
    """Read named starting peptides and optional per-sequence repetitions."""
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"name", "sequence"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError("Input CSV must contain name and sequence columns")
        sequences = []
        names = set()
        for row_number, row in enumerate(reader, start=2):
            name = row["name"].strip()
            sequence = row["sequence"].strip().upper()
            repetitions = int((row.get("repetitions") or "1").strip())
            if not name or not sequence:
                raise ValueError(f"Empty name or sequence in CSV row {row_number}")
            if name in names:
                raise ValueError(f"Duplicate sequence name in input CSV: {name}")
            if repetitions < 1:
                raise ValueError(
                    f"Repetitions must be positive in CSV row {row_number}"
                )
            names.add(name)
            sequences.append(
                {"name": name, "sequence": sequence, "repetitions": repetitions}
            )
    if not sequences:
        raise ValueError("Input CSV must contain at least one sequence")
    return sequences


def _set_nested_value(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Grid path does not reference a config object: {path}")
        target = child
    target[keys[-1]] = value


def _expand_grid(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand explicit dotted-path value lists into a Cartesian parameter grid."""
    base = deepcopy(config)
    grid = base.pop("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("grid must map dotted config paths to value lists")
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid values for {path} must be a non-empty list")
    if not grid:
        return [({}, base)]
    paths = list(grid)
    expanded = []
    for values in product(*(grid[path] for path in paths)):
        parameters = dict(zip(paths, values))
        variant = deepcopy(base)
        for path, value in parameters.items():
            _set_nested_value(variant, path, value)
        expanded.append((parameters, variant))
    return expanded


def _validate_config(config: dict[str, Any]) -> None:
    optimizer_name = config["optimizer"]["name"]
    black_box_name = config["black_box"]["name"]
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")
    if black_box_name not in SUPPORTED_BLACK_BOXES:
        raise ValueError(f"Unsupported black box: {black_box_name!r}")
    if optimizer_name == "lebo" and config.get("method") not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported LE-BO method: {config.get('method')!r}")
    if config["execution"]["backend"] not in {"local", "srun"}:
        raise ValueError("execution.backend must be 'local' or 'srun'")
    if config["execution"]["max_parallel_runs"] < 1:
        raise ValueError("execution.max_parallel_runs must be positive")


def _method_label(config: dict[str, Any]) -> str:
    """Return the local-enumeration label only for LE-BO tasks."""
    return config.get("method", "") if config["optimizer"]["name"] == "lebo" else ""


def _build_encoder(config: dict[str, Any]):
    import torch

    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
        HydrAMPEncoderDecoder,
    )

    encoder = config["encoder"]
    return HydrAMPEncoderDecoder(
        jacobian_mode=encoder["jacobian_mode"],
        device=config["device"],
        default_condition=torch.tensor(
            encoder["default_condition"], device=config["device"]
        ),
        temp=encoder["temperature"],
        jacobian_eps=encoder["jacobian_eps"],
        field_eps=encoder["field_eps"],
    )


def _build_black_box(config: dict[str, Any]):
    black_box = config["black_box"]
    name = black_box["name"]
    common = black_box.get("common", {})
    if name == "apex":
        from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox

        return APEXBlackBox(
            mic_aggregate=black_box["apex"]["mic_aggregate"],
            mic_bacteria=black_box["apex"]["mic_bacteria"],
            device=config["device"],
            **common,
        )
    if name == "battleamp":
        from pep_compass.optimization.black_box.battleamp_black_box import (
            BattleAMPBlackBox,
        )

        return BattleAMPBlackBox(device=config["device"], **common)
    if name == "hydrophobicity":
        from pep_compass.optimization.black_box.hydrophobicity_black_box import (
            HydrophobicityBlackBox,
        )

        return HydrophobicityBlackBox(
            scale=black_box["hydrophobicity"]["scale"], **common
        )
    if name == "toxipep":
        from pep_compass.optimization.black_box.toxipep_black_box import ToxiPepBlackBox

        return ToxiPepBlackBox(device=config["device"], **common)
    raise AssertionError(f"Validated black box is not implemented: {name}")


def _build_mutation_enumerator(
    config: dict[str, Any],
):
    from pep_compass.local_enumeration.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )

    mutation = config["mutation"]
    return MutationEnumerationInTangentSpace(
        max_len=mutation["max_len"],
        direction_significance_threshold=mutation["direction_significance_threshold"],
        min_number_of_directions=mutation["min_number_of_directions"],
        token_threshold=mutation["token_threshold"],
    )


def _build_walker(
    config: dict[str, Any], encoder_decoder
):
    from pep_compass.local_enumeration.sampling_walker import (
        SecondOrderRiemannianBrownianEfficientSampling,
    )

    walker = config["walker"]
    return SecondOrderRiemannianBrownianEfficientSampling(
        encoder_decoder=encoder_decoder,
        horizontal_threshold=walker["horizontal_threshold"],
        time_step=walker["time_step"],
        max_horizontal_update_norm=walker["max_horizontal_update_norm"],
        vertical_movement=walker["vertical_movement"],
    )


def _build_filter(config: dict[str, Any], encoder_decoder):
    from pep_compass.local_enumeration.mutation.mutation_filters import (
        LamsFilter,
        LpbeboFilter,
        MoveFilter,
        RandomLeBoFilter,
        TandemFilter,
    )

    filter_config = config["filter"]
    method = config["method"]
    common = {"maximum_candidates": filter_config["maximum_candidates"]}
    if method == "lpbebo":
        return LpbeboFilter(
            encoder_decoder,
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if method == "lams":
        return LamsFilter(
            encoder_decoder,
            horizontal_threshold=config["walker"]["horizontal_threshold"],
            similarity_threshold=filter_config["similarity_threshold"],
            **common,
        )
    if method == "tandem":
        return TandemFilter(
            encoder_decoder,
            horizontal_threshold=config["walker"]["horizontal_threshold"],
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if method == "move":
        return MoveFilter(
            encoder_decoder,
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if method in {"random_walker", "random_mutang"}:
        return RandomLeBoFilter(
            mode="walker" if method == "random_walker" else "mutang_random",
            selection_fraction=filter_config["selection_fraction"],
            maximum_positions=filter_config["maximum_positions"],
            residues_per_position=filter_config["residues_per_position"],
            **common,
        )
    raise ValueError(f"LE-BO method {method!r} does not define a filter")


def _build_local_enumerator(
    config: dict[str, Any], encoder_decoder
):
    from pep_compass.local_enumeration.local_enumerator import (
        FilteredMutationLocalEnumerator,
        SamplingFilteredMutationLocalEnumerator,
        SamplingMutationLocalEnumerator,
    )

    mutation_enumerator = _build_mutation_enumerator(config)
    local = config["local_enumeration"]
    if config["method"] == "lpbebo":
        return FilteredMutationLocalEnumerator(
            encoder_decoder=encoder_decoder,
            mutation_generator=mutation_enumerator,
            candidate_filter=_build_filter(config, encoder_decoder),
            max_neighbour_levenstein=local["max_neighbour_levenshtein"],
            device=config["device"],
        )
    common = {
        "encoder_decoder": encoder_decoder,
        "sampling_walker": _build_walker(config, encoder_decoder),
        "mutation_enumerator": mutation_enumerator,
        "walker_trajectories_number": local["walker_trajectories"],
        "time_walk_budget": local["walk_time_budget"],
        "max_neighbour_levenstein": local["max_neighbour_levenshtein"],
        "device": config["device"],
    }
    if config["method"] == "lebo":
        return SamplingMutationLocalEnumerator(**common)
    return SamplingFilteredMutationLocalEnumerator(
        candidate_filter=_build_filter(config, encoder_decoder), **common
    )


def _build_latent_black_box(config: dict[str, Any], discrete_black_box):
    from pep_compass.optimization.black_box.hydramp_black_box_wrapper import (
        HydrAMPBlackBoxWrapper,
    )

    encoder = config["encoder"]
    return HydrAMPBlackBoxWrapper(
        black_box=discrete_black_box,
        device=config["device"],
        jacobian_eps=encoder["jacobian_eps"],
        field_eps=encoder["field_eps"],
    )


def _build_optimizer(config: dict[str, Any], discrete_black_box):
    optimizer_config = config["optimizer"]
    name = optimizer_config["name"]
    if name == "lebo":
        from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
            LocalEnumerationBayesianOptimizer,
        )

        encoder_decoder = _build_encoder(config)
        lebo = optimizer_config["lebo"]
        optimizer = LocalEnumerationBayesianOptimizer(
            black_box=discrete_black_box,
            local_enumerator=_build_local_enumerator(config, encoder_decoder),
            device=config["device"],
            evaluations_per_iteration=lebo["evaluations_per_iteration"],
            levenstain_diversity_threshold=lebo[
                "levenshtein_diversity_threshold"
            ],
            initial_peptides_number=lebo["initial_peptides"],
            turbo_success_tolerance=lebo["turbo_success_tolerance"],
            turbo_failure_tolerance=lebo["turbo_failure_tolerance"],
            turbo_length_init=lebo["turbo_length_init"],
            turbo_length_min=lebo["turbo_length_min"],
            turbo_length_max=lebo["turbo_length_max"],
            acquisition_batch_size=lebo["acquisition_batch_size"],
            standardize=lebo["standardize"],
            best_as_center=lebo["best_as_center"],
            blosum_diversity_matrix=lebo["blosum_diversity_matrix"],
            blosum_diversity_max_score=lebo["blosum_diversity_max_score"],
        )
        return optimizer, discrete_black_box, encoder_decoder
    if name == "random_mutation":
        from pep_compass.optimization.baselines.random_mutation import (
            RandomMutationOptimizer,
        )

        random_config = optimizer_config["random_mutation"]
        optimizer = RandomMutationOptimizer(
            black_box=discrete_black_box,
            esm_model_name=random_config["esm_model_name"],
            esm_ppl_threshold=random_config["esm_ppl_threshold"],
            esm_device=random_config["esm_device"],
            esm_max_resampling_attempts=random_config[
                "esm_max_resampling_attempts"
            ],
        )
        return optimizer, discrete_black_box, None
    if name in {"cmaes", "saasbo"}:
        import torch

        latent_black_box = _build_latent_black_box(config, discrete_black_box)
        if name == "cmaes":
            from pep_compass.optimization.baselines.latent_cmaes import (
                LatentCMAESOptimizer,
            )

            optimizer = LatentCMAESOptimizer(
                black_box=latent_black_box, device=config["device"]
            )
            cmaes = optimizer_config["cmaes"]
            optimizer.population_size = cmaes["population_size"]
            optimizer.initial_sigma = cmaes["initial_sigma"]
            optimizer.constraint_penalty = cmaes["constraint_penalty"]
        else:
            from pep_compass.optimization.baselines.saasbo import SaasboOptimizer

            saasbo = optimizer_config["saasbo"]
            optimizer = SaasboOptimizer(
                black_box=latent_black_box,
                device=torch.device(config["device"]),
                batch_size=saasbo["batch_size"],
                warmup_steps=saasbo["warmup_steps"],
                num_samples=saasbo["num_samples"],
                thinning=saasbo["thinning"],
                dim=saasbo["dimension"],
            )
        return optimizer, latent_black_box, latent_black_box.encoder_decoder
    if name == "lambo2":
        try:
            from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
        except ImportError as error:
            raise RuntimeError(
                "LaMBO2 requires the optional poli-baselines[lambo2] dependencies"
            ) from error
        return LaMBO2, discrete_black_box, None
    raise AssertionError(f"Validated optimizer is not implemented: {name}")


def _run_task(task: dict[str, Any]) -> None:
    import numpy as np

    from pep_compass.optimization.black_box.csv_observer import CSVObserver
    from pep_compass.optimization.black_box.negative_black_box import NegativeBlackBox

    config = task["config"]
    black_box = _build_black_box(config)
    optimizer, observed_black_box, encoder_decoder = _build_optimizer(config, black_box)
    observer = CSVObserver(maximize=observed_black_box.maximize)
    observed_black_box.set_observer(observer)
    observer.initialize_observer(
        observed_black_box.get_black_box_info(),
        {
            "experiment_id": task["experiment_id"],
            "experiment_path": task["output_path"],
        },
        task["seed"],
        encoder_decoder=encoder_decoder,
    )
    logger.info("Starting %s", task["experiment_id"])
    if config["optimizer"]["name"] == "lambo2":
        target = (
            NegativeBlackBox(black_box)
            if config["optimizer"]["lambo2"]["negate_objective"]
            else black_box
        )
        solver = optimizer(
            black_box=target,
            x0=np.array([list(task["sequence"])], dtype="<U1"),
        )
        solver.solve(max_iter=config["optimizer"]["lambo2"]["iterations"])
    else:
        optimizer.optimize(
            evaluation_budget=config["evaluation_budget"],
            starting_point=task["sequence"],
            rng_seed=task["seed"],
        )


def _run_task_file(path: Path) -> None:
    with path.open(encoding="utf-8") as task_file:
        _run_task(json.load(task_file))


def _srun_command(task_path: Path, execution: dict[str, Any]) -> list[str]:
    srun = execution["srun"]
    return [
        srun["command"],
        *srun["arguments"],
        sys.executable,
        str(Path(__file__).resolve()),
        "--task-file",
        str(task_path.resolve()),
    ]


def _run_srun_tasks(task_paths: list[Path], execution: dict[str, Any]) -> None:
    """Submit isolated srun steps while enforcing the configured concurrency."""
    commands = [_srun_command(task_path, execution) for task_path in task_paths]

    def execute(command: list[str]) -> None:
        logger.info("Executing %s", " ".join(command))
        subprocess.run(command, check=True)

    with ThreadPoolExecutor(max_workers=execution["max_parallel_runs"]) as executor:
        futures = [executor.submit(execute, command) for command in commands]
        for future in futures:
            future.result()


def _prepare_tasks(
    config: dict[str, Any],
    sequences: list[dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    variants = _expand_grid(config)
    task_directory = output_root / "tasks"
    task_directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    task_paths = []
    base_seed = config["seed"]
    grid_manifest_path = output_root / "grid_manifest.csv"
    run_manifest_path = output_root / "run_manifest.csv"
    with grid_manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as grid_file, run_manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as run_file:
        grid_writer = csv.DictWriter(
            grid_file, fieldnames=["grid_id", "output_path", "parameters"]
        )
        run_writer = csv.DictWriter(
            run_file,
            fieldnames=[
                "task_id",
                "grid_id",
                "name",
                "sequence",
                "repetition",
                "seed",
                "output_path",
                "task_file",
            ],
        )
        grid_writer.writeheader()
        run_writer.writeheader()
        task_index = 0
        for grid_index, (parameters, variant) in enumerate(variants):
            _validate_config(variant)
            grid_id = f"grid_{grid_index:04d}"
            variant_path = output_root / grid_id
            variant_path.mkdir(parents=True, exist_ok=True)
            variant["output_path"] = str(variant_path)
            with (variant_path / "resolved_config.json").open(
                "w", encoding="utf-8"
            ) as resolved_file:
                json.dump(variant, resolved_file, indent=2)
            grid_writer.writerow(
                {
                    "grid_id": grid_id,
                    "output_path": variant_path,
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
            for peptide in sequences:
                for repetition in range(peptide["repetitions"]):
                    task_id = f"task_{task_index:06d}"
                    seed = base_seed + task_index
                    experiment_id = (
                        f"{variant['optimizer']['name']}_"
                        f"{_method_label(variant) or 'na'}_"
                        f"{peptide['name']}_r{repetition}_{seed}"
                    )
                    task = {
                        "task_id": task_id,
                        "grid_id": grid_id,
                        "name": peptide["name"],
                        "sequence": peptide["sequence"],
                        "repetition": repetition,
                        "seed": seed,
                        "experiment_id": experiment_id,
                        "output_path": str(variant_path),
                        "config": variant,
                    }
                    task_path = task_directory / f"{task_id}.json"
                    with task_path.open("w", encoding="utf-8") as task_file:
                        json.dump(task, task_file, indent=2)
                    tasks.append(task)
                    task_paths.append(task_path)
                    run_writer.writerow(
                        {
                            "task_id": task_id,
                            "grid_id": grid_id,
                            "name": peptide["name"],
                            "sequence": peptide["sequence"],
                            "repetition": repetition,
                            "seed": seed,
                            "output_path": variant_path,
                            "task_file": task_path,
                        }
                    )
                    task_index += 1
    return tasks, task_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured peptide optimization grids locally or with srun."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--task-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--output", type=Path, help="Override the output root.")
    parser.add_argument("--budget", type=int, help="Override evaluations per run.")
    parser.add_argument("--seed", type=int, help="Override the base seed.")
    parser.add_argument("--execution", choices=["local", "srun"])
    parser.add_argument("--max-parallel-runs", type=int)
    parser.add_argument(
        "--srun-argument",
        action="append",
        help="Replace configured srun arguments; repeat once per argument.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and materialize all grid tasks without loading models.",
    )
    return parser.parse_args()


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        "device": args.device,
        "output_path": str(args.output) if args.output is not None else None,
        "evaluation_budget": args.budget,
        "seed": args.seed,
    }
    for path, value in overrides.items():
        if value is not None:
            config[path] = value
            config.get("grid", {}).pop(path, None)
    if args.execution is not None:
        config["execution"]["backend"] = args.execution
    if args.max_parallel_runs is not None:
        config["execution"]["max_parallel_runs"] = args.max_parallel_runs
    if args.srun_argument is not None:
        config["execution"]["srun"]["arguments"] = args.srun_argument


def main() -> None:
    args = _parse_args()
    if args.task_file is not None:
        _run_task_file(args.task_file)
        return
    config = _load_config(args.config)
    _apply_overrides(config, args)
    if config.get("seed") is None:
        config["seed"] = int(time.time())
    output_root = Path(config["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    tasks, task_paths = _prepare_tasks(
        config, _load_sequences(Path(config["input_csv"])), output_root
    )
    logger.info("Prepared %s runs in %s", len(tasks), output_root)
    if args.dry_run:
        return
    if config["execution"]["backend"] == "srun":
        _run_srun_tasks(task_paths, config["execution"])
    else:
        for task in tasks:
            _run_task(task)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
