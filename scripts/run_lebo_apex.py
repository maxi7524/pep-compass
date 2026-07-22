"""Run a configured LE-BO-family experiment against the APEX oracle."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import torch

from pep_compass.local_enumeration.local_enumerator import (
    FilteredMutationLocalEnumerator,
    SamplingFilteredMutationLocalEnumerator,
    SamplingMutationLocalEnumerator,
)
from pep_compass.local_enumeration.mutation.mutation_filters import (
    LamsFilter,
    LpbeboFilter,
    MoveFilter,
    RandomLeBoFilter,
    TandemFilter,
)
from pep_compass.local_enumeration.mutation_enumerator import (
    MutationEnumerationInTangentSpace,
)
from pep_compass.local_enumeration.sampling_walker import (
    SecondOrderRiemannianBrownianEfficientSampling,
)
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)
from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
    LocalEnumerationBayesianOptimizer,
)

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = {
    "lebo",
    "lpbebo",
    "lams",
    "tandem",
    "move",
    "random_walker",
    "random_mutang",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Circular config inheritance detected at {path}")
    visited.add(path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    parent = config.pop("extends", None)
    if parent is not None:
        parent_config = _load_config(path.parent / parent, visited)
        config = _deep_merge(parent_config, config)
    input_csv = config.get("input_csv")
    if input_csv is not None:
        config["input_csv"] = str((path.parent / input_csv).resolve())
    return config


def _load_sequences(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"name", "sequence"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain name and sequence columns")
        sequences = []
        names = set()
        for row_number, row in enumerate(reader, start=2):
            name = row["name"].strip()
            sequence = row["sequence"].strip().upper()
            repetitions_text = (row.get("repetitions") or "1").strip()
            if not name or not sequence:
                raise ValueError(f"Empty name or sequence in CSV row {row_number}")
            if name in names:
                raise ValueError(f"Duplicate sequence name in input CSV: {name}")
            repetitions = int(repetitions_text)
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
    grid = config.pop("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("grid must be an object mapping config paths to value lists")
    if not grid:
        return [({}, config)]
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid values for {path} must be a non-empty list")
    expanded = []
    paths = list(grid)
    for values in product(*(grid[path] for path in paths)):
        parameters = dict(zip(paths, values))
        variant = deepcopy(config)
        for path, value in parameters.items():
            _set_nested_value(variant, path, value)
        expanded.append((parameters, variant))
    return expanded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LPBEBO, LAMS, TANDEM, MOVE, random LE-BO or LE-BO on APEX."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--output", type=Path, help="Override the output directory.")
    parser.add_argument("--budget", type=int, help="Override evaluations per run.")
    parser.add_argument("--seed", type=int, help="Base seed for reproducible runs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the CSV and grid without loading models or running optimization.",
    )
    return parser.parse_args()


def _apply_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    grid = config.get("grid", {})

    def override(path: str, value: Any) -> None:
        config[path] = value
        if isinstance(grid, dict):
            grid.pop(path, None)

    if args.device is not None:
        override("device", args.device)
    if args.output is not None:
        override("output_path", str(args.output))
    if args.budget is not None:
        override("evaluation_budget", args.budget)
    if args.seed is not None:
        override("seed", args.seed)
    return config


def _build_encoder(config: dict[str, Any]) -> HydrAMPEncoderDecoder:
    encoder_config = config["encoder"]
    return HydrAMPEncoderDecoder(
        jacobian_mode=encoder_config["jacobian_mode"],
        device=config["device"],
        default_condition=torch.tensor(encoder_config["default_condition"]),
        temp=encoder_config["temperature"],
        jacobian_eps=encoder_config["jacobian_eps"],
        field_eps=encoder_config["field_eps"],
    )


def _build_mutation_enumerator(
    config: dict[str, Any],
) -> MutationEnumerationInTangentSpace:
    mutation_config = config["mutation"]
    return MutationEnumerationInTangentSpace(
        max_len=mutation_config["max_len"],
        direction_significance_threshold=mutation_config[
            "direction_significance_threshold"
        ],
        min_number_of_directions=mutation_config["min_number_of_directions"],
        token_threshold=mutation_config["token_threshold"],
    )


def _build_walker(
    config: dict[str, Any], encoder_decoder: HydrAMPEncoderDecoder
) -> SecondOrderRiemannianBrownianEfficientSampling:
    walker_config = config["walker"]
    return SecondOrderRiemannianBrownianEfficientSampling(
        encoder_decoder=encoder_decoder,
        horizontal_threshold=walker_config["horizontal_threshold"],
        time_step=walker_config["time_step"],
        max_horizontal_update_norm=walker_config["max_horizontal_update_norm"],
        vertical_movement=walker_config["vertical_movement"],
    )


def _build_filter(config: dict[str, Any], encoder_decoder: HydrAMPEncoderDecoder):
    filter_config = config.get("filter", {})
    method = config["method"]
    common = {
        "maximum_candidates": filter_config.get("maximum_candidates", 6000)
    }
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
    raise ValueError(f"Method {method!r} does not define a candidate filter")


def _build_local_enumerator(
    config: dict[str, Any], encoder_decoder: HydrAMPEncoderDecoder
):
    mutation_enumerator = _build_mutation_enumerator(config)
    local_config = config["local_enumeration"]
    method = config["method"]
    if method == "lpbebo":
        return FilteredMutationLocalEnumerator(
            encoder_decoder=encoder_decoder,
            mutation_generator=mutation_enumerator,
            candidate_filter=_build_filter(config, encoder_decoder),
            max_neighbour_levenstein=local_config["max_neighbour_levenshtein"],
            device=config["device"],
        )
    walker = _build_walker(config, encoder_decoder)
    common = {
        "encoder_decoder": encoder_decoder,
        "sampling_walker": walker,
        "mutation_enumerator": mutation_enumerator,
        "walker_trajectories_number": local_config["walker_trajectories"],
        "time_walk_budget": local_config["walk_time_budget"],
        "max_neighbour_levenstein": local_config["max_neighbour_levenshtein"],
        "device": config["device"],
    }
    if method == "lebo":
        return SamplingMutationLocalEnumerator(**common)
    return SamplingFilteredMutationLocalEnumerator(
        candidate_filter=_build_filter(config, encoder_decoder), **common
    )


def _build_optimizer(
    config: dict[str, Any], black_box: APEXBlackBox, local_enumerator
) -> LocalEnumerationBayesianOptimizer:
    optimizer_config = config["optimizer"]
    return LocalEnumerationBayesianOptimizer(
        black_box=black_box,
        local_enumerator=local_enumerator,
        device=config["device"],
        evaluations_per_iteration=optimizer_config["evaluations_per_iteration"],
        levenstain_diversity_threshold=optimizer_config[
            "levenshtein_diversity_threshold"
        ],
        initial_peptides_number=optimizer_config["initial_peptides"],
        turbo_success_tolerance=optimizer_config["turbo_success_tolerance"],
        turbo_failure_tolerance=optimizer_config["turbo_failure_tolerance"],
        turbo_length_init=optimizer_config["turbo_length_init"],
        turbo_length_min=optimizer_config["turbo_length_min"],
        turbo_length_max=optimizer_config["turbo_length_max"],
        acquisition_batch_size=optimizer_config["acquisition_batch_size"],
        standardize=optimizer_config["standardize"],
        best_as_center=optimizer_config["best_as_center"],
        blosum_diversity_matrix=optimizer_config.get("blosum_diversity_matrix"),
        blosum_diversity_max_score=optimizer_config.get(
            "blosum_diversity_max_score"
        ),
    )


def _run_variant(
    config: dict[str, Any], sequences: list[dict[str, Any]], output_path: Path
) -> None:
    black_box = APEXBlackBox(
        mic_aggregate=config["apex"]["mic_aggregate"],
        mic_bacteria=config["apex"]["mic_bacteria"],
        device=config["device"],
    )
    observer = CSVObserver(black_box.maximize)
    black_box.set_observer(observer)
    encoder_decoder = _build_encoder(config)
    optimizer = _build_optimizer(
        config,
        black_box,
        _build_local_enumerator(config, encoder_decoder),
    )

    run_index = 0
    base_seed = config.get("seed")
    for protein in sequences:
        for _ in range(protein["repetitions"]):
            seed = (base_seed if base_seed is not None else int(time.time())) + run_index
            run_index += 1
            experiment_id = (
                f"{config['method']}_{protein['name']}_{seed}_"
                f"{datetime.now():%Y%m%d_%H%M%S}"
            )
            logger.info("Starting %s", experiment_id)
            observer.initialize_observer(
                black_box.get_black_box_info(),
                {
                    "experiment_id": experiment_id,
                    "experiment_path": str(output_path),
                },
                seed,
                encoder_decoder=encoder_decoder,
            )
            optimizer.optimize(
                evaluation_budget=config["evaluation_budget"],
                starting_point=protein["sequence"],
                rng_seed=seed,
            )


def main() -> None:
    args = _parse_args()
    config = _apply_overrides(_load_config(args.config), args)
    if config.get("seed") is None:
        config["seed"] = int(time.time())
    sequences = _load_sequences(Path(config["input_csv"]))
    output_root = Path(config["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    variants = _expand_grid(config)
    manifest_path = output_root / "grid_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["grid_id", "output_path", "parameters"],
        )
        writer.writeheader()
        for index, (parameters, variant) in enumerate(variants):
            if variant.get("method") not in SUPPORTED_METHODS:
                raise ValueError(f"Unsupported method: {variant.get('method')!r}")
            grid_id = f"grid_{index:04d}"
            variant_path = output_root / grid_id
            variant_path.mkdir(parents=True, exist_ok=True)
            variant["output_path"] = str(variant_path)
            with (variant_path / "resolved_config.json").open(
                "w", encoding="utf-8"
            ) as config_file:
                json.dump(variant, config_file, indent=2)
            writer.writerow(
                {
                    "grid_id": grid_id,
                    "output_path": str(variant_path),
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
            logger.info("Prepared %s with parameters %s", grid_id, parameters)
            if not args.dry_run:
                _run_variant(variant, sequences, variant_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
