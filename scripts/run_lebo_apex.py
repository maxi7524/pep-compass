"""Run a configured LE-BO-family experiment against the APEX oracle."""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from datetime import datetime
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
    benchmark_path = config.pop("benchmark", None)
    if benchmark_path is not None:
        with (path.parent / benchmark_path).open(encoding="utf-8") as benchmark_file:
            config["proteins"] = json.load(benchmark_file)["proteins"]
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LPBEBO, LAMS, TANDEM, MOVE, random LE-BO or LE-BO on APEX."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument("--output", type=Path, help="Override the output directory.")
    parser.add_argument("--budget", type=int, help="Override evaluations per run.")
    parser.add_argument("--seed", type=int, help="Base seed for reproducible runs.")
    return parser.parse_args()


def _apply_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if args.device is not None:
        config["device"] = args.device
    if args.output is not None:
        config["output_path"] = str(args.output)
    if args.budget is not None:
        config["evaluation_budget"] = args.budget
    if args.seed is not None:
        config["seed"] = args.seed
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


def main() -> None:
    args = _parse_args()
    config = _apply_overrides(_load_config(args.config), args)
    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

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
    for name, protein in config["proteins"].items():
        for _ in range(protein["repetitions"]):
            seed = (base_seed if base_seed is not None else int(time.time())) + run_index
            run_index += 1
            experiment_id = (
                f"{config['method']}_{name}_{seed}_{datetime.now():%Y%m%d_%H%M%S}"
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
