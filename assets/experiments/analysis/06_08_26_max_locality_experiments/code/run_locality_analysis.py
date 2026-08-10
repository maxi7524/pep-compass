"""Run the complete locality analysis and export tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# User configuration
RESULTS_PATH = Path("results/locality")
OUTPUT_PATH = Path("results/outputs/locality")

TRAJECTORY_COUNTS = [1, 2, 5, 10, 20, 30, 50, 75, 100]
LEVENSTEIN_RADII = list(range(1, 9))
BOOTSTRAP_REPETITIONS = 300
CONFIDENCE = 0.95
RANDOM_SEED = 972
CHUNK_SIZE = 100_000
SKETCH_PRECISION = 12
LATENT_CANDIDATE_SAMPLE_SIZE = 50_000
USE_CACHE = True

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pep_compass.analysis import ExperimentAnalysis
from pep_compass.analysis.reader import ExperimentReader


def _save_result(name: str, result: object) -> None:
    """Persist one numerical result and its reproducibility metadata.

    :param name: Stable output stem.
    :param result: :class:`AnalysisResult` returned by the analysis facade.
    """
    result.data.to_csv(OUTPUT_PATH / f"{name}.csv", index=False)
    with (OUTPUT_PATH / f"{name}.metadata.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(result.metadata, output, indent=2, default=str)


def _save_figure(name: str, axes: object) -> None:
    """Save a visualizer result and release its Matplotlib resources."""
    axes_item = axes.flat[0] if hasattr(axes, "flat") else axes
    figure = axes_item.figure
    figure.savefig(OUTPUT_PATH / f"{name}.png", bbox_inches="tight")
    figure.savefig(OUTPUT_PATH / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Execute numerical locality analyses and export every artifact."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    reader = ExperimentReader(RESULTS_PATH)
    selection = reader.select()
    analysis = ExperimentAnalysis(
        selection,
        plot_theme="paper",
        use_cache=USE_CACHE,
    )

    run_catalog = pd.DataFrame([run.metadata for run in reader.runs])
    run_catalog.to_csv(OUTPUT_PATH / "run_catalog.csv", index=False)

    trajectory_result = analysis.locality.trajectory_saturation(
        trajectory_counts=TRAJECTORY_COUNTS,
        levenshtein_radii=LEVENSTEIN_RADII,
        repetitions=BOOTSTRAP_REPETITIONS,
        confidence=CONFIDENCE,
        random_seed=RANDOM_SEED,
        accepted_only=False,
        sketch_precision=SKETCH_PRECISION,
        chunk_size=CHUNK_SIZE,
    )
    # _save_result("trajectory_saturation", trajectory_result)
    _save_figure(
        "trajectory_saturation",
        analysis.visualize.locality.trajectory_saturation(trajectory_result),
    )
    _save_figure(
        "trajectory_marginal_yield",
        analysis.visualize.locality.trajectory_marginal(trajectory_result),
    )

    step_result = analysis.locality.step_saturation(
        levenshtein_radii=LEVENSTEIN_RADII,
        accepted_only=False,
        sketch_precision=SKETCH_PRECISION,
        chunk_size=CHUNK_SIZE,
    )
    # _save_result("step_saturation", step_result)
    _save_figure(
        "step_saturation",
        analysis.visualize.locality.step_saturation(step_result),
    )
    _save_figure(
        "step_marginal_yield",
        analysis.visualize.locality.step_marginal(step_result),
    )

    latent_result = analysis.locality.latent_locality(
        candidate_sample_size=LATENT_CANDIDATE_SAMPLE_SIZE,
        candidate_scan_chunk_size=5_000,
        chunk_size=5_000,
        quantile_sample_size=20_000,
        diagnostic_sample_size=2_000,
        random_seed=RANDOM_SEED,
    )
    # _save_result("latent_locality", latent_result)
    _save_figure(
        "latent_locality",
        analysis.visualize.locality.latent_locality(latent_result),
    )

    method_result = analysis.locality.method_comparison(
        levenshtein_radii=LEVENSTEIN_RADII,
        sketch_precision=SKETCH_PRECISION,
        chunk_size=CHUNK_SIZE,
    )
    # _save_result("method_comparison", method_result)
    # _save_figure(
    #     "method_comparison_yield",
    #     analysis.visualize.locality.method_comparison(method_result),
    # )
    # _save_figure(
    #     "method_comparison_retention",
    #     analysis.visualize.locality.method_retention(method_result),
    # )

    # parameter_result = analysis.locality.parameter_selection()
    # _save_result("parameter_selection", parameter_result)
    # _save_figure(
    #     "parameter_selection",
    #     analysis.visualize.locality.parameter_selection(parameter_result),
    # )
    reader.cached_analyses().to_csv(
        OUTPUT_PATH / "analysis_cache_catalog.csv", index=False
    )
    print(f"Analysis artifacts saved to {OUTPUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()
