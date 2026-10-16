"""Bounded LE-BO composition based on the repository's reference configuration."""

from pep_compass.core.wrapper import LocalEnumeration, Loop, Pipeline, Step
from pep_compass.registry.pipelines.catalog import register


@register("lebo")
def create(*, device="cpu", seed=1234, mutation_candidates=30000, oracle_calls=1400):
    """Return a fresh LE-BO pipeline without initializing models.

    :param device: Autoencoder and acquisition device; oracle remains on CPU.
    :param seed: Run seed.
    :param mutation_candidates: Explicit local allocation cap for practical runs.
    :param oracle_calls: Soft complete-batch evaluation threshold.
    :return: Editable pipeline with reference trajectory and trust-region settings.

    The allocation cap is an explicit operational addition to the historical
    reference. This preset does not claim reproduction of publication results.
    """
    local = LocalEnumeration.create(
        trajectories=10,
        walk_time=0.1,
        include_walk_points=False,
        walker=Step.create(
            component="walker",
            strategy="sorbes",
            parameters={
                "geometry": {"method": "kappa_stable", "parameters": {"kappa": 0.01}},
                "position_update": {
                    "method": "main",
                    "parameters": {"epsilon": 0.1, "delta_max": 0.5},
                },
            },
        ),
        mutation_generator=Step.create(
            component="mutation_generator",
            strategy="mutang",
            parameters={"maximum_candidates": mutation_candidates},
        ),
        filters=(
            Step.create(
                component="filter",
                strategy="levenshtein",
                parameters={
                    "reference_field": "local_enumeration.center_sequence",
                    "maximum_distance": 4,
                },
            ),
        ),
    )
    steps = [
        local,
        Step.create(
            component="filter", strategy="deduplicate", parameters={"key": "sequence"}
        ),
        Step.create(
            component="filter",
            strategy="trust_region",
            parameters={
                "objective": "apex",
                "geometry": "sequence",
                "initial_radius": 2,
            },
        ),
        Step.create(
            component="filter",
            strategy="robot",
            parameters={
                "objective": "apex",
                "batch_size": 3,
                "maximize": False,
                "diversity_threshold": 2,
                "acquisition_batch_size": 32,
                "standardize": False,
                "device": device,
            },
        ),
        Step.create(
            component="oracle",
            strategy="apex",
            parameters={
                "model": "default",
                "mic_aggregate": "mean",
                "mic_bacteria": [1, 2, 3],
                "device": "cpu",
                "evaluation_batch_size": 128,
            },
        ),
        Step.create(
            component="filter",
            strategy="trust_region_update",
            parameters={
                "objective": "apex",
                "initial_radius": 2,
                "minimum_radius": 2,
                "maximum_radius": 2,
                "expand_factor": 2.0,
                "shrink_factor": 0.5,
                "success_tolerance": 100,
                "failure_tolerance": 1,
                "maximize": False,
            },
        ),
    ]
    pipeline = Pipeline.create(
        steps=[Loop.create(iterations=1000, steps=steps, name="optimization")]
    )
    pipeline.add_autoencoder(
        device=device,
        parameters={"jacobian_mode": "approx", "jacobian_eps": 0.05, "field_eps": 0.05},
    )
    pipeline.set_runner_settings(
        execution={
            "seed": seed,
            "oracle_calls": oracle_calls,
            "generated_candidates": 1000000,
        }
    )
    return pipeline
