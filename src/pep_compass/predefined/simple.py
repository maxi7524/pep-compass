"""Small deterministic local-enumeration example with a scalar oracle."""

from pep_compass.core.wrapper import LocalEnumeration, Pipeline, Step
from pep_compass.registry.pipelines.catalog import register


@register("simple")
def create(*, device="cpu", seed=7, mutation_candidates=4, oracle_calls=3):
    """Return a bounded example without loading model weights.

    :param device: Shared autoencoder device.
    :param seed: Reproducible run seed.
    :param mutation_candidates: Local mutation allocation bound.
    :param oracle_calls: Soft complete-batch evaluation threshold.
    :return: Editable pipeline with one local enumeration and hydrophobicity scores.
    """
    pipeline = Pipeline.create(
        steps=[
            LocalEnumeration.create(
                trajectories=1,
                iterations=1,
                walk_time=None,
                include_walk_points=False,
                walker=Step.create(
                    component="walker",
                    strategy="sorbes",
                    parameters={"time_step": 0.01},
                ),
                mutation_generator=Step.create(
                    component="mutation_generator",
                    strategy="mutang",
                    parameters={"maximum_candidates": mutation_candidates},
                ),
            ),
            Step.create(component="filter", strategy="deduplicate"),
            Step.create(
                component="filter", strategy="candidate_subset", parameters={"count": 3}
            ),
            Step.create(component="oracle", strategy="hydrophobicity"),
        ]
    )
    pipeline.add_autoencoder(device=device, parameters={"jacobian_mode": "approx"})
    pipeline.set_runner_settings(execution={"seed": seed, "oracle_calls": oracle_calls})
    return pipeline
