from pep_compass import Pipeline
from pep_compass.autoencoder import HydrAMP
from pep_compass.steps.filters import (
    Deduplicate,
    Levenshtein,
    Robot,
    TrustRegion,
    TrustRegionUpdate,
)
from pep_compass.steps.flow import LocalEnumeration, Loop
from pep_compass.steps.mutations import Mutang
from pep_compass.steps.oracles import APEX
from pep_compass.steps.walkers import Sorbes
from pep_compass.steps.walkers.strategies.sorbes import KappaStableGeometry

"""Bounded LE-BO composition based on the repository's reference configuration."""


def create(
    *,
    device="cpu",
    seed=1234,
    mutation_candidates=30000,
    oracle_calls=1400,
    iterations=1000,
    trajectories=10,
    walk_time=0.1,
):
    """Return a fresh LE-BO pipeline without initializing models.

    :param device: Autoencoder and acquisition device; oracle remains on CPU.
    :param seed: Run seed.
    :param mutation_candidates: Explicit local allocation cap for practical runs.
    :param oracle_calls: Soft complete-batch evaluation threshold.
    :param iterations: Maximum optimization-loop iterations.
    :param trajectories: Local trajectories per input sequence.
    :param walk_time: Positive accumulated walk time per local trajectory.
    :return: Editable pipeline with reference trajectory and trust-region settings.

    The allocation cap is an explicit operational addition to the historical
    reference. This preset does not claim reproduction of publication results.
    """
    local = LocalEnumeration(
        trajectories=trajectories,
        walk_time=walk_time,
        include_walk_points=False,
        walker=Sorbes(geometry=KappaStableGeometry(kappa=0.01)),
        mutation_generator=Mutang(maximum_candidates=mutation_candidates),
        filters=(
            Levenshtein(
                reference_field="local_enumeration.center_sequence", maximum_distance=4
            ),
        ),
    )
    steps = [
        local,
        Deduplicate(key="sequence"),
        TrustRegion(objective="apex", geometry="sequence", initial_radius=2),
        Robot(
            objective="apex",
            batch_size=3,
            maximize=False,
            diversity_threshold=2,
            acquisition_batch_size=32,
            standardize=False,
            device=device,
        ),
        APEX(
            model="default",
            mic_aggregate="mean",
            mic_bacteria=[1, 2, 3],
            device="cpu",
            evaluation_batch_size=128,
        ),
        TrustRegionUpdate(
            objective="apex",
            initial_radius=2,
            minimum_radius=2,
            maximum_radius=2,
            expand_factor=2.0,
            shrink_factor=0.5,
            success_tolerance=100,
            failure_tolerance=1,
            maximize=False,
        ),
    ]
    pipeline = Pipeline(steps=[Loop(iterations=iterations, steps=steps)])
    pipeline.add_autoencoder(
        HydrAMP(
            device=device, jacobian_mode="approx", jacobian_eps=0.05, field_eps=0.05
        )
    )
    pipeline.set_runner_settings(
        execution={
            "seed": seed,
            "oracle_calls": oracle_calls,
            "generated_candidates": 1000000,
        }
    )
    return pipeline
