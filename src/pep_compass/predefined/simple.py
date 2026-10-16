from pep_compass import Pipeline
from pep_compass.autoencoder import HydrAMP
from pep_compass.steps.filters import (
    CandidateSubset,
    Deduplicate,
)
from pep_compass.steps.flow import LocalEnumeration
from pep_compass.steps.mutations import Mutang
from pep_compass.steps.oracles import Hydrophobicity
from pep_compass.steps.walkers import Sorbes

"""Small deterministic local-enumeration example with a scalar oracle."""


def create(*, device="cpu", seed=7, mutation_candidates=4, oracle_calls=3):
    """Return a bounded example without loading model weights.

    :param device: Shared autoencoder device.
    :param seed: Reproducible run seed.
    :param mutation_candidates: Local mutation allocation bound.
    :param oracle_calls: Soft complete-batch evaluation threshold.
    :return: Editable pipeline with one local enumeration and hydrophobicity scores.
    """
    pipeline = Pipeline(
        steps=[
            LocalEnumeration(
                trajectories=1,
                iterations=1,
                walk_time=None,
                include_walk_points=False,
                walker=Sorbes(time_step=0.01),
                mutation_generator=Mutang(maximum_candidates=mutation_candidates),
            ),
            Deduplicate(),
            CandidateSubset(count=3),
            Hydrophobicity(),
        ]
    )
    pipeline.add_autoencoder(HydrAMP(device=device, jacobian_mode="approx"))
    pipeline.set_runner_settings(execution={"seed": seed, "oracle_calls": oracle_calls})
    return pipeline
