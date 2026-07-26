import numpy as np

from pep_compass.local_enumeration.mutation.mutation_filters import RandomLeBoFilter


def test_random_mutang_temperature_controls_nucleus_size() -> None:
    mutations = {0: list(range(1, 11)), 1: list(range(10, 20))}

    np.random.seed(7)
    cold_filter = RandomLeBoFilter(
        mode="mutang_random", selection_fraction=0.6, temperature=0.1
    )
    cold_count = len(cold_filter.filter_candidates("AA", mutations))

    np.random.seed(7)
    hot_filter = RandomLeBoFilter(
        mode="mutang_random", selection_fraction=0.6, temperature=10.0
    )
    hot_count = len(hot_filter.filter_candidates("AA", mutations))

    assert cold_count < hot_count <= hot_filter.last_generated_count
