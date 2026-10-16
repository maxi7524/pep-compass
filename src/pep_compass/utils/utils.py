import time


def set_seed(seed: int):
    """Seed all supported generators through the backend RNG contract.

    :param seed: Non-negative integer seed.
    """
    from pep_compass.utils.randomness import seed_generators

    seed_generators(seed)


class Timer:
    def __init__(self):
        self.times = {}
        self._name = None
        self._start = None

    def __call__(self, name):
        """Allow usage like: with timer("task"): ..."""
        self._name = name
        return self

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self._start
        self.times[self._name] = elapsed

    def __str__(self):
        return (
            "{\n\t"
            + "\n\t".join(
                f"{name}: {elapsed:.4f}s" for name, elapsed in self.times.items()
            )
            + "\n}"
        )

    def reset(self):
        self.times = {}
        self._name = None
        self._start = None
