"""Registration contract for MIC-to-activity transformations."""

from collections.abc import Callable

import numpy as np

from pep_compass.registry.factories.registry import Registry

ActivityTransform = Callable[[np.ndarray], np.ndarray]
activity_transform_registry: Registry[ActivityTransform] = Registry(
    "activity transform"
)
