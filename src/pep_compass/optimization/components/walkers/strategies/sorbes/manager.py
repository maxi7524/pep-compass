"""Construction of the fixed SORBES stage set."""

from pep_compass.optimization.components.walkers.strategies.sorbes.boundary.main import MainBoundary
from pep_compass.optimization.components.walkers.strategies.sorbes.directions.active_inactive import ActiveInactiveDirections
from pep_compass.optimization.components.walkers.strategies.sorbes.geometry.kappa_stable import KappaStableGeometry
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.article import ArticlePositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.main import MainPositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.without_acceleration import WithoutAccelerationPositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.scaling.stable_dimension import StableDimensionScaling


class SorbesStrategyManager:
    """Build supported stage implementations without changing stage order."""

    position_updates = {
        "main": MainPositionUpdate,
        "article": ArticlePositionUpdate,
        "without_acceleration": WithoutAccelerationPositionUpdate,
    }

    @classmethod
    def build(cls, autoencoder, parameters):
        """Construct all stages from explicit nested declarations."""
        geometry = parameters.get("geometry", {})
        directions = parameters.get("directions", {})
        position_update = parameters.get("position_update", {})
        geometry_parameters = geometry.get("parameters", {})
        direction_parameters = directions.get("parameters", {})
        update_parameters = position_update.get("parameters", {})
        update_method = position_update.get("method", "main")
        try:
            update_type = cls.position_updates[update_method]
        except KeyError as error:
            raise ValueError(f"Unknown SORBES position update: {update_method}") from error
        return (
            KappaStableGeometry(autoencoder, **geometry_parameters),
            ActiveInactiveDirections(**direction_parameters),
            StableDimensionScaling(),
            update_type(**update_parameters),
            MainBoundary(),
        )
