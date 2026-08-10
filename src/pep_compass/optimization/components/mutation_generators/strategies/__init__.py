"""Built-in mutation generator strategies."""

from pep_compass.optimization.components.mutation_generators.strategies.mutang import Mutang, MutangGenerator
from pep_compass.optimization.components.mutation_generators.strategies.mutang.combination.baseline import BaselineCombination
from pep_compass.optimization.components.mutation_generators.strategies.mutang.direction_selection.baseline import BaselineDirectionSelection
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.kappa_stable import KappaStableGeometry
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.shared import SharedGeometry
from pep_compass.optimization.components.mutation_generators.strategies.mutang.mutation_selection.threshold import ThresholdSelection
from pep_compass.optimization.components.mutation_generators.strategies.mutang.scoring.max_absolute_loading import MaxAbsoluteLoading
from pep_compass.optimization.components.mutation_generators.manager import MutationGeneratorManager
from pep_compass.utils.strategy_factory import parameter_contract


@MutationGeneratorManager.register("mutang")
def build_mutang(autoencoder, maximum_candidates=None, strategies=None, max_len=25,
                 direction_significance_threshold=1e-3, min_number_of_directions=5,
                 token_threshold=0.1, alphabet=None):
    """Build baseline MUTANG with optional geometry selection."""
    alphabet = alphabet or list(" ACDEFGHIKLMNPQRSTVWY")
    configuration = strategies or {}
    geometry_config = configuration.get("geometry", {"method": "shared"})
    geometry = SharedGeometry() if geometry_config.get("method") == "shared" else KappaStableGeometry(**geometry_config.get("parameters", {}))
    mutang = Mutang(
        geometry,
        BaselineDirectionSelection(direction_significance_threshold, min_number_of_directions),
        MaxAbsoluteLoading(max_len, len(alphabet)),
        ThresholdSelection(token_threshold),
        BaselineCombination(alphabet, maximum_candidates),
        autoencoder,
    )
    return MutangGenerator(mutang)

__all__ = ["MutangGenerator"]
