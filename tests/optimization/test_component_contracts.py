"""Contract tests applied automatically to registered components."""

import sys

import pep_compass.optimization.components.filters.strategies  # noqa: F401
import pep_compass.optimization.components.mutation_generators.strategies  # noqa: F401
import pep_compass.optimization.components.walkers.strategies  # noqa: F401
import pep_compass.optimization.components.oracles.strategies  # noqa: F401
import pytest

from pep_compass.optimization.components.filters import FilterManager
from pep_compass.optimization.components.mutation_generators import (
    MutationGeneratorManager,
)
from pep_compass.optimization.components.oracles import OracleManager
from pep_compass.optimization.components.walkers import WalkerManager


@pytest.mark.parametrize(
    "registry",
    (
        WalkerManager._registry,
        MutationGeneratorManager._registry,
        FilterManager._registry,
        OracleManager._registry,
    ),
)
def test_registered_component_factories_have_stable_non_empty_names(registry) -> None:
    """Every component family must expose callable factories for validation."""
    assert registry
    assert all(isinstance(name, str) and name for name in registry)
    assert all(callable(factory) for factory in registry.values())


def test_oracle_registration_does_not_import_model_implementations() -> None:
    """Discovering oracle names must not initialize optional model stacks."""
    implementation_modules = {
        "pep_compass.optimization.components.oracles.strategies.apex.oracle",
        "pep_compass.optimization.components.oracles.strategies.battleamp.oracle",
        "pep_compass.optimization.components.oracles.strategies.eipred.oracle",
        "pep_compass.optimization.components.oracles.strategies.hydrophobicity.oracle",
        "pep_compass.optimization.components.oracles.strategies.mbc_attention.oracle",
        "pep_compass.optimization.components.oracles.strategies.toxipep.oracle",
    }

    assert implementation_modules.isdisjoint(sys.modules)
    assert OracleManager.methods() == (
        "apex",
        "battleamp",
        "eipred",
        "hydrophobicity",
        "mbc_attention",
        "toxipep",
    )
