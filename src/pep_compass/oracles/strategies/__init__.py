"""Built-in lazily constructed oracle strategies."""

from importlib import import_module
from typing import Any

from pep_compass.oracles.manager import OracleManager
from pep_compass.oracles.strategies.black_box import BlackBoxOracle


def _black_box_oracle(
    module_name: str,
    class_name: str,
    method: str,
    parameters: dict[str, Any],
) -> BlackBoxOracle:
    """Construct one black-box adapter without eagerly importing its model."""
    adapter_batch_size = parameters.pop("evaluation_batch_size", None)
    module = import_module(module_name)
    black_box_type = getattr(module, class_name)
    return BlackBoxOracle(
        black_box_type(**parameters),
        field_name=f"oracle.{method}.score",
        batch_size=adapter_batch_size,
    )


@OracleManager.register("apex")
def build_apex(**parameters: Any) -> BlackBoxOracle:
    """Build the APEX oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.apex.oracle",
        "APEXBlackBox",
        "apex",
        parameters,
    )


@OracleManager.register("battleamp")
def build_battleamp(**parameters: Any) -> BlackBoxOracle:
    """Build the BattleAMP oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.battleamp.oracle",
        "BattleAMPBlackBox",
        "battleamp",
        parameters,
    )


@OracleManager.register("eipred")
def build_eipred(**parameters: Any) -> BlackBoxOracle:
    """Build the EIPred oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.eipred.oracle",
        "EIPredBlackBox",
        "eipred",
        parameters,
    )


@OracleManager.register("hydrophobicity")
def build_hydrophobicity(**parameters: Any) -> BlackBoxOracle:
    """Build the hydrophobicity oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.hydrophobicity.oracle",
        "HydrophobicityBlackBox",
        "hydrophobicity",
        parameters,
    )


@OracleManager.register("mbc_attention")
def build_mbc_attention(**parameters: Any) -> BlackBoxOracle:
    """Build the MBC-Attention oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.mbc_attention.oracle",
        "MBCAttentionBlackBox",
        "mbc_attention",
        parameters,
    )


@OracleManager.register("toxipep")
def build_toxipep(**parameters: Any) -> BlackBoxOracle:
    """Build the ToxiPep oracle strategy."""
    return _black_box_oracle(
        "pep_compass.oracles.strategies.toxipep.oracle",
        "ToxiPepBlackBox",
        "toxipep",
        parameters,
    )


__all__ = ["BlackBoxOracle"]
