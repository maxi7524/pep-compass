"""Built-in lazily constructed oracle strategies."""

from functools import partial
from importlib import import_module
from typing import Any

from pep_compass.optimization.components.oracles.configuration import (
    ApexConfig,
    DeviceModelOracleConfig,
    EipredConfig,
    HydrophobicityConfig,
    canonical_parameters,
)
from pep_compass.optimization.components.oracles.strategies.black_box import (
    BlackBoxOracle,
)
from pep_compass.registry.components.families.oracles import OracleManager

# Load model descriptors explicitly, without importing inference implementations.
import_module("pep_compass.registry.models.oracles")

_COMMON = {
    "batch_size",
    "parallelize",
    "num_workers",
    "evaluation_budget",
    "force_isolation",
    "evaluation_batch_size",
}


def validate_model(method, parameters):
    """Validate named model selection without importing model implementations."""
    from pep_compass.registry.models.oracles import (
        has_oracle_model_provider,
        validate_oracle_model,
    )

    if has_oracle_model_provider(method):
        validate_oracle_model(method, parameters.get("model", "default"))


def oracle_fields(method, parameters):
    """Declare fields emitted by the common oracle adapter."""
    return (
        f"oracle.{method}.score",
        f"oracle.{method}.direction",
        f"oracle.{method}.name",
    )


def _black_box_oracle(
    module_name: str,
    class_name: str,
    method: str,
    parameters: dict[str, Any],
) -> BlackBoxOracle:
    """Construct one black-box adapter without eagerly importing its model."""
    adapter_batch_size = parameters.pop("evaluation_batch_size", None)
    if parameters.get("evaluation_budget") is None:
        parameters.pop("evaluation_budget", None)
    module = import_module(module_name)
    black_box_type = getattr(module, class_name)
    return BlackBoxOracle(
        black_box_type(**parameters),
        field_name=f"oracle.{method}.score",
        batch_size=adapter_batch_size,
    )


@OracleManager.register(
    "apex",
    model_provider="oracle.apex",
    config_type=ApexConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "apex"),
    output_fields=partial(oracle_fields, "apex"),
    accepted_parameters=_COMMON
    | {"mic_aggregate", "mic_bacteria", "model", "device", "models_directory"},
)
def build_apex(**parameters: Any) -> BlackBoxOracle:
    """Build the APEX oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.apex.oracle",
        "APEXBlackBox",
        "apex",
        parameters,
    )


@OracleManager.register(
    "apex_original",
    model_provider="oracle.apex",
    config_type=ApexConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "apex_original"),
    output_fields=partial(oracle_fields, "apex_original"),
    accepted_parameters=_COMMON
    | {"mic_aggregate", "mic_bacteria", "model", "device", "models_directory"},
)
def build_apex_original(**parameters: Any) -> BlackBoxOracle:
    """Build the preserved pre-refactor APEX oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.apex_original.oracle",
        "APEXBlackBox",
        "apex_original",
        parameters,
    )


@OracleManager.register(
    "battleamp",
    model_provider="oracle.battleamp",
    config_type=DeviceModelOracleConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "battleamp"),
    output_fields=partial(oracle_fields, "battleamp"),
    accepted_parameters=_COMMON | {"device", "model", "models_directory"},
)
def build_battleamp(**parameters: Any) -> BlackBoxOracle:
    """Build the BattleAMP oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.battleamp.oracle",
        "BattleAMPBlackBox",
        "battleamp",
        parameters,
    )


@OracleManager.register(
    "eipred",
    model_provider="oracle.eipred",
    config_type=EipredConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "eipred"),
    output_fields=partial(oracle_fields, "eipred"),
    accepted_parameters=_COMMON | {"mic_aggregate", "model", "models_directory"},
)
def build_eipred(**parameters: Any) -> BlackBoxOracle:
    """Build the EIPred oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.eipred.oracle",
        "EIPredBlackBox",
        "eipred",
        parameters,
    )


@OracleManager.register(
    "hydrophobicity",
    config_type=HydrophobicityConfig,
    normalizer=canonical_parameters,
    output_fields=partial(oracle_fields, "hydrophobicity"),
    accepted_parameters=_COMMON | {"scale"},
)
def build_hydrophobicity(**parameters: Any) -> BlackBoxOracle:
    """Build the hydrophobicity oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.hydrophobicity.oracle",
        "HydrophobicityBlackBox",
        "hydrophobicity",
        parameters,
    )


@OracleManager.register(
    "mbc_attention",
    model_provider="oracle.mbc_attention",
    config_type=DeviceModelOracleConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "mbc_attention"),
    output_fields=partial(oracle_fields, "mbc_attention"),
    accepted_parameters=_COMMON | {"device", "model", "models_directory"},
)
def build_mbc_attention(**parameters: Any) -> BlackBoxOracle:
    """Build the MBC-Attention oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.mbc_attention.oracle",
        "MBCAttentionBlackBox",
        "mbc_attention",
        parameters,
    )


@OracleManager.register(
    "toxipep",
    model_provider="oracle.toxipep",
    config_type=DeviceModelOracleConfig,
    normalizer=canonical_parameters,
    validator=partial(validate_model, "toxipep"),
    output_fields=partial(oracle_fields, "toxipep"),
    accepted_parameters=_COMMON | {"device", "model", "models_directory"},
)
def build_toxipep(**parameters: Any) -> BlackBoxOracle:
    """Build the ToxiPep oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.toxipep.oracle",
        "ToxiPepBlackBox",
        "toxipep",
        parameters,
    )
