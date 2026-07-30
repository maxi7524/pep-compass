"""Build the composable optimization runtime from configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pep_compass.filters.manager import FilterManager
from pep_compass.optimization.flow import Flow, Loop, Parallel, build_merger
from pep_compass.optimization.runner import OptimizationRunner
from pep_compass.optimization.state import OptimizationLimits
from pep_compass.optimization.step import Step


class PepCompassCore:
    """Developer-facing composition root for configured optimization steps."""

    def __init__(self, encoder_decoder, *, tracker=None) -> None:
        self.encoder_decoder = encoder_decoder
        self.tracker = tracker
        self._ensure_builtin_strategies()

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, tracker=None
    ) -> "PepCompassCore":
        """Construct the composition root and encoder-decoder from configuration."""
        encoder_config = config.get("encoder_decoder")
        if not isinstance(encoder_config, Mapping):
            raise ValueError("encoder_decoder configuration is required.")
        method = encoder_config.get("method")
        if method != "hydramp":
            raise ValueError(f"Unknown encoder-decoder method: {method}")
        import torch

        from pep_compass.core.encoder_decoder.strategies.hydramp.adapter import (
            HydrAMPEncoderDecoder,
        )

        parameters = dict(encoder_config.get("parameters", {}))
        device = encoder_config.get("device", "cpu")
        if "default_condition" in parameters:
            parameters["default_condition"] = torch.as_tensor(
                parameters["default_condition"],
                device=device,
            )
        encoder_decoder = HydrAMPEncoderDecoder(device=device, **parameters)
        return cls(encoder_decoder, tracker=tracker)

    @staticmethod
    def _ensure_builtin_strategies() -> None:
        import pep_compass.filters.strategies  # noqa: F401
        import pep_compass.mutation_generators.strategies  # noqa: F401
        import pep_compass.walkers.strategies  # noqa: F401

    def build_runner(
        self, config: Mapping[str, Any], *, tracker=None
    ) -> OptimizationRunner:
        """Build an optimization runner from the ``optimization.steps`` tree."""
        optimization = config.get("optimization", config)
        raw_steps = optimization.get("steps")
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
            raise ValueError("optimization.steps must be a sequence.")
        root = self._build_flow(raw_steps)
        limits_config = optimization.get("limits", {})
        limits = OptimizationLimits(
            oracle_calls=limits_config.get("oracle_calls"),
            generated_candidates=limits_config.get("generated_candidates"),
        )
        return OptimizationRunner(
            self.encoder_decoder,
            root,
            self.tracker if tracker is None else tracker,
            limits,
        )

    def _build_flow(self, configurations: Sequence[Mapping[str, Any]]) -> Flow:
        return Flow(
            [self._build_step(configuration) for configuration in configurations]
        )

    def _build_step(self, configuration: Mapping[str, Any]) -> Step:
        if len(configuration) != 1:
            raise ValueError(
                "Every configured step must have exactly one operation key."
            )
        operation, settings = next(iter(configuration.items()))
        if operation == "loop":
            return self._build_loop(settings)
        if operation == "parallel":
            return self._build_parallel(settings)
        if operation == "walker":
            return self._build_walker(settings)
        if operation == "mutation_generator":
            return self._build_mutation_generator(settings)
        if operation == "filter":
            return self._build_filter(settings)
        if operation == "oracle":
            return self._build_oracle(settings)
        raise ValueError(f"Unknown optimization operation: {operation}")

    def _build_loop(self, settings: Mapping[str, Any]) -> Loop:
        steps = settings.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError("loop.steps must be a sequence.")
        return Loop(self._build_flow(steps), int(settings["iterations"]))

    def _build_parallel(self, settings: Mapping[str, Any]) -> Parallel:
        branch_configs = settings.get("branches")
        if not isinstance(branch_configs, Sequence) or isinstance(
            branch_configs, (str, bytes)
        ):
            raise ValueError("parallel.branches must be a sequence.")
        branches: dict[str, Step] = {}
        for index, branch in enumerate(branch_configs):
            name = str(branch.get("name", f"branch_{index}"))
            if name in branches:
                raise ValueError(f"Parallel branch name is duplicated: {name}")
            branches[name] = self._build_flow(branch["steps"])
        return Parallel(
            branches,
            execution=settings.get("execution", "sequential"),
            merger=build_merger(settings.get("merge", "concatenate")),
        )

    def _build_walker(self, settings: Mapping[str, Any]) -> Step:
        from pep_compass.walkers.strategies.subriemannian import (
            SecondOrderRiemannianBrownianEfficientSampling,
        )
        from pep_compass.walkers.manager import WalkerManager

        method, parameters = self._method_and_parameters(settings)
        if method != "sorbes":
            return WalkerManager.build(method, **parameters)
        sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
            encoder_decoder=self.encoder_decoder,
            **parameters,
        )
        return WalkerManager.build(method, sampling_walker=sampling_walker)

    def _build_mutation_generator(self, settings: Mapping[str, Any]) -> Step:
        from pep_compass.mutation_generators.strategies.tangent_space import (
            MutationEnumerationInTangentSpace,
        )
        from pep_compass.mutation_generators.manager import MutationGeneratorManager

        method, parameters = self._method_and_parameters(settings)
        if method != "mutang":
            return MutationGeneratorManager.build(method, **parameters)
        enumerator = MutationEnumerationInTangentSpace(**parameters)
        return MutationGeneratorManager.build(method, mutation_enumerator=enumerator)

    def _build_filter(self, settings: Mapping[str, Any]) -> Step:
        method, parameters = self._method_and_parameters(settings)
        if method in {
            "lpbebo",
            "lams",
            "tandem",
            "move",
            "random_walker",
            "random_mutang",
        }:
            return self._build_legacy_mutation_filter(method, parameters)
        return FilterManager.build(method, **parameters)

    def _build_legacy_mutation_filter(
        self,
        method: str,
        parameters: dict[str, Any],
    ) -> Step:
        from pep_compass.filters.strategies.mutation_choice import MutationChoiceFilter
        from pep_compass.filters.strategies.mutation_filters import (
            LamsFilter,
            LpbeboFilter,
            MoveFilter,
            RandomLeBoFilter,
            TandemFilter,
        )

        if method == "lpbebo":
            implementation = LpbeboFilter(self.encoder_decoder, **parameters)
        elif method == "lams":
            implementation = LamsFilter(self.encoder_decoder, **parameters)
        elif method == "tandem":
            implementation = TandemFilter(self.encoder_decoder, **parameters)
        elif method == "move":
            implementation = MoveFilter(self.encoder_decoder, **parameters)
        else:
            mode = "walker" if method == "random_walker" else "mutang_random"
            implementation = RandomLeBoFilter(mode=mode, **parameters)
        return MutationChoiceFilter(implementation)

    def _build_oracle(self, settings: Mapping[str, Any]) -> Step:
        from pep_compass.oracles.strategies.black_box import BlackBoxOracle

        method, parameters = self._method_and_parameters(settings)
        batch_size = parameters.pop("batch_size", None)
        black_box = self._build_black_box(method, parameters)
        return BlackBoxOracle(
            black_box,
            field_name=f"oracle.{method}.score",
            batch_size=batch_size,
        )

    def _build_black_box(self, method: str, parameters: dict[str, Any]):
        if method == "apex":
            from pep_compass.oracles.strategies.apex.oracle import APEXBlackBox

            return APEXBlackBox(**parameters)
        if method == "battleamp":
            from pep_compass.oracles.strategies.battleamp.oracle import (
                BattleAMPBlackBox,
            )

            return BattleAMPBlackBox(**parameters)
        if method == "hydrophobicity":
            from pep_compass.oracles.strategies.hydrophobicity.oracle import (
                HydrophobicityBlackBox,
            )

            return HydrophobicityBlackBox(**parameters)
        if method == "toxipep":
            from pep_compass.oracles.strategies.toxipep.oracle import (
                ToxiPepBlackBox,
            )

            return ToxiPepBlackBox(**parameters)
        raise ValueError(f"Unknown oracle method: {method}")

    @staticmethod
    def _method_and_parameters(
        settings: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        method = settings.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("A configured strategy requires a non-empty method.")
        parameters = settings.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("Strategy parameters must be a mapping.")
        return method, dict(parameters)
