"""Contract tests for the composable optimization engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import pytest

from pep_compass.filters.strategies.selectors.deduplicate import DeduplicateFilter
from pep_compass.filters.strategies.decision_models.esm_filter import ESMDecisionFilter
from pep_compass.filters.base import Filter
from pep_compass.filters.manager import FilterManager
from pep_compass.experiments.composable import ComposableExperiment
from pep_compass.experiments.input import (
    load_input_sequences,
    materialize_input_tasks,
)
from pep_compass.experiments.variants import materialize_variants
from pep_compass.experiments.plan import materialize_execution_plan
from pep_compass.experiments.backends import (
    execute_subprocess_plan,
    write_slurm_array_script,
)
from pep_compass.core.builder import PepCompassCore
from pep_compass.core.validation import validate_configuration
from pep_compass.optimization.batch import (
    CandidateBatch,
    OptionalField,
    TensorField,
)
from pep_compass.optimization.context import OptimizationContext
from pep_compass.optimization.flow import Flow, Loop, Parallel
from pep_compass.optimization.runner import OptimizationRunner
from pep_compass.optimization.state import OptimizationLimits
from pep_compass.optimization.step import Step
from pep_compass.optimization.tracking import CSVStepTracker, InMemoryStepTracker
from pep_compass.mutation_generators.strategies.mutang import MutangGenerator
from pep_compass.mutation_generators.base import MutationGenerator
from pep_compass.mutation_generators.manager import MutationGeneratorManager
from pep_compass.oracles.strategies.black_box import BlackBoxOracle
from pep_compass.oracles.manager import OracleManager
from pep_compass.walkers.base import Walker
from pep_compass.walkers.manager import WalkerManager


class _SuffixStep(Step):
    def __init__(self, suffix: str, field_name: str | None = None) -> None:
        self.suffix = suffix
        self.field_name = field_name

    @property
    def name(self) -> str:
        return f"suffix_{self.suffix}"

    def _execute(self, batch, context):
        result = batch.with_sequences(
            [f"{sequence}{self.suffix}" for sequence in batch.sequences]
        )
        if self.field_name is not None:
            result = result.with_field(
                self.field_name,
                TensorField(torch.ones(len(batch), device=batch.latent_origins.device)),
            )
        return result


class _EncoderDecoder:
    def encode_peptides(self, sequences):
        return torch.arange(len(sequences) * 2, dtype=torch.float32).reshape(-1, 2)


class _MutationEnumerator:
    def get_mutations_from_s_u(self, singular_values, left_vectors):
        return {0: [1]}

    def mutate_peptide(self, sequence, mutations):
        return [sequence, f"X{sequence[1:]}"]


class _ESMScorer:
    def pll(self, sequence):
        return {"A": -3.0, "B": -1.0, "C": -2.0}[sequence]


class _ConditionalExperimentRunner:
    def __init__(self, calls, fail_sequence=None):
        self.calls = calls
        self.fail_sequence = fail_sequence

    def run(self, sequences, *, seed=None):
        self.calls.append((tuple(sequences), seed))
        if sequences[0] == self.fail_sequence:
            raise RuntimeError("mock run failure")
        return OptimizationRunner(_EncoderDecoder(), Flow([])).run(sequences, seed=seed)


class _ConditionalExperimentCore:
    def __init__(self, fail_sequence=None):
        self.calls = []
        self.fail_sequence = fail_sequence

    def build_runner(self, config, *, tracker=None):
        return _ConditionalExperimentRunner(self.calls, self.fail_sequence)


def _batch() -> CandidateBatch:
    return CandidateBatch(
        ["A", "B"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_batch_selection_keeps_every_column_aligned() -> None:
    batch = _batch().with_field("score", TensorField(torch.tensor([0.1, 0.9])))

    selected = batch.select(torch.tensor([1]))

    assert selected.sequences == ("B",)
    assert selected.latent_origins.tolist() == [[3.0, 4.0]]
    assert isinstance(selected.fields["score"], TensorField)
    assert selected.fields["score"].values.tolist() == pytest.approx([0.9])


def test_parallel_concatenates_outputs_and_marks_missing_fields() -> None:
    parallel = Parallel(
        {
            "geometry": _SuffixStep("G", "geometry.score"),
            "random": _SuffixStep("R"),
        }
    )

    result = parallel(_batch(), OptimizationContext(_EncoderDecoder()))

    assert result.sequences == ("AG", "BG", "AR", "BR")
    field = result.fields["geometry.score"]
    assert isinstance(field, OptionalField)
    assert field.valid.tolist() == [True, True, False, False]


def test_concurrent_parallel_has_deterministic_branch_order() -> None:
    parallel = Parallel(
        {"first": _SuffixStep("1"), "second": _SuffixStep("2")},
        execution="concurrent",
    )

    result = parallel(_batch(), OptimizationContext(_EncoderDecoder(), seed=7))

    assert result.sequences == ("A1", "B1", "A2", "B2")


def test_loop_tracks_nested_iteration_indices() -> None:
    tracker = InMemoryStepTracker()
    loop = Loop(Flow([_SuffixStep("X")]), iterations=2)

    result = loop(_batch(), OptimizationContext(_EncoderDecoder(), tracker=tracker))

    assert result.sequences == ("AXX", "BXX")
    suffix_records = [
        record for record in tracker.records if record.step_name == "suffix_X"
    ]
    assert [record.loop_indices for record in suffix_records] == [(0,), (1,)]


def test_tracking_depth_disables_deeper_steps_without_changing_results() -> None:
    tracker = InMemoryStepTracker(max_depth=1)
    flow = Flow([_SuffixStep("X")])

    result = flow(_batch(), OptimizationContext(_EncoderDecoder(), tracker=tracker))

    assert result.sequences == ("AX", "BX")
    assert [record.step_name for record in tracker.records] == ["Flow"]


def test_csv_tracking_respects_depth_and_persists_loop_indices(tmp_path) -> None:
    tracker = CSVStepTracker(tmp_path, level="normal", max_depth=3)
    runner = OptimizationRunner(
        _EncoderDecoder(),
        Loop(Flow([_SuffixStep("X")]), iterations=2),
        tracker,
    )

    runner.run(["A"])

    steps = (tmp_path / "steps.csv").read_text(encoding="utf-8")
    assert "iteration[0]" in steps
    assert "iteration[1]" in steps
    assert "suffix_X" not in steps


def test_short_csv_tracking_only_records_oracle_candidates(tmp_path) -> None:
    tracker = CSVStepTracker(tmp_path, level="short")
    root = Flow(
        [
            _SuffixStep("X"),
            BlackBoxOracle(
                lambda sequences: [[1.0]],
                field_name="oracle.test.score",
            ),
        ]
    )

    OptimizationRunner(_EncoderDecoder(), root, tracker).run(["A"])

    steps = (tmp_path / "steps.csv").read_text(encoding="utf-8")
    candidates = (tmp_path / "candidates.csv").read_text(encoding="utf-8")
    assert "BlackBoxOracle" in steps
    assert "suffix_X" not in steps
    assert "AX" in candidates


def test_all_csv_tracking_serializes_only_each_candidate_field_value(tmp_path) -> None:
    tracker = CSVStepTracker(tmp_path, level="all", store_fields=True)
    root = _SuffixStep("X", field_name="candidate.score")

    OptimizationRunner(_EncoderDecoder(), root, tracker).run(["A", "B"])

    with (tmp_path / "candidates.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [json.loads(row["fields"])["candidate.score"] for row in rows] == [
        1.0,
        1.0,
    ]


def test_deduplication_is_explicit_and_preserves_distinct_latents_by_default() -> None:
    batch = CandidateBatch(
        ["A", "A"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    context = OptimizationContext(_EncoderDecoder())

    by_sequence = DeduplicateFilter("sequence")(batch, context)
    by_pair = DeduplicateFilter("sequence_and_latent")(batch, context)

    assert len(by_sequence) == 1
    assert len(by_pair) == 2


def test_esm_decision_filter_preserves_columns_and_selects_stable_top_k() -> None:
    batch = CandidateBatch(
        ["A", "B", "C"],
        torch.tensor([[1.0], [2.0], [3.0]]),
    )

    result = ESMDecisionFilter(top_k=2, scorer=_ESMScorer())(
        batch,
        OptimizationContext(_EncoderDecoder()),
    )

    assert result.sequences == ("B", "C")
    assert result.latent_origins.tolist() == [[2.0], [3.0]]
    score = result.fields["filter.esm_plausibility.score"]
    assert isinstance(score, TensorField)
    assert score.values.tolist() == [-1.0, -2.0]


def test_runner_supports_experiment_without_oracle() -> None:
    runner = OptimizationRunner(_EncoderDecoder(), Flow([_SuffixStep("X")]))

    result = runner.run(["A"], seed=11)

    assert result.candidates.sequences == ("AX",)
    assert result.best_candidate is None
    assert result.best_score is None
    assert result.objective_name is None


def test_inline_mock_sequences_are_loaded_without_transformation() -> None:
    mock_sequences = ["AAAA", "CCCC", "ACDE"]

    result = load_input_sequences({"sequences": mock_sequences})

    assert result == mock_sequences


def test_csv_input_expands_repetitions_relative_to_configuration(tmp_path) -> None:
    input_path = tmp_path / "peptides.csv"
    input_path.write_text(
        "name,sequence,repetitions\nfirst,AAAA,2\nsecond,CCCC,1\n",
        encoding="utf-8",
    )

    result = load_input_sequences(
        {"csv": {"path": "peptides.csv"}},
        base_directory=tmp_path,
    )

    assert result == ["AAAA", "AAAA", "CCCC"]


def test_repetitions_materialize_independent_tasks_in_source_order(tmp_path) -> None:
    input_path = tmp_path / "peptides.csv"
    input_path.write_text(
        "sequence,repetitions\nAAAA,2\nCCCC,1\n",
        encoding="utf-8",
    )

    tasks = materialize_input_tasks(
        {"csv": {"path": "peptides.csv"}},
        base_directory=tmp_path,
    )

    assert [task.sequence for task in tasks] == ["AAAA", "AAAA", "CCCC"]
    assert [task.repetition for task in tasks] == [0, 1, 0]
    assert [task.run_id for task in tasks] == [
        "run_00000",
        "run_00001",
        "run_00002",
    ]


def test_composable_experiment_isolates_repetitions_and_writes_manifest(
    tmp_path,
) -> None:
    config = {
        "experiment": {
            "seed": 17,
            "input": {"sequences": ["AAAA", "CCCC"], "repetitions": 2},
            "output": {"directory": "results"},
        },
        "optimization": {"steps": []},
    }

    result = ComposableExperiment(
        config,
        PepCompassCore(_EncoderDecoder()),
        config_directory=tmp_path,
    ).run()

    assert [run.seed for run in result.runs] == [17, 18, 19, 20]
    assert [run.result.candidates.sequences for run in result.runs] == [
        ("AAAA",),
        ("AAAA",),
        ("CCCC",),
        ("CCCC",),
    ]
    assert (tmp_path / "results" / "run_manifest.csv").exists()
    assert (tmp_path / "results" / "runs" / "run_00003" / "result.json").exists()


def test_grid_materializes_cartesian_optimization_variants() -> None:
    config = {
        "experiment": {
            "grid": {
                "optimization.limits.oracle_calls": [10, 20],
                "optimization.limits.generated_candidates": [100, 200],
            }
        },
        "optimization": {
            "limits": {"oracle_calls": None, "generated_candidates": None},
            "steps": [],
        },
    }

    variants = materialize_variants(config)

    assert len(variants) == 4
    assert [variant.values for variant in variants] == [
        {
            "optimization.limits.oracle_calls": 10,
            "optimization.limits.generated_candidates": 100,
        },
        {
            "optimization.limits.oracle_calls": 10,
            "optimization.limits.generated_candidates": 200,
        },
        {
            "optimization.limits.oracle_calls": 20,
            "optimization.limits.generated_candidates": 100,
        },
        {
            "optimization.limits.oracle_calls": 20,
            "optimization.limits.generated_candidates": 200,
        },
    ]


def test_execution_plan_assigns_stable_global_indices_and_seeds() -> None:
    config = {
        "experiment": {
            "seed": 10,
            "input": {"sequences": ["A", "B"]},
            "grid": {"optimization.limits.oracle_calls": [1, 2]},
        },
        "optimization": {"limits": {"oracle_calls": None}, "steps": []},
    }

    plan = materialize_execution_plan(config)

    assert [entry.index for entry in plan] == [0, 1, 2, 3]
    assert [entry.seed for entry in plan] == [10, 11, 12, 13]
    assert [entry.task.sequence for entry in plan] == ["A", "B", "A", "B"]


def test_subprocess_backend_executes_one_isolated_worker_per_plan_entry(tmp_path) -> None:
    config = {
        "experiment": {"input": {"sequences": ["A", "B"]}},
        "optimization": {"steps": []},
    }
    plan = materialize_execution_plan(config)
    calls = []

    def command_runner(command, *, check):
        calls.append((command, check))

    execute_subprocess_plan(
        plan,
        tmp_path / "config.yaml",
        max_workers=2,
        command_runner=command_runner,
    )

    assert [call[0][call[0].index("--run-index") + 1] for call in calls] == ["0", "1"]
    assert all(call[1] for call in calls)


def test_slurm_backend_writes_array_script_without_submitting(tmp_path) -> None:
    config = {
        "experiment": {"input": {"sequences": ["A", "B"]}},
        "optimization": {"steps": []},
    }
    script = write_slurm_array_script(
        materialize_execution_plan(config),
        tmp_path / "config with spaces.yaml",
        tmp_path / "run.slurm",
        settings={"job_name": "pep-validation", "time": "01:00:00"},
        resume=True,
    )

    content = script.read_text(encoding="utf-8")
    assert content.splitlines()[1] == "#SBATCH --array=0,1"
    assert "#SBATCH --job-name=pep-validation" in content
    assert '"$SLURM_ARRAY_TASK_ID"' in content
    assert "--resume" in content


def test_grid_and_repetitions_create_isolated_variant_run_directories(tmp_path) -> None:
    config = {
        "experiment": {
            "seed": 5,
            "input": {"sequences": ["AAAA"], "repetitions": 2},
            "output": {"directory": "results"},
            "grid": {"optimization.limits.generated_candidates": [10, 20]},
        },
        "optimization": {
            "limits": {"generated_candidates": None},
            "steps": [],
        },
    }

    result = ComposableExperiment(
        config,
        PepCompassCore(_EncoderDecoder()),
        config_directory=tmp_path,
    ).run()

    assert [run.seed for run in result.runs] == [5, 6, 7, 8]
    assert [run.variant.variant_id for run in result.runs] == [
        "variant_00000",
        "variant_00000",
        "variant_00001",
        "variant_00001",
    ]
    assert (
        tmp_path
        / "results"
        / "variants"
        / "variant_00001"
        / "runs"
        / "run_00001"
        / "result.json"
    ).exists()


def test_experiment_continues_after_failed_run_and_persists_status(tmp_path) -> None:
    config = {
        "experiment": {
            "seed": 4,
            "input": {"sequences": ["A", "B", "C"]},
            "output": {"directory": "results"},
        },
        "optimization": {"steps": []},
    }
    core = _ConditionalExperimentCore(fail_sequence="B")

    result = ComposableExperiment(
        config,
        core,
        config_directory=tmp_path,
        on_error="continue",
    ).run()

    assert [run.status for run in result.runs] == ["completed", "failed", "completed"]
    failed = json.loads(
        (tmp_path / "results" / "runs" / "run_00001" / "result.json").read_text()
    )
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: mock run failure"


def test_experiment_resume_skips_completed_runs_without_calling_runner(tmp_path) -> None:
    config = {
        "experiment": {
            "input": {"sequences": ["A", "B"]},
            "output": {"directory": "results"},
        },
        "optimization": {"steps": []},
    }
    first_core = _ConditionalExperimentCore()
    ComposableExperiment(config, first_core, config_directory=tmp_path).run()
    resumed_core = _ConditionalExperimentCore(fail_sequence="A")

    resumed = ComposableExperiment(
        config,
        resumed_core,
        config_directory=tmp_path,
        resume=True,
    ).run()

    assert resumed_core.calls == []
    assert [run.status for run in resumed.runs] == ["skipped", "skipped"]


def test_configuration_validation_rejects_unknown_strategy_before_building() -> None:
    config = {
        "encoder_decoder": {"method": "hydramp"},
        "optimization": {
            "steps": [{"oracle": {"method": "missing", "parameters": {}}}]
        },
    }

    with pytest.raises(ValueError, match="Unknown oracle method"):
        validate_configuration(config)


def test_configuration_validation_accepts_nested_registered_tree() -> None:
    config = {
        "encoder_decoder": {"method": "hydramp"},
        "optimization": {
            "limits": {"oracle_calls": None, "generated_candidates": 100},
            "steps": [
                {
                    "loop": {
                        "iterations": 2,
                        "steps": [
                            {
                                "parallel": {
                                    "execution": "concurrent",
                                    "merge": "concatenate",
                                    "branches": [
                                        {
                                            "name": "left",
                                            "steps": [
                                                {
                                                    "filter": {
                                                        "method": "deduplicate",
                                                        "parameters": {"key": "sequence"},
                                                    }
                                                }
                                            ],
                                        },
                                        {"name": "right", "steps": []},
                                    ],
                                }
                            }
                        ],
                    }
                }
            ],
        },
    }

    validate_configuration(config)


def test_reference_lebo_configuration_materializes_complete_loop() -> None:
    from pep_compass.core.configuration import load_configuration

    repository_root = Path(__file__).resolve().parents[1]
    config = load_configuration(
        repository_root / "configs" / "optimization" / "reference_lebo.yaml"
    )

    validate_configuration(config)
    loop = config["optimization"]["steps"][0]["loop"]
    operations = [next(iter(step)) for step in loop["steps"]]
    assert operations == [
        "walker",
        "mutation_generator",
        "filter",
        "filter",
        "filter",
        "oracle",
    ]


def test_core_executes_complete_mock_strategy_tree_across_loop_iterations() -> None:
    class MockWalker(Walker):
        def _execute(self, batch, context):
            return CandidateBatch(
                [f"{sequence}W" for sequence in batch.sequences],
                batch.latent_origins + 1,
                batch.fields,
            )

    class MockGenerator(MutationGenerator):
        def _execute(self, batch, context):
            parents = torch.arange(len(batch)).repeat_interleave(2)
            expanded = batch.repeat_from_parents(parents)
            sequences = [
                f"{sequence}{suffix}"
                for sequence in batch.sequences
                for suffix in ("0", "1")
            ]
            context.state.record_generated_candidates(len(sequences))
            return expanded.with_sequences(sequences)

    class MockSelector(Filter):
        def _execute(self, batch, context):
            return batch.select(
                [index for index, sequence in enumerate(batch.sequences) if sequence.endswith("1")]
            )

    previous = {
        "walker": WalkerManager._registry.get("mock_tree"),
        "generator": MutationGeneratorManager._registry.get("mock_tree"),
        "filter": FilterManager._registry.get("mock_tree"),
        "oracle": OracleManager._registry.get("mock_tree"),
    }
    WalkerManager._registry["mock_tree"] = MockWalker
    MutationGeneratorManager._registry["mock_tree"] = MockGenerator
    FilterManager._registry["mock_tree"] = MockSelector
    OracleManager._registry["mock_tree"] = lambda: BlackBoxOracle(
        lambda sequences: [[float(len(sequence))] for sequence in sequences],
        field_name="oracle.mock_tree.score",
    )
    tracker = InMemoryStepTracker()
    config = {
        "optimization": {
            "limits": {"oracle_calls": 10, "generated_candidates": 100},
            "steps": [
                {
                    "loop": {
                        "iterations": 2,
                        "steps": [
                            {"walker": {"method": "mock_tree"}},
                            {"mutation_generator": {"method": "mock_tree"}},
                            {"filter": {"method": "mock_tree"}},
                            {"oracle": {"method": "mock_tree"}},
                        ],
                    }
                }
            ],
        }
    }
    try:
        result = PepCompassCore(_EncoderDecoder(), tracker=tracker).build_runner(config).run(["A"])
    finally:
        registries = (
            (WalkerManager._registry, "walker"),
            (MutationGeneratorManager._registry, "generator"),
            (FilterManager._registry, "filter"),
            (OracleManager._registry, "oracle"),
        )
        for registry, key in registries:
            if previous[key] is None:
                registry.pop("mock_tree", None)
            else:
                registry["mock_tree"] = previous[key]

    assert result.candidates.sequences == ("AW1W1",)
    assert result.candidates.latent_origins.tolist() == [[2.0, 3.0]]
    assert result.best_score == 5.0
    oracle_records = [record for record in tracker.records if record.step_name == "BlackBoxOracle"]
    assert [record.loop_indices for record in oracle_records] == [(0,), (1,)]


def test_core_builds_nested_loop_and_parallel_without_oracle() -> None:
    config = {
        "optimization": {
            "steps": [
                {
                    "parallel": {
                        "execution": "concurrent",
                        "merge": "concatenate",
                        "branches": [
                            {
                                "name": "left",
                                "steps": [
                                    {
                                        "filter": {
                                            "method": "deduplicate",
                                            "parameters": {"key": "sequence"},
                                        }
                                    }
                                ],
                            },
                            {
                                "name": "right",
                                "steps": [
                                    {
                                        "loop": {
                                            "iterations": 2,
                                            "steps": [
                                                {
                                                    "filter": {
                                                        "method": "deduplicate",
                                                        "parameters": {
                                                            "key": "sequence_and_latent"
                                                        },
                                                    }
                                                }
                                            ],
                                        }
                                    }
                                ],
                            },
                        ],
                    }
                }
            ]
        }
    }

    result = PepCompassCore(_EncoderDecoder()).build_runner(config).run(["A", "A"])

    assert result.candidates.sequences == ("A", "A", "A")


def test_mutang_preserves_parent_latent_origin_for_every_product_candidate() -> None:
    batch = _batch()
    batch = batch.with_field(
        "walker.singular_values",
        TensorField(torch.ones((2, 1))),
    )
    batch = batch.with_field(
        "walker.left_vectors",
        TensorField(torch.ones((2, 1, 1))),
    )

    result = MutangGenerator(_MutationEnumerator())(
        batch,
        OptimizationContext(_EncoderDecoder()),
    )

    assert result.sequences == ("A", "X", "B", "X")
    assert result.latent_origins.tolist() == [
        [1.0, 2.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [3.0, 4.0],
    ]


def test_oracle_adds_scores_without_replacing_candidates_or_latents() -> None:
    def black_box(sequences):
        return [[len(sequence)] for sequence in sequences]

    batch = _batch()
    result = BlackBoxOracle(black_box, field_name="oracle.test.score")(
        batch,
        OptimizationContext(_EncoderDecoder()),
    )

    assert result.sequences == batch.sequences
    assert result.latent_origins.data_ptr() == batch.latent_origins.data_ptr()
    scores = result.fields["oracle.test.score"]
    assert isinstance(scores, TensorField)
    assert scores.values.tolist() == [1, 1]


def test_runner_summarizes_oracle_but_not_filter_scores() -> None:
    def black_box(sequences):
        return [[2.0], [1.0]]

    runner = OptimizationRunner(
        _EncoderDecoder(),
        BlackBoxOracle(black_box, field_name="oracle.test.score"),
    )

    result = runner.run(["AA", "B"])

    assert result.best_candidate is not None
    assert result.best_candidate.sequence == "B"
    assert result.best_score == 1.0
    assert result.objective_name == "test"
    assert result.objective_direction == "minimize"


def test_oracle_budget_stops_loop_and_limits_evaluated_batch() -> None:
    calls: list[list[str]] = []

    def black_box(sequences):
        calls.append(list(sequences))
        return [[1.0] for _ in sequences]

    runner = OptimizationRunner(
        _EncoderDecoder(),
        Loop(
            BlackBoxOracle(black_box, field_name="oracle.test.score"),
            iterations=10,
        ),
        limits=OptimizationLimits(oracle_calls=3),
    )

    result = runner.run(["A", "B"])

    assert calls == [["A", "B"], ["A"]]
    assert result.candidates.sequences == ("A",)


@pytest.mark.parametrize(
    ("registry", "base"),
    [
        (WalkerManager._registry, Walker),
        (MutationGeneratorManager._registry, MutationGenerator),
        (FilterManager._registry, Filter),
    ],
)
def test_every_registered_component_implements_its_universal_contract(
    registry,
    base,
) -> None:
    assert registry
    assert all(isinstance(name, str) and name for name in registry)
    assert all(callable(factory) for factory in registry.values())


def test_every_bundled_oracle_is_registered_without_loading_its_model() -> None:
    import pep_compass.oracles.strategies  # noqa: F401

    assert OracleManager.methods() == (
        "apex",
        "battleamp",
        "eipred",
        "hydrophobicity",
        "mbc_attention",
        "toxipep",
    )
    assert all(callable(factory) for factory in OracleManager._registry.values())
