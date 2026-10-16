"""Public composition, retention, replay and exact serialization contracts."""

from __future__ import annotations

import json
import random
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch
from fixtures.autoencoders import MockAutoencoder

from pep_compass import LocalEnumeration, Loop, Parallel, Pipeline, Step
from pep_compass.data_schemas.candidates import (
    CandidateBatch,
    SharedField,
    TensorField,
)
from pep_compass.optimization.components.filters.base import Filter
from pep_compass.optimization.components.mutation_generators.base import (
    MutationGenerator,
)
from pep_compass.optimization.components.oracles.base import Oracle
from pep_compass.optimization.components.walkers.base import Walker
from pep_compass.optimization.saving import AnalysisReader, DataUnavailableError
from pep_compass.registry.components.catalog import ComponentCatalog, component_catalog
from pep_compass.registry.components.loading import load_components
from pep_compass.registry.factories.registry import Registry


class LengthOracle(Oracle):
    """Evaluate deterministic sequence lengths through the real state contract."""

    def _execute(self, batch, context):
        scores = [float(len(sequence)) for sequence in batch.sequences]
        lineage = {
            name: field.values.tolist()
            for name, field in batch.fields.items()
            if name.startswith("lineage.") and isinstance(field, TensorField)
        }
        context.state.record_observations(
            "length",
            batch.sequences,
            scores,
            batch.latent_origins,
            direction="maximize",
            lineage=lineage,
        )
        return batch.with_field(
            "oracle.length.score", TensorField(torch.tensor(scores))
        ).with_field("oracle.length.direction", SharedField("maximize"))


class FirstFilter(Filter):
    def _execute(self, batch, context):
        return batch.select([0] if len(batch) else [])


class RandomWalker(Walker):
    def _execute(self, batch, context):
        draws = context.rng.integers(0, 3, size=len(batch))
        sequences = [
            sequence + "ACD"[int(draw)]
            for sequence, draw in zip(batch.sequences, draws, strict=True)
        ]
        return CandidateBatch(
            sequences,
            batch.latent_origins + torch.rand_like(batch.latent_origins),
            {
                **batch.fields,
                "walker.adjusted_time_step": TensorField(
                    torch.full((len(batch),), 0.1)
                ),
            },
        )


class PairGenerator(MutationGenerator):
    def _execute(self, batch, context):
        indices = torch.arange(len(batch)).repeat_interleave(2)  # (2 * B,)
        result = batch.repeat_from_parents(indices)
        sequences = [sequence + random.choice("GK") for sequence in result.sequences]
        result = result.with_sequences(sequences)
        context.state.record_generated_candidates(len(result))
        return result


@pytest.fixture
def catalog():
    """Isolate registrations while retaining genuine built-in implementations."""
    load_components()
    catalog = ComponentCatalog()
    for family in component_catalog.names():
        original = component_catalog.family(family).registry
        registry = Registry(original.namespace, expected_type=original.expected_type)
        registry._entries.update(original._entries)
        catalog.register_family(family, registry)
    for family, name, implementation in (
        ("oracle", "test_length", LengthOracle),
        ("filter", "test_first", FirstFilter),
        ("walker", "test_random", RandomWalker),
        ("mutation_generator", "test_pair", PairGenerator),
    ):
        catalog.family(family).registry.register(name)(implementation)
    return catalog


def declaration(catalog, component, strategy, parameters=None):
    return Step.create(
        component=component, strategy=strategy, parameters=parameters, catalog=catalog
    )


def make_pipeline(catalog, steps=()):
    return Pipeline.create(steps=steps, autoencoder=MockAutoencoder(), catalog=catalog)


def local(catalog, iterations=2):
    return LocalEnumeration.create(
        walker=declaration(catalog, "walker", "test_random"),
        mutation_generator=declaration(catalog, "mutation_generator", "test_pair"),
        filters=[declaration(catalog, "filter", "deduplicate")],
        trajectories=2,
        iterations=iterations,
    )


def test_catalog_and_instance_descriptions_are_distinct(catalog):
    pipeline = make_pipeline(catalog)
    node = pipeline.add_step("filter", "deduplicate", name="unique")
    assert any(
        item["strategy"] == "test_length" for item in pipeline.describe(display="raw")
    )
    assert (
        pipeline.describe_component(node_id=node.node_id, display="raw")["node"]["name"]
        == "unique"
    )
    assert (
        pipeline.describe_component(
            component="filter", strategy="deduplicate", display="raw"
        )["parameters"][0]["default"]
        == "sequence"
    )


def test_three_composition_styles_execute_the_same_graph(catalog):
    separate = declaration(catalog, "oracle", "test_length")
    variants = [
        make_pipeline(catalog, [Loop.create(iterations=2, steps=[separate])]),
        make_pipeline(
            catalog,
            [
                Loop.create(
                    iterations=2, steps=[declaration(catalog, "oracle", "test_length")]
                )
            ],
        ),
    ]
    incremental = make_pipeline(catalog)
    with incremental.add_loop(iterations=2) as scope:
        scope.add_step("oracle", "test_length")
    variants.append(incremental)
    archives = [
        pipeline.run(["AA", "ACD"], seed=7).archive.drop(columns="lineage_json")
        for pipeline in variants
    ]
    for archive in archives[1:]:
        pd.testing.assert_frame_equal(archive, archives[0])


def test_base_keeps_evaluations_removed_by_later_filter(catalog):
    pipeline = make_pipeline(
        catalog,
        [
            declaration(catalog, "oracle", "test_length"),
            declaration(catalog, "filter", "test_first"),
        ],
    )
    result = pipeline.run(["AA", "ACD", "ACDEF"], seed=7)
    assert result.archive.sequence.tolist() == ["AA", "ACD", "ACDEF"]
    assert result.summary["terminal_count"] == 1
    assert result.reader.get_best_candidate()["sequence"] == "ACDEF"
    with pytest.raises(DataUnavailableError):
        result.reader.get_final_batch()


def test_atomic_scopes_and_stable_editing(catalog):
    pipeline = make_pipeline(catalog)
    with pytest.raises(RuntimeError, match="abort"):
        with pipeline.add_loop(iterations=2) as loop:
            loop.add_step("filter", "deduplicate")
            raise RuntimeError("abort")
    assert not pipeline.to_schema().steps
    node = pipeline.add_step("filter", "deduplicate")
    snapshot = pipeline.to_schema()
    pipeline.build()
    pipeline.replace_step(node.node_id, declaration(catalog, "filter", "test_first"))
    assert pipeline._compiled is None
    assert pipeline.to_schema().steps[0].node_id == node.node_id
    assert snapshot.steps[0].strategy == "deduplicate"
    pipeline.insert_step(
        declaration(catalog, "oracle", "test_length"), before=node.node_id
    )
    pipeline.remove_step(node.node_id)
    assert len(pipeline.to_schema().steps) == 1


def test_nested_parallel_scopes_commit_one_operator(catalog):
    pipeline = make_pipeline(catalog)
    with pipeline.add_parallel(name="branches") as parallel:
        with parallel.add_branch("left") as branch:
            branch.add_step("filter", "deduplicate")
        with parallel.add_branch("right") as branch:
            with branch.add_loop(iterations=1) as loop:
                loop.add_step("filter", "test_first")
    pipeline.set_runner_settings(storage={"save_terminal": True})
    result = pipeline.run(["AA", "ACD"], seed=7)
    assert result.reader.get_final_batch().sequences == ("AA", "ACD", "AA")


@pytest.mark.parametrize("format", ["json", "text", "mermaid", "svg"])
def test_graph_views_include_identity_and_relations(catalog, format):
    pipeline = make_pipeline(catalog, [local(catalog)])
    rendered = pipeline.to_schema(format)
    assert pipeline.to_schema().steps[0].node_id in rendered
    assert any(
        label in rendered
        for label in ("local_enumeration", "LocalEnumeration", "Local enumeration")
    )
    if format == "svg":
        import xml.etree.ElementTree as ET

        ET.fromstring(rendered)
    if format == "json":
        restored = Pipeline.from_schema(
            rendered, autoencoder=MockAutoencoder(), catalog=catalog
        )
        assert restored.to_schema().to_dict() == pipeline.to_schema().to_dict()


def test_memory_sqlite_and_export_have_the_same_logical_data(catalog, tmp_path):
    pipeline = make_pipeline(catalog, [declaration(catalog, "oracle", "test_length")])
    options = {
        "save_terminal": True,
        "save_lineage": True,
        "save_observation_history": True,
    }
    memory = pipeline.run(["AA", "AA", "ACD"], seed=7, storage=options)
    durable = pipeline.run(
        ["AA", "AA", "ACD"],
        seed=7,
        storage={**options, "backend": "sqlite", "path": str(tmp_path / "durable")},
    )
    pd.testing.assert_frame_equal(memory.archive, durable.archive)
    pd.testing.assert_frame_equal(
        memory.reader.get_history(), durable.reader.get_history()
    )
    memory.save(tmp_path / "exported")
    reopened = AnalysisReader(tmp_path / "exported")
    pd.testing.assert_frame_equal(reopened.get_archive(), memory.archive)
    assert reopened.get_final_batch().sequences == ("AA", "AA", "ACD")
    assert set(path.name for path in (tmp_path / "exported").iterdir()) == {
        "settings.json",
        "results.sqlite",
    }


def test_buffering_flushes_last_iteration_without_sampling(catalog, tmp_path):
    pipeline = make_pipeline(
        catalog,
        [
            Loop.create(
                iterations=3, steps=[declaration(catalog, "oracle", "test_length")]
            )
        ],
    )
    result = pipeline.run(
        ["AA"],
        seed=7,
        storage={
            "backend": "sqlite",
            "path": str(tmp_path / "run"),
            "flush_every": 10,
            "save_observation_history": True,
        },
    )
    assert result.reader.tables["iterations"].count_rows() == 3
    assert len(result.reader.get_history()) == 3
    assert result.summary["oracle_calls"] == 3


def test_multiple_replay_boundaries_restore_rngs_and_run_state(catalog):
    pipeline = make_pipeline(catalog, [local(catalog), local(catalog, iterations=1)])
    result = pipeline.run(["AA"], seed=7, storage={"reconstruction": "trajectories"})
    checkpoints = result.reader._table_frame("checkpoints")
    assert len(checkpoints) == 2
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for identity in checkpoints.checkpoint_id:
        replay = result.reader.reconstruct_local_enumeration(
            int(identity), autoencoder=MockAutoencoder(), catalog=catalog
        )
        assert replay.matches
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_seeded_runs_and_retention_do_not_change_computation(catalog):
    pipeline = make_pipeline(
        catalog, [local(catalog), declaration(catalog, "oracle", "test_length")]
    )
    first = pipeline.run(["AA"], seed=7)
    second = pipeline.run(
        ["AA"],
        seed=7,
        storage={
            "reconstruction": "trajectories",
            "component_data": "full",
            "save_terminal": True,
        },
    )
    pd.testing.assert_frame_equal(first.archive, second.archive)
    assert (
        first.summary["generated_candidates"] == second.summary["generated_candidates"]
    )


def test_replay_restores_context_state_after_static_preparation(catalog):
    class StatefulWalker(RandomWalker):
        def precompute(self, context):
            context.step_state[self.node_id] = 0

        def _execute(self, batch, context):
            context.step_state[self.node_id] += 1
            result = super()._execute(batch, context)
            suffix = "A" * context.step_state[self.node_id]
            return result.with_sequences([s + suffix for s in result.sequences])

    catalog.family("walker").registry.register("test_stateful")(StatefulWalker)
    operation = LocalEnumeration.create(
        walker=declaration(catalog, "walker", "test_stateful"),
        mutation_generator=declaration(catalog, "mutation_generator", "test_pair"),
        iterations=1,
        trajectories=1,
    )
    pipeline = make_pipeline(catalog, [Loop.create(iterations=2, steps=[operation])])
    result = pipeline.run(["AA"], seed=7, storage={"reconstruction": "boundaries"})
    for checkpoint_id in (1, 2):
        replay = result.reader.reconstruct_local_enumeration(
            checkpoint_id, autoencoder=MockAutoencoder(), catalog=catalog
        )
        assert replay.matches


def test_automatic_build_uses_the_effective_run_seed(catalog):
    class RandomizedConstructor(Filter):
        def __init__(self):
            self.values = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )

        def _execute(self, batch, context):
            return batch.with_sequences([s + str(self.values) for s in batch.sequences])

    catalog.family("filter").registry.register("test_random_init")(
        RandomizedConstructor
    )
    sequences = []
    for external_seed in (100, 200):
        random.seed(external_seed)
        np.random.seed(external_seed)
        torch.manual_seed(external_seed)
        pipeline = make_pipeline(catalog)
        pipeline.add_step("filter", "test_random_init")
        result = pipeline.run(["AA"], seed=7, storage={"save_terminal": True})
        sequences.append(result.reader.get_final_batch().sequences)
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize("phase", ["precompute", "prepare_call", "execute"])
def test_failures_are_durable_and_identify_the_phase(catalog, tmp_path, phase):
    class Failure(Filter):
        def precompute(self, context):
            if phase == "precompute":
                raise RuntimeError("expected failure")

        def prepare_call(self, batch, context):
            if phase == "prepare_call":
                raise RuntimeError("expected failure")

        def _execute(self, batch, context):
            raise RuntimeError("expected failure")

    catalog.family("filter").registry.register("test_failure")(Failure)
    pipeline = make_pipeline(catalog, [declaration(catalog, "filter", "test_failure")])
    with pytest.raises(RuntimeError, match="expected failure"):
        pipeline.run(["AA"], storage={"backend": "sqlite", "path": str(tmp_path)})
    reader = AnalysisReader(tmp_path)
    assert reader.summary["status"] == "failed"
    records = reader._table_frame("step_executions")
    assert ((records.phase == phase) & (records.status == "failed")).any()


def test_save_component_preserves_batch_and_named_snapshot(catalog):
    pipeline = make_pipeline(catalog)
    pipeline.add_step("save", "snapshot", parameters={"name": "selected"})
    result = pipeline.run(["AA", "ACD"])
    assert result.reader.get_snapshot("selected")[0].sequences == ("AA", "ACD")
    assert result.archive.empty


def test_settings_and_configuration_tampering_fail_explicitly(catalog, tmp_path):
    pipeline = make_pipeline(catalog)
    with pytest.raises(ValueError, match="requires SQLite"):
        pipeline.set_runner_settings(results={"return_mode": "path"})
    result = pipeline.run(["AA"], storage={"backend": "sqlite", "path": str(tmp_path)})
    result.close()
    settings_path = tmp_path / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings["inputs"] = ["CHANGED"]
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="mismatch"):
        AnalysisReader(tmp_path)


def test_repeated_observations_preserve_first_seen_tie_order(catalog):
    pipeline = make_pipeline(catalog, [declaration(catalog, "oracle", "test_length")])
    result = pipeline.run(["AA", "CC", "AA"])
    assert result.archive.sequence.tolist() == ["AA", "CC"]
    assert (
        result.reader.get_best_candidate()["sequence"]
        == result.summary["best_sequence"]
        == "AA"
    )


def test_base_updates_latest_scores_independently_for_each_objective(catalog):
    class RepeatedOracle(Oracle):
        def _execute(self, batch, context):
            index = context.scope.loop_indices[-1]
            for name, direction in (("up", "maximize"), ("down", "minimize")):
                context.state.record_observations(
                    name,
                    batch.sequences,
                    [float(index)] * len(batch),
                    batch.latent_origins,
                    direction=direction,
                )
            return batch

    catalog.family("oracle").registry.register("test_repeat")(RepeatedOracle)
    pipeline = make_pipeline(
        catalog,
        [
            Loop.create(
                iterations=2, steps=[declaration(catalog, "oracle", "test_repeat")]
            )
        ],
    )
    result = pipeline.run(["AA", "CC"])
    for objective in ("up", "down"):
        archive = result.reader.get_archive(objective=objective)
        assert archive.sequence.tolist() == ["AA", "CC"]
        assert archive.score.tolist() == [1.0, 1.0]
        assert result.reader.get_best_candidate(objective=objective)["sequence"] == "AA"
    assert result.summary["oracle_calls"] == 8


def test_replay_restores_budget_already_consumed_before_boundary(catalog):
    pipeline = make_pipeline(
        catalog, [local(catalog, iterations=1), local(catalog, iterations=2)]
    )
    result = pipeline.run(
        ["AA"],
        seed=7,
        execution={"generated_candidates": 26},
        storage={"reconstruction": "boundaries"},
    )
    replay = result.reader.reconstruct_local_enumeration(
        2, autoencoder=MockAutoencoder(), catalog=catalog
    )
    assert replay.matches
    checkpoint = result.reader.backend.load_payload(
        result.reader.backend.execute(
            "SELECT payload_id FROM checkpoints WHERE checkpoint_id=2"
        ).fetchone()[0]
    )
    assert checkpoint["state"]["generated_candidates"] > 0


def test_missing_registered_input_contract_fails_before_model_initialization(catalog):
    pipeline = make_pipeline(catalog)
    pipeline.add_step(
        "mutation_generator", "mutang", parameters={"maximum_candidates": 2}
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        pipeline.build()


def test_available_description_includes_internal_strategies(catalog):
    pipeline = make_pipeline(catalog)
    description = pipeline.describe_component(
        component="walker", strategy="sorbes", display="raw"
    )
    assert "geometry" in description["nested_strategies"]
    assert description["nested_strategies"]["geometry"][0]["strategy"] == "kappa_stable"


def test_run_cannot_edit_active_graph(catalog):
    pipeline = make_pipeline(catalog)

    class EditingFilter(Filter):
        def _execute(self, batch, context):
            pipeline.add_step("filter", "deduplicate")
            return batch

    catalog.family("filter").registry.register("test_editing")(EditingFilter)
    pipeline.add_step("filter", "test_editing")
    with pytest.raises(RuntimeError, match="cannot be edited"):
        pipeline.run(["AA"])
    assert not pipeline._running


@pytest.mark.parametrize("mode", ["reader", "path"])
def test_public_return_modes_share_one_run_result(catalog, tmp_path, mode):
    pipeline = make_pipeline(catalog, [declaration(catalog, "oracle", "test_length")])
    returned = pipeline.run(
        ["AA"],
        storage={"backend": "sqlite", "path": str(tmp_path)},
        results={"return_mode": mode},
    )
    if mode == "reader":
        assert returned.get_archive().sequence.tolist() == ["AA"]
    else:
        assert returned == tmp_path
        assert AnalysisReader(returned).get_archive().sequence.tolist() == ["AA"]
    assert pipeline.last_result is not None


def test_component_can_select_its_own_display_information(catalog):
    class DescribedFilter(Filter):
        def _execute(self, batch, context):
            return batch

        def display_info(self):
            return {"important_setting": 17}

    catalog.family("filter").registry.register("test_described")(DescribedFilter)
    pipeline = make_pipeline(
        catalog, [declaration(catalog, "filter", "test_described")]
    ).build()
    assert "important_setting" in pipeline.to_schema("text")


def test_finite_oracle_budget_is_reset_between_runs():
    pipeline = Pipeline().add_autoencoder(parameters={"jacobian_mode": "approx"})
    pipeline.add_step("oracle", "hydrophobicity", parameters={"evaluation_budget": 2})
    first = pipeline.run(["AA", "AC"], seed=7)
    second = pipeline.run(["AA", "AC"], seed=7)
    pd.testing.assert_frame_equal(first.archive, second.archive)


@pytest.mark.parametrize("operators", list(product(("loop", "parallel"), repeat=3)))
def test_generated_compositions_match_existing_engine_reference(catalog, operators):
    """Generated nested structures must preserve the established execution map."""
    from pep_compass.optimization.engine import Flow
    from pep_compass.optimization.engine import Loop as EngineLoop
    from pep_compass.optimization.engine import Parallel as EngineParallel
    from pep_compass.optimization.pipeline import PepCompassPipeline

    def assemble(depth):
        if depth == len(operators):
            return declaration(catalog, "walker", "test_random"), RandomWalker()
        left, left_reference = assemble(depth + 1)
        if operators[depth] == "loop":
            return Loop.create(iterations=2, steps=[left]), EngineLoop(
                Flow([left_reference]), 2
            )
        right, right_reference = assemble(depth + 1)
        return Parallel.create(
            branches={"left": [left], "right": [right]}
        ), EngineParallel(
            {"left": Flow([left_reference]), "right": Flow([right_reference])}
        )

    node, reference_root = assemble(0)
    reference = PepCompassPipeline(
        autoencoder=MockAutoencoder(), root=Flow([reference_root, LengthOracle()])
    ).run(["AA", "AC"], seed=17)
    result = make_pipeline(
        catalog, [node, declaration(catalog, "oracle", "test_length")]
    ).run(["AA", "AC"], seed=17, storage={"save_terminal": True})
    assert result.archive.sequence.tolist() == list(reference.candidates.sequences)
    terminal = result.reader.get_final_batch()
    assert terminal.sequences == reference.terminal_candidates.sequences
    assert torch.equal(
        terminal.latent_origins, reference.terminal_candidates.latent_origins
    )


def test_optional_mutation_fields_are_accepted_only_by_capable_scorers(catalog):
    local_node = LocalEnumeration.create(
        walker=declaration(catalog, "walker", "sorbes"),
        mutation_generator=declaration(
            catalog, "mutation_generator", "mutang", {"maximum_candidates": 2}
        ),
        iterations=1,
        include_walk_points=True,
    )
    scoring = declaration(
        catalog,
        "filter",
        "ranked",
        {
            "scoring": {"method": "decoder_likelihood"},
            "selection": {"method": "top_k", "parameters": {"k": 1}},
        },
    )
    make_pipeline(catalog, [local_node, scoring]).build()
    incompatible = declaration(catalog, "filter", "random_mutang")
    with pytest.raises(ValueError, match="does not accept optional"):
        make_pipeline(catalog, [local_node, incompatible]).build()


def test_iteration_duration_and_interval_have_distinct_meaning(catalog):
    pipeline = make_pipeline(
        catalog,
        [
            Loop.create(
                iterations=2, steps=[declaration(catalog, "oracle", "test_length")]
            )
        ],
    )
    result = pipeline.run(["AA"])
    iterations = result.reader._table_frame("iterations")
    assert pd.isna(iterations.iloc[0].interval_seconds)
    assert (
        iterations.iloc[1].interval_seconds >= iterations.iloc[1].duration_seconds >= 0
    )


def test_replay_builds_only_the_requested_fragment(catalog):
    calls = []

    class ObservedOracle(LengthOracle):
        def __init__(self):
            calls.append("constructed")

    catalog.family("oracle").registry.register("test_observed")(ObservedOracle)
    pipeline = make_pipeline(
        catalog, [local(catalog), declaration(catalog, "oracle", "test_observed")]
    )
    result = pipeline.run(["AA"], seed=7, storage={"reconstruction": "boundaries"})
    assert calls == ["constructed"]
    assert result.reader.reconstruct_local_enumeration(
        1, autoencoder=MockAutoencoder(), catalog=catalog
    ).matches
    assert calls == ["constructed"]
