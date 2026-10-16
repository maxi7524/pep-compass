"""Public catalog/detail presentation, independent of model construction."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pep_compass import Pipeline
from pep_compass.core.wrapper import descriptions
from pep_compass.registry.components.catalog import ComponentCatalog
from pep_compass.registry.factories.registry import Registry


def test_catalog_modes_keep_raw_records_and_return_a_real_dataframe(capsys):
    pipeline = Pipeline()
    raw = pipeline.describe(display="raw")
    frame = pipeline.describe(display="pd")
    assert isinstance(raw, list)
    assert isinstance(frame, pd.DataFrame)
    assert list(zip(frame.component, frame.strategy)) == [
        (v["component"], v["strategy"]) for v in raw
    ]
    assert "internal_stages" in frame
    assert pipeline.describe() is None
    assert "hydrophobicity" in capsys.readouterr().out


def test_component_detail_preserves_internal_stages_and_model_choices():
    pipeline = Pipeline()
    raw = pipeline.describe_component("walker", "sorbes", display="raw")
    assert "geometry" in raw["nested_strategies"]
    detail = pipeline.describe_component("walker", "sorbes", display="pd")
    assert detail.section.str.startswith("internal.geometry.").any()
    assert "time_step" in detail.name.tolist()
    assert "models" in set(
        pipeline.describe_component("autoencoder", "hydramp", display="pd").section
    )


def test_describe_component_uses_the_instance_catalog():
    catalog = ComponentCatalog()
    registry = Registry("custom")

    @registry.register("chosen")
    def factory(value=7):
        """An isolated custom component."""
        return value

    catalog.register_family("custom", registry)
    pipeline = Pipeline(catalog=catalog)
    assert (
        pipeline.describe_component("custom", "chosen", display="raw")["parameters"][0][
            "default"
        ]
        == 7
    )


def test_placed_component_description_keeps_identity_and_resolved_values():
    pipeline = Pipeline()
    node = pipeline.add_component("filter", "deduplicate")
    raw = pipeline.describe_component(node_id=node.node_id, display="raw")
    assert raw["node"]["node_id"] == node.node_id
    frame = pipeline.describe_component(node_id=node.node_id, display="pd")
    assert frame.loc[frame.name == "node_id", "value"].iloc[0] == node.node_id
    assert (
        frame.loc[
            (frame.section == "parameters") & (frame.name == "key"), "value"
        ].iloc[0]
        == "sequence"
    )


@pytest.mark.parametrize(
    "method", ["describe", "describe_component", "list_components"]
)
def test_invalid_display_mode_fails_explicitly(method):
    pipeline = Pipeline()
    parameters = (
        {}
        if method in {"describe", "list_components"}
        else {"component": "filter", "strategy": "deduplicate"}
    )
    with pytest.raises(ValueError, match="display must"):
        getattr(pipeline, method)(**parameters, display="unknown")


def test_notebook_display_is_automatic_escaped_and_does_not_return_duplicate_output(
    monkeypatch,
):
    import IPython.display

    catalog = ComponentCatalog()
    registry = Registry("custom")

    @registry.register("<script>alert(1)</script>")
    def factory(value="<img src=x onerror=alert(1)>"):
        """Unsafe <b>markup</b> is description text, never executable HTML."""
        return value

    catalog.register_family("custom", registry)
    pipeline = Pipeline(catalog=catalog)
    captured = []
    monkeypatch.setattr(descriptions, "_notebook_available", lambda: True)
    monkeypatch.setattr(IPython.display, "display", captured.append)
    assert pipeline.describe_component("custom", "<script>alert(1)</script>") is None
    assert len(captured) == 1
    assert "&lt;script&gt;" in captured[0].data
    assert "&lt;img" in captured[0].data
    assert "<script>" not in captured[0].data
    assert "<img" not in captured[0].data


def test_notebook_detection_distinguishes_a_kernel_from_plain_python(monkeypatch):
    import IPython

    monkeypatch.setattr(IPython, "get_ipython", lambda: None)
    assert not descriptions._notebook_available()
    monkeypatch.setattr(
        IPython, "get_ipython", lambda: SimpleNamespace(kernel=object())
    )
    assert descriptions._notebook_available()


def test_catalog_hides_parameter_and_model_counts():
    assert list(Pipeline().describe(display="pd").columns) == [
        "component",
        "strategy",
        "description",
        "internal_stages",
    ]


def test_sphinx_fields_are_parameter_help_and_technical_contracts_are_opt_in():
    catalog = ComponentCatalog()
    registry = Registry("fixture")

    @registry.register(
        "sphinx",
        services={"dependency"},
        input_fields=("latent",),
        optional_inputs=("score",),
    )
    def component(count=2, *, dependency=None):
        """Select the configured candidate count.

        :param count: Maximum number of candidates.
            The complete selected batch is returned.
        :type count: int
        :param dependency: Injected shared object.
        :return: Selected batch.
        """
        return count

    catalog.register_family("fixture", registry)
    pipeline = Pipeline(catalog=catalog)
    frame = pipeline.describe_component("fixture", "sphinx", display="pd")
    prose = frame.loc[frame.name == "description", "value"].iloc[0]
    assert prose == "Select the configured candidate count."
    help_text = frame.loc[frame.name == "count", "description"].iloc[0]
    assert (
        help_text
        == "Maximum number of candidates. The complete selected batch is returned."
    )
    assert not {"services", "input_fields", "output_fields", "optional_inputs"} & set(
        frame.name
    )
    technical = pipeline.describe_component(
        "fixture", "sphinx", display="pd", technical=True
    )
    assert {"services", "input_fields", "output_fields", "optional_inputs"} <= set(
        technical.name
    )
    assert (
        ":param count:"
        in pipeline.describe_component("fixture", "sphinx", display="raw")[
            "description"
        ]
    )
