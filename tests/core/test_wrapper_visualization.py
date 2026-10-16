"""Graph routing and readable labels without running computational components."""

import re
import xml.etree.ElementTree as ET

import pytest

from pep_compass import LocalEnumeration, Loop, Parallel, Pipeline, Step

SVG = {"s": "http://www.w3.org/2000/svg"}


def unique():
    return Step.create(component="filter", strategy="deduplicate")


def points(edge):
    return [
        tuple(map(float, point.split(",")))
        for point in re.findall(r"[-\d.]+,[-\d.]+", edge.attrib["d"])
    ]


def test_nested_entry_points_right_and_returns_point_left():
    graph = Pipeline.create(
        steps=[
            Loop.create(
                iterations=2,
                steps=[
                    Parallel.create(
                        branches={
                            "left": [unique()],
                            "right": [Loop.create(iterations=2, steps=[unique()])],
                        }
                    )
                ],
            )
        ]
    )
    svg = ET.fromstring(graph.to_schema("svg"))
    entries = svg.findall('.//s:path[@data-kind="enter"]', SVG)
    returns = svg.findall('.//s:path[@data-kind="return"]', SVG)
    assert len(entries) == len(returns) == 4
    for edge in entries:
        a, b = points(edge)[-2:]
        assert b[0] > a[0] and b[1] == a[1]
    for edge in returns:
        a, b = points(edge)[-2:]
        assert b[0] < a[0] and b[1] == a[1]


@pytest.mark.parametrize("local", [False, True])
def test_feedback_arrow_ends_on_the_first_body_or_walker_node(local):
    body = unique()
    if local:
        body = Step.create(component="walker", strategy="sorbes")
        node = LocalEnumeration.create(
            walker=body,
            mutation_generator=Step.create(
                component="mutation_generator",
                strategy="mutang",
                parameters={"maximum_candidates": 4},
            ),
            filters=[unique()],
            iterations=1,
        )
    else:
        node = Loop.create(iterations=2, steps=[body])
    svg = ET.fromstring(Pipeline.create(steps=[node]).to_schema("svg"))
    edge = svg.find(
        f'.//s:path[@data-kind="feedback"][@data-owner="{node.node_id}"]', SVG
    )
    rectangle = svg.find(f'.//s:g[@data-node-id="{body.node_id}"]/s:rect', SVG)
    target = points(edge)[-1]
    assert target == (
        float(rectangle.attrib["x"]) + float(rectangle.attrib["width"]),
        float(rectangle.attrib["y"]) + 24,
    )
    assert points(edge)[-2][0] > target[0]
    assert "stroke-dasharray" in edge.attrib


def test_named_oracle_keeps_its_family_strategy_and_escaped_label():
    node = Step.create(
        component="oracle", strategy="hydrophobicity", name="Score <candidate>"
    )
    graph = Pipeline.create(steps=[node])
    for format in ("svg", "mermaid"):
        rendered = graph.to_schema(format)
        assert "Oracle" in rendered and "hydrophobicity" in rendered
        assert "&lt;candidate&gt;" in rendered
        assert "<candidate>" not in rendered
    assert (
        graph.show("mermaid")._repr_mimebundle_()["text/plain"].startswith("flowchart")
    )


def test_compact_strategy_labels_keep_exact_values_in_the_tooltip():
    node = Step.create(
        component="walker", strategy="sorbes", parameters={"time_step": 0.01}
    )
    svg = ET.fromstring(Pipeline.create(steps=[node]).to_schema("svg"))
    labels = " ".join(element.text or "" for element in svg.findall(".//s:text", SVG))
    assert "kappa_stable(" in labels
    # REMARK: the public time_step alias becomes epsilon=sqrt(time_step).
    assert node.parameters["position_update"]["parameters"]["epsilon"] == 0.1
    assert "epsilon: 0.1" in labels
    assert "delta max: 0.5" in labels
    assert "0.010000000000000002" not in labels
    assert '"parameters"' not in labels
    assert "Position: 0" in svg.find(".//s:g[@data-node-id]/s:title", SVG).text
