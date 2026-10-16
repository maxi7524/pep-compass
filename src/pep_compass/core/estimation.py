"""Static stability estimates for declared PepCompass computation graphs."""

from __future__ import annotations

from dataclasses import dataclass

from pep_compass.data_schemas.specification import (
    ComponentSpecification,
    FlowSpecification,
    LocalEnumerationSpecification,
    LoopSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)


@dataclass(frozen=True, slots=True)
class StabilityEstimate:
    """Describe a conservative static candidate and latent-memory estimate."""

    input_candidates: int
    output_candidates_upper: int | None
    peak_candidates_upper: int | None
    latent_bytes_upper: int | None
    warnings: tuple[str, ...]
    nodes: tuple["NodeStabilityEstimate", ...] = ()


@dataclass(frozen=True, slots=True)
class NodeStabilityEstimate:
    """Describe cardinality propagation through one declared graph node.

    :param path: Stable path within the neutral pipeline specification.
    :param operation: Human-readable operation and configured method.
    :param input_candidates: Candidate upper bound entering the node.
    :param output_candidates: Candidate upper bound leaving the node.
    :param peak_candidates: Candidate upper bound retained within the node.
    :param uncertainty: Explanation when an upper bound cannot be proven.
    """

    path: str
    operation: str
    input_candidates: int | None
    output_candidates: int | None
    peak_candidates: int | None
    uncertainty: str | None = None


@dataclass(frozen=True, slots=True)
class _NodeEstimate:
    """Carry cardinality bounds through recursive graph estimation."""

    output: int | None
    peak: int | None
    warnings: tuple[str, ...] = ()


def estimate_pipeline_stability(
    specification: PipelineSpecification,
    *,
    input_candidates: int,
    latent_dimension: int,
    latent_element_bytes: int = 4,
    catalog,
) -> StabilityEstimate:
    """Estimate candidate amplification and retained latent storage.

    :param specification: Neutral pipeline declaration.
    :param input_candidates: Number of candidates entering the pipeline.
    :param latent_dimension: Autoencoder latent dimension ``D``.
    :param latent_element_bytes: Storage per latent tensor element.
    :return: Conservative static estimate with explicit uncertainty warnings.
    """
    if input_candidates < 1 or latent_dimension < 1 or latent_element_bytes < 1:
        raise ValueError("Stability-estimation dimensions must be positive.")
    nodes: list[NodeStabilityEstimate] = []
    node = _estimate_step(
        specification.root,
        input_candidates,
        path="pipeline",
        nodes=nodes,
        catalog=catalog,
    )
    latent_bytes = (
        node.peak * latent_dimension * latent_element_bytes
        if node.peak is not None
        else None
    )
    return StabilityEstimate(
        input_candidates=input_candidates,
        output_candidates_upper=node.output,
        peak_candidates_upper=node.peak,
        latent_bytes_upper=latent_bytes,
        warnings=tuple(dict.fromkeys(node.warnings)),
        nodes=tuple(nodes),
    )


def _estimate_step(
    specification: StepSpecification,
    input_candidates: int | None,
    *,
    path: str = "pipeline",
    nodes: list[NodeStabilityEstimate] | None = None,
    catalog=None,
) -> _NodeEstimate:
    """Propagate an upper cardinality bound through one graph node."""
    node_records = nodes if nodes is not None else []
    if isinstance(specification, ComponentSpecification):
        estimate = _estimate_component(specification, input_candidates, catalog=catalog)
        _record_node(
            node_records,
            path,
            f"{specification.kind}:{specification.method}",
            input_candidates,
            estimate,
        )
        return estimate
    if isinstance(specification, FlowSpecification):
        current = input_candidates
        peak = input_candidates
        warnings: tuple[str, ...] = ()
        for index, step in enumerate(specification.steps):
            estimate = _estimate_step(
                step,
                current,
                path=f"{path}.steps[{index}]",
                nodes=node_records,
                catalog=catalog,
            )
            current = estimate.output
            peak = _maximum_known(peak, estimate.peak)
            warnings += estimate.warnings
        estimate = _NodeEstimate(current, peak, warnings)
        _record_node(node_records, path, "flow", input_candidates, estimate)
        return estimate
    if isinstance(specification, LoopSpecification):
        current = input_candidates
        peak = input_candidates
        warnings: tuple[str, ...] = ()
        for index in range(specification.iterations):
            estimate = _estimate_step(
                specification.body,
                current,
                path=f"{path}.iteration[{index}]",
                nodes=node_records,
                catalog=catalog,
            )
            current = estimate.output
            peak = _maximum_known(peak, estimate.peak)
            warnings += estimate.warnings
            if current is None:
                break
        estimate = _NodeEstimate(current, peak, warnings)
        _record_node(node_records, path, "loop", input_candidates, estimate)
        return estimate
    if isinstance(specification, ParallelSpecification):
        branches = [
            _estimate_step(
                branch.body,
                input_candidates,
                path=f"{path}.branch[{branch.name}]",
                nodes=node_records,
                catalog=catalog,
            )
            for branch in specification.branches
        ]
        output = _sum_known(branch.output for branch in branches)
        branch_peak = _sum_known(branch.peak for branch in branches)
        peak = _maximum_known(input_candidates, branch_peak, output)
        estimate = _NodeEstimate(
            output,
            peak,
            tuple(warning for branch in branches for warning in branch.warnings),
        )
        _record_node(node_records, path, "parallel", input_candidates, estimate)
        return estimate
    if isinstance(specification, LocalEnumerationSpecification):
        generated = _estimate_step(
            specification.generator,
            input_candidates,
            path=f"{path}.mutation_generator",
            nodes=node_records,
            catalog=catalog,
        )
        filtered = _estimate_step(
            specification.filters,
            generated.output,
            path=f"{path}.filters",
            nodes=node_records,
            catalog=catalog,
        )
        if specification.iterations is None:
            estimate = _NodeEstimate(
                None,
                None,
                filtered.warnings
                + ("Local enumeration uses a runtime walk-time bound.",),
            )
            _record_node(
                node_records,
                path,
                "local_enumeration",
                input_candidates,
                estimate,
            )
            return estimate
        emissions = 1 + specification.iterations
        output = (
            filtered.output * specification.trajectories * emissions
            if filtered.output is not None
            else None
        )
        if specification.include_walk_points and output is not None:
            output += (
                input_candidates * specification.trajectories * specification.iterations
            )
        estimate = _NodeEstimate(output, output, filtered.warnings)
        _record_node(
            node_records,
            path,
            "local_enumeration",
            input_candidates,
            estimate,
        )
        return estimate
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _estimate_component(
    specification: ComponentSpecification,
    input_candidates: int | None,
    *,
    catalog=None,
) -> _NodeEstimate:
    """Apply known component-specific cardinality contracts."""
    if input_candidates is None:
        return _NodeEstimate(None, None)
    from pep_compass.registry.components.contracts import component_output_bound

    output = component_output_bound(
        specification.kind,
        specification.method,
        specification.parameters,
        input_candidates,
        catalog=catalog,
    )
    if output is None:
        return _NodeEstimate(
            None, None, (f"Unbounded mutation generator: {specification.method}",)
        )
    return _NodeEstimate(output, max(input_candidates, output))


def _record_node(
    nodes: list[NodeStabilityEstimate],
    path: str,
    operation: str,
    input_candidates: int | None,
    estimate: _NodeEstimate,
) -> None:
    """Append a public node estimate with an explicit uncertainty reason."""
    uncertainty = None
    if estimate.output is None or estimate.peak is None:
        uncertainty = (
            estimate.warnings[-1]
            if estimate.warnings
            else ("An upstream node has no finite candidate bound.")
        )
    nodes.append(
        NodeStabilityEstimate(
            path=path,
            operation=operation,
            input_candidates=input_candidates,
            output_candidates=estimate.output,
            peak_candidates=estimate.peak,
            uncertainty=uncertainty,
        )
    )


def _maximum_known(*values: int | None) -> int | None:
    """Return the maximum only when every supplied bound is known."""
    return max(values) if all(value is not None for value in values) else None


def _sum_known(values) -> int | None:
    """Return the sum only when every supplied bound is known."""
    materialized = tuple(values)
    return (
        sum(materialized) if all(value is not None for value in materialized) else None
    )
