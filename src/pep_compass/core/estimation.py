"""Static stability estimates for declared PepCompass computation graphs."""

from __future__ import annotations

from dataclasses import dataclass

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
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
    node = _estimate_step(specification.root, input_candidates)
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
        warnings=node.warnings,
    )


def _estimate_step(
    specification: StepSpecification,
    input_candidates: int | None,
) -> _NodeEstimate:
    """Propagate an upper cardinality bound through one graph node."""
    if isinstance(specification, ComponentSpecification):
        return _estimate_component(specification, input_candidates)
    if isinstance(specification, FlowSpecification):
        current = input_candidates
        peak = input_candidates
        warnings: tuple[str, ...] = ()
        for step in specification.steps:
            estimate = _estimate_step(step, current)
            current = estimate.output
            peak = _maximum_known(peak, estimate.peak)
            warnings += estimate.warnings
        return _NodeEstimate(current, peak, warnings)
    if isinstance(specification, LoopSpecification):
        current = input_candidates
        peak = input_candidates
        warnings: tuple[str, ...] = ()
        for _ in range(specification.iterations):
            estimate = _estimate_step(specification.body, current)
            current = estimate.output
            peak = _maximum_known(peak, estimate.peak)
            warnings += estimate.warnings
            if current is None:
                break
        return _NodeEstimate(current, peak, warnings)
    if isinstance(specification, ParallelSpecification):
        branches = [
            _estimate_step(branch.body, input_candidates)
            for branch in specification.branches
        ]
        output = _sum_known(branch.output for branch in branches)
        branch_peak = _sum_known(branch.peak for branch in branches)
        peak = _maximum_known(input_candidates, branch_peak, output)
        return _NodeEstimate(
            output,
            peak,
            tuple(warning for branch in branches for warning in branch.warnings),
        )
    if isinstance(specification, LocalEnumerationSpecification):
        generated = _estimate_step(specification.generator, input_candidates)
        filtered = _estimate_step(specification.filters, generated.output)
        if specification.iterations is None:
            return _NodeEstimate(
                None,
                None,
                filtered.warnings
                + ("Local enumeration uses a runtime walk-time bound.",),
            )
        emissions = 1 + specification.iterations
        output = (
            filtered.output * specification.trajectories * emissions
            if filtered.output is not None
            else None
        )
        if specification.include_walk_points and output is not None:
            output += (
                input_candidates
                * specification.trajectories
                * specification.iterations
            )
        return _NodeEstimate(output, output, filtered.warnings)
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _estimate_component(
    specification: ComponentSpecification,
    input_candidates: int | None,
) -> _NodeEstimate:
    """Apply known component-specific cardinality contracts."""
    if input_candidates is None:
        return _NodeEstimate(None, None)
    if specification.kind == "mutation_generator":
        maximum = specification.parameters.get("maximum_candidates")
        if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0:
            output = input_candidates * maximum
            return _NodeEstimate(output, output)
        return _NodeEstimate(
            None,
            None,
            (f"Unbounded mutation generator: {specification.method}",),
        )
    if specification.kind == "filter" and specification.method == "candidate_subset":
        count = specification.parameters.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            output = min(input_candidates, count)
            return _NodeEstimate(output, input_candidates)
    return _NodeEstimate(input_candidates, input_candidates)


def _maximum_known(*values: int | None) -> int | None:
    """Return the maximum only when every supplied bound is known."""
    return max(values) if all(value is not None for value in values) else None


def _sum_known(values) -> int | None:
    """Return the sum only when every supplied bound is known."""
    materialized = tuple(values)
    return sum(materialized) if all(value is not None for value in materialized) else None
