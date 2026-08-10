"""Local peptide enumeration with an isolated SORBES trajectory stream."""

from __future__ import annotations

from pep_compass.data.optimization import CandidateBatch, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

_TRANSIENT_WALKER_FIELDS = (
    "walker.singular_values",
    "walker.left_vectors",
    "walker.adjusted_time_step",
    "walker.tangent_space",
    "tangent_geometry",
    "point_id",
)


class LocalEnumeration(Step):
    """Collect local mutations without feeding them back into SORBES.

    The operation implements the control flow of Local Enumeration. For every
    input seed and trajectory it first emits filtered MUTANG candidates at the
    initial point. It then advances SORBES one step at a time, emits the decoded
    walk point, and emits a separately filtered MUTANG batch. Only the walk
    point continues to the next iteration.

    Global selection, deduplication, oracle evaluation, and Bayesian
    optimization remain ordinary steps placed after this operation.

    :param walker: One-step trajectory transformation, normally SORBES.
    :param mutation_generator: Local mutation expansion, normally MUTANG.
    :param filters: Ordered filter flow applied to each local mutation batch.
    :param trajectories: Independent trajectories started from every seed.
    :param iterations: Fixed steps per trajectory, mutually exclusive with
        ``walk_time``.
    :param walk_time: Accumulated adjusted SORBES time per trajectory, mutually
        exclusive with ``iterations``.
    :param include_walk_points: Include decoded SORBES points in the result.
    """

    def __init__(
        self,
        *,
        walker: Step,
        mutation_generator: Step,
        filters: Step,
        trajectories: int,
        iterations: int | None,
        walk_time: float | None,
        include_walk_points: bool = True,
    ) -> None:
        if trajectories < 1:
            raise ValueError("Local-enumeration trajectories must be positive.")
        if (iterations is None) == (walk_time is None):
            raise ValueError(
                "Local enumeration requires exactly one of iterations or walk_time."
            )
        if iterations is not None and iterations < 1:
            raise ValueError("Local-enumeration iterations must be positive.")
        if walk_time is not None and walk_time <= 0:
            raise ValueError("Local-enumeration walk time must be positive.")
        self.walker = walker
        self.mutation_generator = mutation_generator
        self.filters = filters
        self.trajectories = trajectories
        self.iterations = iterations
        self.walk_time = walk_time
        self.include_walk_points = include_walk_points
        requirement = getattr(mutation_generator, "geometry_requirement", None)
        contract = getattr(walker, "geometry_contract", None)
        if getattr(mutation_generator, "requires_local_enumeration", False):
            if requirement is None or contract is None or not contract.satisfies(requirement):
                raise ValueError(
                    "LocalEnumeration walker geometry does not satisfy MUTANG shared geometry."
                )

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute each reusable child exactly once."""
        self.walker.precompute(context.enter_step(self.walker.name))
        self.mutation_generator.precompute(
            context.enter_step(self.mutation_generator.name)
        )
        self.filters.precompute(context.enter_step(self.filters.name))

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute each seed trajectory sequentially and collect emissions."""
        # Local enumeration
        ## Keep each trajectory private to prevent mutation-tree expansion
        emissions: list[CandidateBatch] = []
        trajectory_index = 0
        for seed_index in range(len(batch)):
            seed = batch.select([seed_index])
            for replica_index in range(self.trajectories):
                if context.state.stop_requested:
                    break
                trajectory_context = context.enter_branch(
                    f"seed_{seed_index:05d}_trajectory_{replica_index:05d}",
                    trajectory_index,
                )
                trajectory_index += 1
                emissions.extend(self._run_trajectory(seed, trajectory_context))

        if not emissions:
            return batch.select([])
        result = CandidateBatch.concatenate(emissions)
        logger.info(
            "Local enumeration seeds=%s trajectories_per_seed=%s "
            "output_candidates=%s.",
            len(batch),
            self.trajectories,
            len(result),
        )
        return result

    def _run_trajectory(
        self,
        seed: CandidateBatch,
        context: OptimizationContext,
    ) -> list[CandidateBatch]:
        """Run one trajectory while retaining candidates outside its state."""
        # Trajectory point
        current = seed
        emissions: list[CandidateBatch] = []
        elapsed = 0.0
        iteration = 0

        # SORBES trajectory
        while self._should_continue(iteration, elapsed, context):
            iteration_context = context.enter_iteration(iteration)
            current = self.walker(current, iteration_context)
            time_increment = self._time_increment(current)
            if self.include_walk_points:
                emissions.append(current.without_fields(_TRANSIENT_WALKER_FIELDS))
            emissions.append(self._generate_and_filter(current, iteration_context))
            elapsed += time_increment
            iteration += 1
            context.state.record_iteration()
        logger.debug(
            "Local-enumeration trajectory iterations=%s elapsed_walk_time=%.6f "
            "emissions=%s.",
            iteration,
            elapsed,
            len(emissions),
        )
        return emissions

    def _generate_and_filter(
        self,
        point: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Generate MUTANG candidates and apply configured local filters."""
        generated = self.mutation_generator(point, context)
        filtered = self.filters(generated, context)
        return filtered.without_fields(_TRANSIENT_WALKER_FIELDS)

    def _should_continue(
        self,
        iteration: int,
        elapsed: float,
        context: OptimizationContext,
    ) -> bool:
        """Evaluate the configured trajectory bound and global stop state."""
        if context.state.stop_requested:
            return False
        if self.iterations is not None:
            return iteration < self.iterations
        assert self.walk_time is not None
        return elapsed < self.walk_time

    def _time_increment(self, point: CandidateBatch) -> float:
        """Return adjusted SORBES time or a unit step for fixed iterations."""
        if self.walk_time is None:
            return 1.0
        field = point.fields.get("walker.adjusted_time_step")
        if not isinstance(field, TensorField) or field.values.numel() != 1:
            raise ValueError(
                "Walk-time local enumeration requires one scalar "
                "walker.adjusted_time_step value per trajectory."
            )
        increment = float(field.values.item())
        if increment <= 0:
            raise ValueError("Walker adjusted time step must be positive.")
        return increment
