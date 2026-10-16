"""Reader adapters for canonical runtime tables and legacy experiment tables."""

from pathlib import Path

from pep_compass.analysis.reader.data_schemas.base import DataSchema
from pep_compass.data_schemas.result_schema import RESULT_SCHEMA

SCHEMAS = {
    table.name: DataSchema(
        table.name, Path(table.relative_path).name, frozenset(table.required_columns)
    )
    for table in RESULT_SCHEMA.tables
}

# Legacy experiment tables retain their original contracts.
SCHEMAS.update(
    {
        "evaluations": DataSchema(
            "evaluations",
            "evaluations.csv",
            frozenset({"run_id", "iteration_id", "candidate_id", "sequence"}),
        ),
        "optimizer_iterations": DataSchema(
            "optimizer_iterations",
            "optimizer_iterations.csv",
            frozenset(
                {"run_id", "iteration_id", "trajectory_count", "walker_step_count"}
            ),
        ),
        "walker_steps": DataSchema(
            "walker_steps",
            "walker_steps.csv",
            frozenset(
                {
                    "run_id",
                    "iteration_id",
                    "trajectory_id",
                    "step_id",
                    "current_latent_position",
                    "next_latent_position",
                }
            ),
        ),
        "unique_candidates": DataSchema(
            "unique_candidates",
            "unique_candidates.csv",
            frozenset({"run_id", "candidate_id", "sequence", "encoded_latent_mean"}),
        ),
        "candidate_occurrences": DataSchema(
            "candidate_occurrences",
            "candidate_occurrences.csv",
            frozenset(
                {"run_id", "iteration_id", "trajectory_id", "step_id", "candidate_id"}
            ),
        ),
        "evaluated_origins": DataSchema(
            "evaluated_origins",
            "evaluated_origins.csv",
            frozenset(
                {
                    "run_id",
                    "evaluation_iteration_id",
                    "candidate_id",
                    "trajectory_id",
                    "step_id",
                }
            ),
        ),
    }
)


def get_schema(name: str) -> DataSchema:
    """Return a registered schema by logical name."""
    try:
        return SCHEMAS[name]
    except KeyError as error:
        raise KeyError(f"Unknown experiment table: {name}") from error
