"""Small historical-format archives generated without the removed runtime API."""

import json

import pandas as pd
import torch

from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
)
from tests.fixtures.archive_tracking import CSVStepTracker


def write_archive(root, workflow, *, seed=None):
    """Generate one verified CSV/JSON/PT run for reader and replay regressions.

    :param root: Isolated fixture directory.
    :param workflow: Deterministic builder accepting tracker/stability_monitor.
    :param seed: Optional reproducible run seed.
    :return: The generated archive run directory.
    """
    directory = root / "variants" / "variant_00000" / "runs" / "run_00000"
    tracker = CSVStepTracker(
        directory / "tracking",
        level="normal",
        run_id="run_00000",
        variant_id="variant_00000",
    )
    pipeline = workflow.build_pipeline(
        {}, tracker=tracker, stability_monitor=NullStabilityMonitor()
    )
    result = pipeline.run(["AA"], seed=seed)
    payload = {
        "schema_version": "2",
        "status": "completed",
        "run_id": "run_00000",
        "variant_id": "variant_00000",
        "seed": seed,
        "sequence": "AA",
        "objective_name": None,
    }
    (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (directory / "resolved_config.json").write_text(
        json.dumps(
            {
                "pipeline": {},
                "autoencoder": {"method": "mock", "model": "default", "device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "candidate_index": range(len(result.candidates)),
            "sequence": result.candidates.sequences,
        }
    ).to_csv(directory / "candidates.csv", index=False)
    from pep_compass.optimization.stability_estimation.monitoring import MemorySnapshot

    pd.DataFrame(columns=list(MemorySnapshot.__dataclass_fields__)).to_csv(
        directory / "tracking/stability.csv", index=False
    )
    torch.save(
        result.candidates.latent_origins.detach().cpu(), directory / "latent_origins.pt"
    )
    return directory
