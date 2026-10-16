"""Execute a single constructed graph through the established pipeline engine."""

from __future__ import annotations

import hashlib
import platform
from uuid import uuid4

import torch

from pep_compass.data_schemas.execution import OptimizationLimits, RunnerSettings
from pep_compass.data_schemas.results import RunResult
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.saving.backends import (
    MemoryStorageBackend,
    SQLiteStorageBackend,
)
from pep_compass.optimization.saving.provenance import source_digest
from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.optimization.saving.saver import Saver
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

logger = get_custom_logger(__name__)


def model_digest(model):
    """Fingerprint tensor weights when an injected model exposes state_dict."""
    if not hasattr(model, "state_dict"):
        return None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Runner:
    """Own one run's RNG, observation retention and resource finalization.

    :param pipeline: Built ExecutablePipeline; operator implementations are reused.
    :param schema: Immutable public graph declaration.
    :param settings: Validated defaults for this runner.
    :param model_manifest: Named model artifact provenance collected at build.
    """

    def __init__(self, pipeline, schema, settings=None, model_manifest=()):
        self.pipeline = pipeline
        self.schema = schema
        self.settings = settings or RunnerSettings()
        self.model_manifest = tuple(model_manifest)

    def run(self, sequences, *, settings=None):
        """Return a RunResult; public return-mode selection belongs to the wrapper.

        :param sequences: Non-empty sequence collection.
        :param settings: Optional immutable settings override.
        :return: Completed run with a lazy reader.
        :raises Exception: Original computational or persistence failure.
        """
        settings = settings or self.settings
        from pep_compass.utils.validation import validate_sequences

        sequences = validate_sequences(sequences)
        if settings.execution.device is not None:
            actual = getattr(self.pipeline.autoencoder, "device", "cpu")
            if str(actual) != settings.execution.device:
                raise ValueError(
                    "Execution device differs from the built autoencoder; rebuild its resources."
                )
        limits = OptimizationLimits(
            settings.execution.oracle_calls, settings.execution.generated_candidates
        )
        state = OptimizationState(limits=limits)
        manifest = {
            "format": "pep_compass.run",
            "format_version": 1,
            "run_id": "run_" + uuid4().hex,
            "schema": self.schema.to_dict(),
            "inputs": list(sequences),
            "runner_settings": settings.to_dict(),
            "models": list(self.model_manifest),
            "reproducibility": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "rng_contract_version": 2,
                "source_sha256": source_digest(),
                "autoencoder_sha256": model_digest(self.pipeline.autoencoder),
            },
        }
        backend = (
            MemoryStorageBackend(manifest)
            if settings.storage.backend == "memory"
            else SQLiteStorageBackend(manifest, settings.storage.path)
        )
        saver = Saver(backend, settings.storage)
        state.observer = saver
        context = OptimizationContext(
            autoencoder=self.pipeline.autoencoder,
            tracker=saver,
            seed=settings.execution.seed,
            state=state,
            stability_monitor=self.pipeline.stability_monitor,
            saver=saver,
        )
        logger.info(
            "Executing graph run=%s backend=%s.",
            manifest["run_id"],
            settings.storage.backend,
        )
        try:
            with random_scope(settings.execution.seed):
                result = self.pipeline._run(
                    list(sequences),
                    seed=settings.execution.seed,
                    context=context,
                    close_tracker=False,
                )
            summary = saver.finalize(result, state)
        except BaseException as error:
            logger.error("Graph run=%s failed.", manifest["run_id"], exc_info=True)
            try:
                saver.finalize(
                    None,
                    state,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            except Exception:
                logger.error("Could not persist the failed-run summary.", exc_info=True)
            try:
                backend.close()
            except Exception:
                logger.error("Could not close failed-run storage.", exc_info=True)
            raise
        return RunResult(
            {"status": "completed", **summary}, AnalysisReader(backend), backend.path
        )
