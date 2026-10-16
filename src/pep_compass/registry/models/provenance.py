"""Build-scoped collection of model artifacts for reproducible execution."""

from contextlib import contextmanager
from contextvars import ContextVar

_artifact_capture = ContextVar("pep_compass_artifact_capture", default=None)


@contextmanager
def capture_model_artifacts():
    """Collect model files actually resolved by factories in this build scope."""
    records = []
    token = _artifact_capture.set(records)
    try:
        yield records
    finally:
        _artifact_capture.reset(token)
