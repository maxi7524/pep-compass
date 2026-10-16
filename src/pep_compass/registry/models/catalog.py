"""Named model lookup, artifact validation and provenance capture."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pep_compass.registry.models.definitions import (
    DirectoryArtifact,
    FileArtifact,
    ModelDescriptor,
)
from pep_compass.registry.models.provenance import _artifact_capture
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelRegistry:
    """Store named model descriptors and validate their artifact sets."""

    def __init__(self) -> None:
        self._descriptors: dict[tuple[str, str], ModelDescriptor] = {}

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register one unique ``(provider, name)`` descriptor."""
        key = (descriptor.provider, descriptor.name)
        if not descriptor.provider or not descriptor.name:
            raise ValueError("Model provider and name cannot be empty.")
        if key in self._descriptors:
            raise ValueError(
                f"Duplicate model registration: {descriptor.provider}/{descriptor.name}."
            )
        self._descriptors[key] = descriptor

    def names(self, provider: str) -> tuple[str, ...]:
        """Return model names registered under one provider."""
        return tuple(
            sorted(
                name
                for registered_provider, name in self._descriptors
                if registered_provider == provider
            )
        )

    def descriptor(self, provider: str, name: str) -> ModelDescriptor:
        """Return one model descriptor.

        :raises ValueError: If the provider/model pair is unknown.
        """
        try:
            return self._descriptors[(provider, name)]
        except KeyError as error:
            raise ValueError(
                f"Unknown model {provider}/{name}. Available: {list(self.names(provider))}."
            ) from error

    def providers(self):
        """List providers without loading model weights or touching artifacts."""
        return tuple(sorted({provider for provider, _ in self._descriptors}))

    def describe(self, provider, name=None):
        """Describe named artifacts and default loader parameters without loading."""
        from dataclasses import asdict

        from pep_compass.data_schemas.graph import thaw

        def describe_model(descriptor):
            return {
                "provider": descriptor.provider,
                "name": descriptor.name,
                "directory": descriptor.directory,
                "artifacts": [
                    {"type": type(artifact).__name__, **asdict(artifact)}
                    for artifact in descriptor.artifacts
                ],
                "parameters": thaw(descriptor.parameters),
                "download_script": descriptor.download_script,
            }

        if name is not None:
            return describe_model(self.descriptor(provider, name))
        return [
            describe_model(self.descriptor(provider, method))
            for method in self.names(provider)
        ]

    def resolve(
        self,
        provider: str,
        name: str,
        *,
        root: str | Path,
    ) -> tuple[ModelDescriptor, tuple[Path, ...]]:
        """Resolve and validate files for one registered model.

        :param provider: Model provider namespace.
        :param name: Model variant name.
        :param root: Root containing descriptor-relative model directories.
        :return: Descriptor and deterministically ordered artifact paths.
        """
        descriptor = self.descriptor(provider, name)
        directory = Path(root) / descriptor.directory
        if not directory.is_dir():
            self._missing(descriptor, f"model directory not found: {directory}")
        paths: list[Path] = []
        for artifact in descriptor.artifacts:
            if isinstance(artifact, DirectoryArtifact):
                path = directory / artifact.relative_path
                if not path.is_dir():
                    self._missing(descriptor, f"model directory not found: {path}")
                for required_file in artifact.required_files:
                    required_path = path / required_file
                    if not required_path.is_file():
                        self._missing(
                            descriptor,
                            f"model directory is incomplete; file not found: "
                            f"{required_path}",
                        )
                paths.append(path)
            elif isinstance(artifact, FileArtifact):
                path = directory / artifact.relative_path
                if not path.is_file():
                    self._missing(descriptor, f"model file not found: {path}")
                paths.append(path)
            else:
                matches = tuple(sorted(directory.glob(artifact.pattern)))
                if len(matches) != artifact.expected_count:
                    self._missing(
                        descriptor,
                        f"expected {artifact.expected_count} files matching "
                        f"{artifact.pattern!r}, found {len(matches)} under {directory}",
                    )
                paths.extend(matches)
        logger.debug(
            "Resolved model provider=%s name=%s artifacts=%s directory=%s.",
            provider,
            name,
            len(paths),
            directory,
        )
        capture = _artifact_capture.get()
        if capture is not None:
            files = []
            for artifact in paths:
                for path in (
                    sorted(artifact.rglob("*")) if artifact.is_dir() else (artifact,)
                ):
                    if path.is_file():
                        digest = hashlib.sha256()
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                        files.append(
                            {"path": str(path.resolve()), "sha256": digest.hexdigest()}
                        )
            capture.append({"provider": provider, "model": name, "files": files})
        return descriptor, tuple(paths)

    @staticmethod
    def _missing(descriptor: ModelDescriptor, detail: str) -> None:
        hint = (
            f" Run {descriptor.download_script} to install the model."
            if descriptor.download_script
            else ""
        )
        raise FileNotFoundError(f"{detail}.{hint}")


# One catalog shared by autoencoders and all model-backed components.
model_catalog = ModelRegistry()
