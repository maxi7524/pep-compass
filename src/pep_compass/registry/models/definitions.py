"""Named models and required filesystem artifact declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FileArtifact:
    """Require one model file relative to a resolved model directory."""

    relative_path: str


@dataclass(frozen=True, slots=True)
class GlobArtifact:
    """Require an exact number of files matching a pattern in one directory."""

    pattern: str
    expected_count: int


@dataclass(frozen=True, slots=True)
class DirectoryArtifact:
    """Require one model directory and selected files below it.

    :param relative_path: Directory relative to the model descriptor directory.
    :param required_files: Files whose presence identifies a complete directory
        artifact, for example the metadata and variables of a SavedModel.
    """

    relative_path: str
    required_files: tuple[str, ...] = ()


ModelArtifact = DirectoryArtifact | FileArtifact | GlobArtifact


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Declare named files required by a selectable model variant."""

    provider: str
    name: str
    directory: str
    artifacts: tuple[ModelArtifact, ...]
    parameters: dict[str, Any]
    download_script: str | None = None

    def __post_init__(self):
        """Keep named-model defaults independent of caller-owned mappings."""
        from pep_compass.data_schemas.graph import freeze

        object.__setattr__(self, "parameters", freeze(self.parameters))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True, slots=True)
class AutoencoderModelDescriptor:
    """Describe a named model variant supported by one autoencoder method.

    :param name: Public model name used in configuration.
    :param parameters: Loader parameters associated with the model variant.
    """

    name: str
    parameters: dict[str, Any]
    directory: str | None = None
    artifacts: tuple[ModelArtifact, ...] = ()
    download_script: str | None = None
