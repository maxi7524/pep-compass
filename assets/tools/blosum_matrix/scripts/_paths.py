"""Shared paths for BLOSUM matrix generation tools."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PACKAGE_SRC = PACKAGE_ROOT / "src"

DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "src"
    / "pep_compass"
    / "optimization"
    / "components"
    / "helpers"
    / "substitution_matrices"
    / "matrices"
)
DEFAULT_BLOCKS_GLOB = "len_*.fasta"


def ensure_package_importable() -> None:
    """Put ``blosum_matrix`` on ``sys.path`` (src layout under tools/blosum_matrix)."""

    path = str(PACKAGE_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
