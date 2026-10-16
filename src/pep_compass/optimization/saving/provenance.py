"""Content-based code provenance for replay across working-tree changes."""

import hashlib
from functools import lru_cache
from pathlib import Path


def source_digest():
    """Fingerprint Python sources; cache hashing while file metadata is unchanged."""
    root = Path(__file__).resolve().parents[2]
    signature = tuple(
        (str(path.relative_to(root)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(root.rglob("*.py"))
    )
    return _digest(str(root), signature)


@lru_cache(maxsize=1)
def _digest(root, signature):
    digest = hashlib.sha256()
    for relative, _, _ in signature:
        digest.update(relative.encode())
        digest.update((Path(root) / relative).read_bytes())
    return digest.hexdigest()
