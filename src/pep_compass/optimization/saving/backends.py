"""Shared relational storage in RAM or a persistent SQLite run bundle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock

from pep_compass.optimization.saving.serialization import dumps, loads
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE runs(run_id TEXT PRIMARY KEY, status TEXT NOT NULL, settings_sha256 TEXT NOT NULL, summary_json TEXT NOT NULL);
CREATE TABLE step_executions(execution_id INTEGER PRIMARY KEY, node_id TEXT, parent_execution_id INTEGER, scope_json TEXT, input_count INTEGER, output_count INTEGER, duration_seconds REAL, status TEXT, phase TEXT, error TEXT);
CREATE TABLE iterations(iteration_id INTEGER PRIMARY KEY, node_id TEXT, iteration_index INTEGER, scope_json TEXT, trajectories INTEGER, elapsed_seconds REAL, duration_seconds REAL, interval_seconds REAL);
CREATE TABLE peptides(sequence TEXT PRIMARY KEY);
CREATE TABLE evaluated_peptides(objective TEXT, sequence TEXT, score REAL, direction TEXT, evaluation_index INTEGER, first_evaluation_index INTEGER, lineage_json TEXT, PRIMARY KEY(objective,sequence));
CREATE TABLE observations(observation_id INTEGER PRIMARY KEY, objective TEXT, sequence TEXT, score REAL, direction TEXT, lineage_json TEXT);
CREATE TABLE candidate_sets(set_id INTEGER PRIMARY KEY, name TEXT, node_id TEXT, execution_id INTEGER, kind TEXT, payload_id TEXT);
CREATE TABLE component_records(record_id INTEGER PRIMARY KEY, node_id TEXT, execution_id INTEGER, name TEXT, payload_id TEXT);
CREATE TABLE checkpoints(checkpoint_id INTEGER PRIMARY KEY, node_id TEXT, execution_id INTEGER, scope_json TEXT, phase TEXT, payload_id TEXT, output_count INTEGER, output_sha256 TEXT);
CREATE TABLE tensor_payloads(payload_id TEXT PRIMARY KEY, codec_version INTEGER, data BLOB NOT NULL);
CREATE INDEX evaluated_objective ON evaluated_peptides(objective);
CREATE INDEX execution_node ON step_executions(node_id,execution_id);
CREATE INDEX checkpoints_node ON checkpoints(node_id,execution_id);
"""


def canonical_json(value):
    """Serialize run settings deterministically, rejecting opaque objects."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


class SQLiteStorageBackend:
    """Persist a single run with transaction batching and bounded table access.

    :param settings: Fully resolved, JSON-compatible run manifest.
    :param path: New output directory, or ``None`` for a memory database.
    :raises FileExistsError: If output would overwrite an existing run bundle.
    """

    def __init__(self, settings, path=None):
        self.path = Path(path).resolve() if path is not None else None
        self.lock = RLock()
        self.closed = False
        self.settings = settings
        serialized = canonical_json(settings)
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            if (self.path / "results.sqlite").exists() or (
                self.path / "settings.json"
            ).exists():
                raise FileExistsError(f"Run bundle already exists: {self.path}")
        self.connection = sqlite3.connect(
            str(self.path / "results.sqlite") if self.path is not None else ":memory:",
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
        self.connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?)",
            (
                settings["run_id"],
                "running",
                hashlib.sha256(serialized.encode()).hexdigest(),
                "{}",
            ),
        )
        if self.path is not None:
            temporary = self.path / "settings.json.tmp"
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.path / "settings.json")
        self.connection.commit()
        logger.info(
            "Opened run storage backend=%s path=%s.",
            "sqlite" if self.path is not None else "memory",
            self.path,
        )

    def execute(self, statement, parameters=()):
        """Execute an internal parameterized statement under the backend lock."""
        with self.lock:
            return self.connection.execute(statement, parameters)

    def payload(self, value):
        """Store a content-addressed, typed payload and return its identity."""
        data = dumps(value)
        identity = hashlib.sha256(data).hexdigest()
        self.execute(
            "INSERT OR IGNORE INTO tensor_payloads VALUES (?,?,?)", (identity, 1, data)
        )
        return identity

    def load_payload(self, identity):
        """Read one explicitly requested payload and check its checksum."""
        row = self.execute(
            "SELECT data FROM tensor_payloads WHERE payload_id=?", (identity,)
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        data = bytes(row[0])
        if hashlib.sha256(data).hexdigest() != identity:
            raise ValueError("Run payload checksum mismatch.")
        return loads(data)

    def flush(self):
        """Commit every buffered record without dropping intermediate events."""
        with self.lock:
            self.connection.commit()

    def export(self, path):
        """Copy retained data and immutable settings to a new run bundle."""
        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)
        if (target / "results.sqlite").exists() or (target / "settings.json").exists():
            raise FileExistsError(target)
        self.flush()
        with self.lock, sqlite3.connect(target / "results.sqlite") as destination:
            self.connection.backup(destination)
        (target / "settings.json").write_text(
            canonical_json(self.settings), encoding="utf-8"
        )
        logger.info("Exported retained run data path=%s.", target)
        return target

    def close(self):
        """Commit pending records and release the storage connection."""
        with self.lock:
            if self.closed:
                return
            self.connection.commit()
            self.connection.close()
            self.closed = True


class MemoryStorageBackend(SQLiteStorageBackend):
    """Use SQLite's in-memory database for exactly the same logical schema."""

    def __init__(self, settings):
        super().__init__(settings)
