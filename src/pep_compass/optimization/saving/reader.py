"""Lazy, backend-independent access to one versioned run bundle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock

import pandas as pd

from pep_compass.optimization.saving.backends import (
    FORMAT_VERSION,
    SQLiteStorageBackend,
    canonical_json,
)
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class DataUnavailableError(LookupError):
    """The requested data category was not retained by this run."""


class RunTableHandle:
    """Implement the shared TableHandle protocol with bounded SQLite scans."""

    def __init__(self, backend, name):
        self.backend, self.name = backend, name

    @property
    def columns(self):
        """Return validated column names from the fixed logical schema."""
        return tuple(
            row[1] for row in self.backend.execute(f'PRAGMA table_info("{self.name}")')
        )

    def count_rows(self):
        """Count rows without loading payload columns."""
        return self.backend.execute(f'SELECT COUNT(*) FROM "{self.name}"').fetchone()[0]

    def scan(self, columns, *, chunk_size=100_000):
        """Yield requested columns in bounded chunks in storage order."""
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")
        if not columns or set(columns) - set(self.columns):
            raise ValueError("Requested columns are not present in this table.")
        selected = ",".join('"' + column + '"' for column in columns)
        cursor = self.backend.execute(f'SELECT {selected} FROM "{self.name}"')
        while rows := cursor.fetchmany(chunk_size):
            yield pd.DataFrame.from_records(
                [tuple(row) for row in rows], columns=columns
            )


class AnalysisReader:
    """Read retained data without reconstructing computations implicitly.

    :param source: Run backend or directory with settings.json/results.sqlite.
    :raises ValueError: If versions, configuration digest or run identities differ.
    """

    def __init__(self, source):
        if isinstance(source, SQLiteStorageBackend):
            self.backend = source
        else:
            path = Path(source).resolve()
            settings = json.loads((path / "settings.json").read_text(encoding="utf-8"))
            backend = object.__new__(SQLiteStorageBackend)
            backend.path, backend.settings, backend.lock = path, settings, RLock()
            backend.closed = False
            backend.connection = sqlite3.connect(
                (path / "results.sqlite").as_uri() + "?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            backend.connection.row_factory = sqlite3.Row
            self.backend = backend
        if self.backend.execute("PRAGMA user_version").fetchone()[0] != FORMAT_VERSION:
            raise ValueError("Unsupported run database version.")
        row = self.backend.execute("SELECT * FROM runs").fetchone()
        digest = hashlib.sha256(
            canonical_json(self.backend.settings).encode()
        ).hexdigest()
        if (
            row is None
            or row["settings_sha256"] != digest
            or row["run_id"] != self.backend.settings["run_id"]
        ):
            raise ValueError("Run settings/database identity mismatch.")
        logger.debug("Opened lazy run reader path=%s.", self.backend.path)

    @property
    def settings(self):
        """Return the resolved run manifest as independent containers."""
        return json.loads(canonical_json(self.backend.settings))

    @property
    def summary(self):
        """Return status and aggregate statistics without loading candidates."""
        row = self.backend.execute("SELECT status,summary_json FROM runs").fetchone()
        return {"status": row[0], **json.loads(row[1])}

    @property
    def tables(self):
        """Expose the shared logical tables through lazy TableHandle objects."""
        names = self.backend.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {row[0]: RunTableHandle(self.backend, row[0]) for row in names}

    def get_archive(self, objective=None):
        """Read latest evaluated sequence/objective entries, including removed rows."""
        statement = "SELECT objective,sequence,score,direction,evaluation_index,first_evaluation_index,lineage_json FROM evaluated_peptides"
        parameters = ()
        if objective is not None:
            statement += " WHERE objective=?"
            parameters = (objective,)
        statement += " ORDER BY first_evaluation_index,objective,sequence"
        cursor = self.backend.execute(statement, parameters)
        return pd.DataFrame.from_records(
            [tuple(row) for row in cursor],
            columns=[column[0] for column in cursor.description],
        )

    def get_final_batch(self):
        """Load the terminal batch only when its retention was enabled."""
        row = self.backend.execute(
            "SELECT payload_id FROM candidate_sets WHERE kind='terminal' ORDER BY set_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise DataUnavailableError(
                "Terminal batch was not retained; enable save_terminal."
            )
        return self.backend.load_payload(row[0])

    def get_snapshot(self, name):
        """Read all explicitly saved batches carrying the requested label."""
        rows = self.backend.execute(
            "SELECT payload_id FROM candidate_sets WHERE name=? ORDER BY set_id",
            (name,),
        ).fetchall()
        if not rows:
            raise DataUnavailableError(f"No retained snapshot named {name!r}.")
        return [self.backend.load_payload(row[0]) for row in rows]

    def get_history(self):
        """Read repeated observations only if their retention was requested."""
        if not self.settings["runner_settings"]["storage"]["save_observation_history"]:
            raise DataUnavailableError("Observation history was not retained.")
        return self._table_frame("observations")

    def get_lineage(self):
        """Return retained archive provenance, without inventing missing history."""
        if not self.settings["runner_settings"]["storage"]["save_lineage"]:
            raise DataUnavailableError("Lineage was not retained.")
        result = self.get_archive()
        result["lineage"] = result["lineage_json"].map(json.loads)
        return result

    def get_metadata(self):
        """Return configuration and run status metadata."""
        return {"settings": self.settings, "summary": self.summary}

    def get_best_candidate(self, objective=None):
        """Return the best retained evaluation for one explicitly selected objective."""
        archive = self.get_archive(objective)
        objectives = archive["objective"].unique()
        if len(objectives) > 1:
            raise ValueError(
                "Select an objective when multiple objectives were evaluated."
            )
        if archive.empty:
            return None
        direction = archive.iloc[0]["direction"]
        index = (
            archive["score"].idxmax()
            if direction == "maximize"
            else archive["score"].idxmin()
        )
        return archive.loc[index].to_dict()

    def get_component_data(self):
        """Read optional component diagnostics on explicit request."""
        if self.settings["runner_settings"]["storage"]["component_data"] == "none":
            raise DataUnavailableError("Component diagnostics were not retained.")
        result = self._table_frame("component_records")
        result["data"] = result["payload_id"].map(self.backend.load_payload)
        return result

    def _table_frame(self, name):
        table = self.tables[name]
        chunks = list(table.scan(table.columns))
        return (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=table.columns)
        )

    def get_checkpoints(self):
        """Return retained replay boundaries without loading their tensor payloads."""
        return self._table_frame("checkpoints")

    def get_boundary_input(self, execution_id):
        """Load the input CandidateBatch retained for one operation invocation.

        :param execution_id: Step execution identity from get_checkpoints().
        :return: Original aligned input batch with latent origins and fields.
        :raises DataUnavailableError: If no reconstruction boundary was retained.
        """
        row = self.backend.execute(
            "SELECT payload_id FROM checkpoints WHERE execution_id=?",
            (int(execution_id),),
        ).fetchone()
        if row is None:
            raise DataUnavailableError("No retained boundary for this execution.")
        return self.backend.load_payload(row[0])["batch"]

    def get_trajectories(self):
        """Load walker-produced sequences, metadata and aligned latent positions.

        :return: Pair of a metadata DataFrame and a tensor shaped (N, D).
        :raises DataUnavailableError: If trajectory retention was disabled.
        """
        import torch

        from pep_compass.data_schemas.candidates import (
            ObjectField,
            OptionalField,
            TensorField,
        )

        if (
            self.settings["runner_settings"]["storage"]["reconstruction"]
            != "trajectories"
        ):
            raise DataUnavailableError(
                "Enable reconstruction='trajectories' to retain walk points."
            )
        rows, latents = [], []
        cursor = self.backend.execute(
            "SELECT execution_id,payload_id FROM candidate_sets WHERE kind='trajectory' ORDER BY set_id"
        )
        for execution_id, payload_id in cursor:
            batch = self.backend.load_payload(payload_id)
            for index, sequence in enumerate(batch.sequences):
                record = {
                    "execution_id": execution_id,
                    "sequence": sequence,
                    "latent_index": len(rows),
                }
                for name in (
                    "tracking.trajectory_id",
                    "tracking.trajectory_index",
                    "tracking.trajectory_step",
                    "lineage.candidate_id",
                ):
                    field = batch.fields.get(name)
                    if isinstance(field, OptionalField):
                        field = field.values if bool(field.valid[index]) else None
                    if isinstance(field, (TensorField, ObjectField)):
                        value = field.values[index]
                        record[name.split(".")[-1]] = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                rows.append(record)
            latents.append(batch.latent_origins)
        frame = pd.DataFrame(
            rows,
            columns=[
                "execution_id",
                "sequence",
                "latent_index",
                "trajectory_id",
                "trajectory_index",
                "trajectory_step",
                "candidate_id",
            ],
        )
        return frame, torch.cat(latents, dim=0) if latents else torch.empty(
            (0, 0)
        )  # (N, D)

    def reconstruct_local_enumeration(self, checkpoint_id, *, autoencoder=None):
        """Re-execute one retained boundary and verify ordered output sequences."""
        from pep_compass.optimization.saving.replay import reconstruct

        return reconstruct(self, checkpoint_id, autoencoder=autoencoder)

    def close(self):
        """Release the reader-owned connection, including a memory database."""
        self.backend.close()

    def __enter__(self):
        """Return this lazy reader for deterministic resource cleanup."""
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False
