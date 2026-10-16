"""Retention-aware consumer of the existing computation lifecycle."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from time import perf_counter

import numpy as np
import torch

from pep_compass.optimization.tracking import StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def sequence_digest(sequences):
    """Hash ordered sequences with the established replay encoding."""
    raw = json.dumps(list(sequences), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def rng_snapshot(context):
    """Capture every supported random source at a replay boundary."""
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_context": context.rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


class Saver(StepTracker):
    """Collect observations and optional replay data without changing batches.

    :param backend: Memory or SQLite backend with the shared logical schema.
    :param settings: Validated StorageSettings.
    """

    def __init__(self, backend, settings):
        self.backend = backend
        self.settings = settings
        self.started_at = perf_counter()
        self._next_execution = 0
        self._stack = []
        self._iterations = 0
        self._last_iteration_time = {}
        self._iteration_starts = {}
        self._evaluations = 0
        self.closed = False

    def begin_step(self, step, batch, scope, context):
        """Begin a measured invocation and optionally capture its replay input."""
        from pep_compass.steps.flow.strategies.local_enumeration import (
            LocalEnumeration,
        )

        identity = self._next_execution
        self._next_execution += 1
        parent = self._stack[-1] if self._stack else None
        self._stack.append(identity)
        handle = {"id": identity, "started": perf_counter(), "checkpoint": None}
        self.backend.execute(
            "INSERT INTO step_executions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                context.node_id,
                parent,
                json.dumps(scope.__dict__),
                len(batch),
                None,
                None,
                "running",
                "prepare_call",
                None,
            ),
        )
        try:
            if self.settings.reconstruction != "none" and isinstance(
                step, LocalEnumeration
            ):
                state = {
                    field.name: getattr(context.state, field.name)
                    for field in fields(context.state)
                    if field.name != "observer"
                }
                payload = self.backend.payload(
                    {
                        "batch": batch,
                        "state": state,
                        "rng": rng_snapshot(context),
                        "seed": context.seed,
                        "scope": scope,
                        "step_state": context.step_state,
                    }
                )
                cursor = self.backend.execute(
                    "INSERT INTO checkpoints(node_id,execution_id,scope_json,phase,payload_id) VALUES (?,?,?,?,?)",
                    (
                        context.node_id,
                        identity,
                        json.dumps(scope.__dict__),
                        "before_prepare_call",
                        payload,
                    ),
                )
                handle["checkpoint"] = cursor.lastrowid
        except BaseException as error:
            self._stack.pop()
            try:
                self.backend.execute(
                    "UPDATE step_executions SET status='failed',phase='checkpoint',error=? WHERE execution_id=?",
                    (f"{type(error).__name__}: {error}", identity),
                )
            except Exception:
                logger.error("Could not record the checkpoint failure.", exc_info=True)
            raise
        return handle

    def end_step(self, handle, step, input_batch, output_batch, scope, context):
        """Complete metadata and optional component/trajectory snapshots."""

        self.backend.execute(
            "UPDATE step_executions SET output_count=?,duration_seconds=?,status='completed',phase='execute' WHERE execution_id=?",
            (len(output_batch), perf_counter() - handle["started"], handle["id"]),
        )
        if handle["checkpoint"] is not None:
            self.backend.execute(
                "UPDATE checkpoints SET output_count=?,output_sha256=? WHERE checkpoint_id=?",
                (
                    len(output_batch),
                    sequence_digest(output_batch.sequences),
                    handle["checkpoint"],
                ),
            )
        if self.settings.reconstruction == "trajectories" and step.category == "walker":
            self.capture(
                output_batch,
                name="trajectory_point",
                context=context,
                kind="trajectory",
            )
        if self.settings.component_data != "none":
            value = step.display_info()
            if self.settings.component_data == "full":
                value = {"display": value, "fields": output_batch.fields}
            payload = self.backend.payload(value)
            self.backend.execute(
                "INSERT INTO component_records(node_id,execution_id,name,payload_id) VALUES (?,?,?,?)",
                (context.node_id, handle["id"], step.name, payload),
            )
        self._finish_handle(handle)

    def execution_started(self, handle):
        """Separate invocation preparation from the transformation phase."""
        self.backend.execute(
            "UPDATE step_executions SET phase='execute' WHERE execution_id=?",
            (handle["id"],),
        )

    def record_precompute(self, step, context, duration, error):
        """Record run-static preparation even when it fails before invocation."""
        identity = self._next_execution
        self._next_execution += 1
        self.backend.execute(
            "INSERT INTO step_executions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                identity,
                context.node_id,
                None,
                json.dumps(context.scope.__dict__),
                0,
                0,
                duration,
                "failed" if error is not None else "completed",
                "precompute",
                None if error is None else f"{type(error).__name__}: {error}",
            ),
        )

    def fail_step(self, handle, step, input_batch, scope, context, error):
        """Persist failure without translating the original exception."""
        self.backend.execute(
            "UPDATE step_executions SET duration_seconds=?,status='failed',error=? WHERE execution_id=?",
            (
                perf_counter() - handle["started"],
                f"{type(error).__name__}: {error}",
                handle["id"],
            ),
        )
        self._finish_handle(handle)

    def _finish_handle(self, handle):
        if self._stack and self._stack[-1] == handle["id"]:
            self._stack.pop()
        elif handle["id"] in self._stack:
            raise RuntimeError("Unbalanced step lifecycle events.")

    def observations(
        self, objective, sequences, scores, latents, *, direction, lineage
    ):
        """Preserve every evaluated sequence independently of later selection."""
        for index, (sequence, score) in enumerate(zip(sequences, scores, strict=True)):
            provenance = (
                {name: values[index] for name, values in (lineage or {}).items()}
                if self.settings.save_lineage
                else {}
            )
            encoded_lineage = json.dumps(provenance)
            self.backend.execute(
                "INSERT OR IGNORE INTO peptides VALUES (?)", (sequence,)
            )
            self.backend.execute(
                "INSERT INTO evaluated_peptides VALUES (?,?,?,?,?,?,?) ON CONFLICT(objective,sequence) DO UPDATE SET score=excluded.score,direction=excluded.direction,evaluation_index=excluded.evaluation_index,lineage_json=excluded.lineage_json",
                (
                    objective,
                    sequence,
                    float(score),
                    direction,
                    self._evaluations,
                    self._evaluations,
                    encoded_lineage,
                ),
            )
            if self.settings.save_observation_history:
                self.backend.execute(
                    "INSERT INTO observations(objective,sequence,score,direction,lineage_json) VALUES (?,?,?,?,?)",
                    (objective, sequence, float(score), direction, encoded_lineage),
                )
            self._evaluations += 1

    def capture(self, batch, *, name, context=None, kind="explicit"):
        """Snapshot a requested batch snapshot; explicit requests override base retention."""
        payload = self.backend.payload(batch)
        self.backend.execute(
            "INSERT INTO candidate_sets(name,node_id,execution_id,kind,payload_id) VALUES (?,?,?,?,?)",
            (
                name,
                context.node_id if context is not None else None,
                self._stack[-1] if self._stack else None,
                kind,
                payload,
            ),
        )

    def iteration_completed(self, context, index, trajectories):
        """Measure one operator event, rather than summing trajectory counts."""
        now = perf_counter()
        key = (context.node_id, context.scope.loop_indices, context.scope.branch_names)
        previous = self._last_iteration_time.get(key)
        started = self._iteration_starts.pop((*key, index), now)
        self._last_iteration_time[key] = now
        self.backend.execute(
            "INSERT INTO iterations(node_id,iteration_index,scope_json,trajectories,elapsed_seconds,duration_seconds,interval_seconds) VALUES (?,?,?,?,?,?,?)",
            (
                context.node_id,
                index,
                json.dumps(context.scope.__dict__),
                trajectories,
                now - self.started_at,
                now - started,
                None if previous is None else now - previous,
            ),
        )
        self._iterations += 1
        if self._iterations % self.settings.flush_every == 0:
            self.backend.flush()

    def iteration_started(self, context, index):
        """Mark body start separately from inter-iteration completion intervals."""
        key = (
            context.node_id,
            context.scope.loop_indices,
            context.scope.branch_names,
            index,
        )
        self._iteration_starts[key] = perf_counter()

    def finalize(self, result, state, *, status="completed", error=None):
        """Persist terminal summary and commit the final incomplete buffer."""
        if result is not None and self.settings.save_terminal:
            self.capture(result.terminal_candidates, name="terminal", kind="terminal")
        summary = {
            "duration_seconds": perf_counter() - self.started_at,
            "oracle_calls": state.oracle_calls,
            "generated_candidates": state.generated_candidates,
            "completed_iterations": state.completed_iterations,
            "iteration_events": self._iterations,
            "best_sequence": result.best_candidate.sequence
            if result is not None and result.best_candidate is not None
            else None,
            "best_score": result.best_score if result is not None else None,
            "objective_name": result.objective_name if result is not None else None,
            "terminal_count": len(result.terminal_candidates)
            if result is not None
            else None,
            "error": error,
        }
        self.backend.execute(
            "UPDATE runs SET status=?, summary_json=?",
            (status, json.dumps(summary, allow_nan=False)),
        )
        self.close()
        return summary

    def close(self):
        """Flush recording; the reader owns the remaining backend lifetime."""
        if not self.closed:
            self.backend.flush()
            self.closed = True
