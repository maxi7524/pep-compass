"""Reusable catalog models for local-enumeration trajectories and candidates."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import numpy as np
    import torch


def normalize_tracking_level(level: str) -> str:
    """Return the internal level for canonical and legacy tracking names.

    :param level: ``quick``, ``normal``, ``long`` or a legacy equivalent.
    :return: Internal ``short``, ``normal``, or ``all`` level.
    :raises ValueError: If ``level`` is unsupported.
    """
    aliases = {
        "quick": "short",
        "short": "short",
        "normal": "normal",
        "long": "all",
        "all": "all",
    }
    try:
        return aliases[level]
    except KeyError as error:
        raise ValueError(
            "tracking level must be quick, normal, or long"
        ) from error


def serialize_tensor(value: "torch.Tensor | np.ndarray | None") -> str:
    """Serialize a tensor-like value as a compact JSON array.

    :param value: Tensor or array to detach and transfer to CPU when required.
    :return: Compact JSON representation, or an empty string for ``None``.
    """
    if value is None:
        return ""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, separators=(",", ":"))


@dataclass(slots=True)
class CandidateOccurrence:
    """Describe one unique candidate occurrence at one walker step.

    :param sequence: Candidate peptide sequence.
    :param parent_sequence: Walker sequence mutated to produce the candidate.
    :param trajectory_id: Zero-based walker trajectory identifier.
    :param step_id: One-based step identifier within the trajectory.
    :param node_id: Stable identifier of this occurrence.
    :param parent_id: Identifier of the generating walker step.
    :param passed_method_filter: Whether the method-specific filter retained it.
    :param passed_constraint_filter: Whether the final constraint retained it.
    :param method_score: Raw method-specific score when available.
    :param method_probability: Softmax probability when available.
    :param method_cumulative_probability: Cumulative probability in score order.
    :param method_rank: One-based rank when available.
    :param levenshtein_to_center: Distance to the enumeration centre.
    :param levenshtein_to_parent: Distance to the generating walker sequence.
    """

    sequence: str
    parent_sequence: str
    trajectory_id: int | None
    step_id: int | None
    source_iteration_id: int | None = None
    node_id: str = ""
    parent_id: str = ""
    passed_method_filter: bool = True
    passed_constraint_filter: bool = True
    method_score: float | None = None
    method_probability: float | None = None
    method_cumulative_probability: float | None = None
    method_rank: int | None = None
    levenshtein_to_center: int | None = None
    levenshtein_to_parent: int | None = None


class CandidateOccurrenceSpool:
    """Spill candidate occurrences to disk instead of retaining an event list."""

    def __init__(self, maximum_memory_bytes: int = 1_048_576) -> None:
        self.count = 0
        self._events = tempfile.SpooledTemporaryFile(  # noqa: SIM115
            mode="w+", max_size=maximum_memory_bytes, encoding="utf-8"
        )

    def append(self, occurrence: CandidateOccurrence) -> None:
        """Append one compact occurrence to the spool.

        :param occurrence: Candidate occurrence to persist temporarily.
        """
        self._events.write(json.dumps(asdict(occurrence), separators=(",", ":")))
        self._events.write("\n")
        self.count += 1

    def __iter__(self) -> Iterator[CandidateOccurrence]:
        self._events.seek(0)
        for line in self._events:
            yield CandidateOccurrence(**json.loads(line))

    def close(self) -> None:
        """Close and remove the temporary backing file."""
        self._events.close()


@dataclass(slots=True)
class WalkerStepRecord:
    """Store geometry and stage counts for one SORBES walker step."""

    trajectory_id: int | None
    step_id: int
    node_id: str
    parent_id: str
    parent_sequence: str
    next_sequence: str
    theoretical_product_count: int
    post_cap_event_count: int
    post_cap_unique_count: int
    post_method_filter_event_count: int
    post_method_filter_unique_count: int
    post_constraint_event_count: int
    post_constraint_unique_count: int
    mutang_position_count: int
    mutang_residue_option_count: int
    latent_dimension: int
    effective_dimension: int
    current_latent_position: str = ""
    next_latent_position: str = ""
    singular_values: str = ""
    adjusted_time_step: float | None = None


@dataclass
class LocalEnumerationTrace:
    """Collect one local-enumeration call without owning output serialization."""

    generated_count: int = 0
    accepted_count: int = 0
    candidates: dict[str, CandidateOccurrence] = field(default_factory=dict)
    steps: list[WalkerStepRecord] = field(default_factory=list)
    all_candidates: CandidateOccurrenceSpool = field(
        default_factory=CandidateOccurrenceSpool
    )
    post_cap_sequences: set[str] = field(default_factory=set)
    post_method_filter_sequences: set[str] = field(default_factory=set)
    post_constraint_sequences: set[str] = field(default_factory=set)
    post_cap_event_count: int = 0
    post_method_filter_event_count: int = 0
    post_constraint_event_count: int = 0

class LocalEnumerationCatalog:
    """Catalog walker steps and candidate stages independently of file formats.

    The catalog is intentionally unaware of LEBO, APEX, CSV paths, and runner
    manifests. It can therefore be reused by a future enumerator implementation
    or embedded as a standalone library component.
    """

    def __init__(self, level: str, store_walker_latents: bool) -> None:
        """Initialize an empty local-enumeration catalog.

        :param level: Tracking mode: ``quick``, ``normal``, or ``long``.
        :param store_walker_latents: Whether walker geometry arrays are retained.
        :raises ValueError: If the tracking level is unsupported.
        """
        self.level = normalize_tracking_level(level)
        self.store_walker_latents = store_walker_latents
        self.trace = LocalEnumerationTrace()

    def record_step(
        self,
        *,
        trajectory_id: int,
        step_id: int,
        parent_sequence: str,
        next_sequence: str,
        current_latent_position: "torch.Tensor",
        next_latent_position: "torch.Tensor",
        singular_values: "torch.Tensor",
        effective_dimension: int,
        adjusted_time_step: float,
        mutations: dict[int, list[int]],
        theoretical_product_count: int,
        post_cap_candidates: list[str],
        post_method_filter_candidates: list[str],
        post_constraint_candidates: list[str],
        method_scores: dict[
            str,
            tuple[float | None, float | None, float | None, int | None],
        ],
        center_sequence: str,
    ) -> None:
        """Record one walker step and its stage-aligned unique candidates.

        :param trajectory_id: Zero-based walker trajectory identifier.
        :param step_id: One-based step within the trajectory.
        :param parent_sequence: Sequence mutated at this step.
        :param next_sequence: Sequence decoded from the next walker position.
        :param current_latent_position: Actual SORBES position before the step.
        :param next_latent_position: Actual SORBES position after the step.
        :param singular_values: Local decoder-Jacobian singular values.
        :param effective_dimension: Horizontal latent dimension at this step.
        :param adjusted_time_step: Time increment selected by SORBES.
        :param mutations: Valid MUTANG residue options grouped by position.
        :param theoretical_product_count: Product size before the candidate cap.
        :param post_cap_candidates: Unique sequences materialized after the cap.
        :param post_method_filter_candidates: Candidates retained by the method.
        :param post_constraint_candidates: Candidates retained by constraints.
        :param method_scores: Per-sequence score, probability, cumulative mass,
            and rank diagnostics.
        :param center_sequence: Original local-enumeration centre.
        """
        post_cap = list(dict.fromkeys(post_cap_candidates))
        post_method = list(dict.fromkeys(post_method_filter_candidates))
        post_constraint = list(dict.fromkeys(post_constraint_candidates))
        method_set = set(post_method)
        constraint_set = set(post_constraint)

        if self.level == "short":
            return

        self.trace.post_cap_event_count += len(post_cap_candidates)
        self.trace.post_method_filter_event_count += len(
            post_method_filter_candidates
        )
        self.trace.post_constraint_event_count += len(post_constraint_candidates)
        self.trace.generated_count += len(post_cap_candidates)
        self.trace.accepted_count += len(post_constraint_candidates)
        self.trace.post_cap_sequences.update(post_cap)
        self.trace.post_method_filter_sequences.update(post_method)
        self.trace.post_constraint_sequences.update(post_constraint)

        node_id = f"trajectory_{trajectory_id}_step_{step_id}"
        parent_id = (
            f"trajectory_{trajectory_id}_step_{step_id - 1}"
            if step_id > 1
            else f"trajectory_{trajectory_id}_root"
        )
        latent_dimension = int(current_latent_position.numel())
        self.trace.steps.append(
            WalkerStepRecord(
                trajectory_id=trajectory_id,
                step_id=step_id,
                node_id=node_id,
                parent_id=parent_id,
                parent_sequence=parent_sequence,
                next_sequence=next_sequence,
                theoretical_product_count=theoretical_product_count,
                post_cap_event_count=len(post_cap_candidates),
                post_cap_unique_count=len(post_cap),
                post_method_filter_event_count=len(post_method_filter_candidates),
                post_method_filter_unique_count=len(post_method),
                post_constraint_event_count=len(post_constraint_candidates),
                post_constraint_unique_count=len(post_constraint),
                mutang_position_count=len(mutations),
                mutang_residue_option_count=sum(map(len, mutations.values())),
                latent_dimension=latent_dimension,
                effective_dimension=effective_dimension,
                current_latent_position=(
                    serialize_tensor(current_latent_position)
                    if self.store_walker_latents
                    else ""
                ),
                next_latent_position=(
                    serialize_tensor(next_latent_position)
                    if self.store_walker_latents
                    else ""
                ),
                singular_values=(
                    serialize_tensor(singular_values)
                    if self.store_walker_latents
                    else ""
                ),
                adjusted_time_step=adjusted_time_step,
            )
        )

        for candidate_index, sequence in enumerate(post_cap):
            score, probability, cumulative, rank = method_scores.get(
                sequence, (None, None, None, None)
            )
            occurrence = CandidateOccurrence(
                sequence=sequence,
                parent_sequence=parent_sequence,
                trajectory_id=trajectory_id,
                step_id=step_id,
                node_id=f"{node_id}_candidate_{candidate_index}",
                parent_id=node_id,
                passed_method_filter=sequence in method_set,
                passed_constraint_filter=sequence in constraint_set,
                method_score=score,
                method_probability=probability,
                method_cumulative_probability=cumulative,
                method_rank=rank,
                levenshtein_to_center=_levenshtein(sequence, center_sequence),
                levenshtein_to_parent=_levenshtein(sequence, parent_sequence),
            )
            if sequence in constraint_set:
                self.trace.candidates.setdefault(sequence, occurrence)
            if self.level == "all" or sequence in constraint_set:
                self.trace.all_candidates.append(occurrence)


def _levenshtein(first: str, second: str) -> int:
    """Return Levenshtein distance without exposing the dependency to callers."""
    import Levenshtein

    return int(Levenshtein.distance(first, second))


# Compatibility aliases for the current optimizer integration.
CandidateProvenance = CandidateOccurrence
CandidateEventSpool = CandidateOccurrenceSpool
EnumerationStep = WalkerStepRecord
EnumerationTrace = LocalEnumerationTrace
