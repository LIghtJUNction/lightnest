"""Bounded, audited editing of a program through repeated proposals.

The proposer sees the current program and prior public feedback.  It never gets
an expected final program from this module.  A validator may allow a legal
intermediate program without declaring the task complete.  Final held-out tests
belong outside this loop.

This is an action runner, not a learner or a process sandbox.  A descendant
inherits accepted program bytes; model-weight inheritance must be measured by
the caller separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Callable


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ByteEdit:
    """Replace an exact span in the current program; offsets are not historical."""

    offset: int
    expected: bytes
    replacement: bytes

    def to_dict(self) -> dict:
        return {"offset": self.offset, "expected_hex": self.expected.hex(),
                "replacement_hex": self.replacement.hex()}


@dataclass(frozen=True)
class PatchDecision:
    edits: tuple[ByteEdit, ...] = ()
    stop: bool = False
    reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {"edits": [edit.to_dict() for edit in self.edits], "stop": self.stop,
                "reason": self.reason, "confidence": self.confidence}


@dataclass(frozen=True)
class ValidationDecision:
    """Acceptance and task completion are deliberately separate decisions."""

    accept: bool
    complete: bool = False
    score: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"accept": self.accept, "complete": self.complete,
                "score": self.score, "reason": self.reason}


@dataclass(frozen=True)
class Transition:
    step: int
    before_sha256: str
    candidate_sha256: str | None
    after_sha256: str
    decision: PatchDecision | None
    accepted: bool
    status: str
    validation: ValidationDecision | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {"step": self.step, "before_sha256": self.before_sha256,
                "candidate_sha256": self.candidate_sha256,
                "after_sha256": self.after_sha256,
                "decision": self.decision.to_dict() if self.decision else None,
                "accepted": self.accepted, "status": self.status,
                "validation": self.validation.to_dict() if self.validation else None,
                "error": self.error}


@dataclass(frozen=True)
class EpisodeResult:
    request: str
    initial: bytes = field(repr=False)
    final: bytes = field(repr=False)
    history: tuple[Transition, ...]
    stop_reason: str
    complete: bool
    predictor_id: str
    max_steps: int
    max_total_changed_bytes: int
    max_output_bytes: int
    parent_episode_sha256: str | None = None
    inherited_history: tuple[Transition, ...] = field(default=(), repr=False)

    @property
    def accepted_steps(self) -> int:
        return sum(transition.accepted for transition in self.history)

    def audit(self) -> dict:
        return {"schema": "lightnest.iterative_patch.v1", "request": self.request,
                "predictor_id": self.predictor_id,
                "initial_sha256": sha256(self.initial), "initial_bytes": len(self.initial),
                "final_sha256": sha256(self.final), "final_bytes": len(self.final),
                "parent_episode_sha256": self.parent_episode_sha256,
                "inherited_steps": len(self.inherited_history),
                "bounds": {"max_steps": self.max_steps,
                           "max_total_changed_bytes": self.max_total_changed_bytes,
                           "max_output_bytes": self.max_output_bytes},
                "stop_reason": self.stop_reason, "complete": self.complete,
                "accepted_steps": self.accepted_steps,
                "transitions": [transition.to_dict() for transition in self.history]}

    @property
    def audit_sha256(self) -> str:
        encoded = json.dumps(self.audit(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return sha256(encoded)


Predictor = Callable[[str, bytes, int, tuple[Transition, ...]], PatchDecision]
Validator = Callable[[str, bytes, bytes, PatchDecision], ValidationDecision]


def _check_decision(decision: PatchDecision) -> None:
    if not isinstance(decision, PatchDecision):
        raise ValueError("predictor must return PatchDecision")
    if type(decision.stop) is not bool or not isinstance(decision.reason, str):
        raise ValueError("invalid decision metadata")
    if not isinstance(decision.edits, tuple) or any(not isinstance(e, ByteEdit) for e in decision.edits):
        raise ValueError("edits must be a tuple of ByteEdit")
    if any(type(e.offset) is not int or not isinstance(e.expected, bytes)
           or not isinstance(e.replacement, bytes) for e in decision.edits):
        raise ValueError("edit metadata must contain an integer offset and byte spans")
    if decision.confidence is not None:
        if (type(decision.confidence) not in (int, float)
                or not math.isfinite(decision.confidence)
                or not 0 <= decision.confidence <= 1):
            raise ValueError("confidence must be finite and in [0,1]")
    if decision.stop == bool(decision.edits):
        raise ValueError("decision must either stop or contain edits")


def apply_edits(current: bytes, edits: tuple[ByteEdit, ...], *, max_output_bytes: int) -> bytes:
    """Apply non-overlapping spans atomically after checking all preconditions."""
    if not isinstance(current, bytes) or not isinstance(edits, tuple) or not edits:
        raise ValueError("expected bytes and a nonempty tuple of edits")
    if type(max_output_bytes) is not int or max_output_bytes < 0:
        raise ValueError("max_output_bytes must be a nonnegative integer")
    for edit in edits:
        if (not isinstance(edit, ByteEdit) or type(edit.offset) is not int
                or not isinstance(edit.expected, bytes) or not isinstance(edit.replacement, bytes)):
            raise ValueError("invalid edit types")
        if edit.offset < 0 or edit.offset + len(edit.expected) > len(current):
            raise ValueError("edit outside current program")
        if edit.expected == edit.replacement:
            raise ValueError("unchanged edit")
        if current[edit.offset:edit.offset + len(edit.expected)] != edit.expected:
            raise ValueError("stale edit precondition")
    ordered = sorted(edits, key=lambda e: e.offset)
    previous_offset = -1
    previous_end = -1
    for edit in ordered:
        if edit.offset == previous_offset or edit.offset < previous_end:
            raise ValueError("overlapping or ambiguous edits")
        previous_offset = edit.offset
        previous_end = edit.offset + len(edit.expected)
    size = len(current) + sum(len(e.replacement) - len(e.expected) for e in edits)
    if size < 0 or size > max_output_bytes:
        raise ValueError("output byte budget exceeded")
    # Construct only after every precondition passes; failures leave current intact.
    parts = []
    cursor = 0
    for edit in ordered:
        parts.extend((current[cursor:edit.offset], edit.replacement))
        cursor = edit.offset + len(edit.expected)
    parts.append(current[cursor:])
    return b"".join(parts)


def run_iterative_patch(request: str, current: bytes, predictor: Predictor, *,
                        validator: Validator | None = None, max_steps: int = 8,
                        max_total_changed_bytes: int = 64, max_output_bytes: int = 1 << 20,
                        predictor_id: str = "unspecified",
                        parent: EpisodeResult | None = None) -> EpisodeResult:
    """Run a bounded episode, re-reading accepted bytes before each proposal.

    Rejected edits consume a step but do not change the current program or the
    changed-byte budget.  The budget counts max(old span size, new span size)
    per accepted edit, including edits that touch the same site more than once.
    STOP alone does not assert semantic success.  Passing a parent binds this
    episode to its exact accepted bytes and complete prior public history.
    """
    if not isinstance(request, str) or not isinstance(current, bytes):
        raise ValueError("expected a request string and program bytes")
    for name, value in (("max_steps", max_steps), ("max_total_changed_bytes", max_total_changed_bytes),
                        ("max_output_bytes", max_output_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if len(current) > max_output_bytes:
        raise ValueError("initial program exceeds output byte budget")
    if not isinstance(predictor_id, str):
        raise ValueError("predictor_id must be a string")
    if parent is not None and (not isinstance(parent, EpisodeResult) or current != parent.final):
        raise ValueError("child must inherit the exact accepted parent bytes")
    inherited = parent.inherited_history + parent.history if parent else ()
    initial = current
    transitions = []
    seen = {sha256(current)}
    if parent:
        seen.add(sha256(parent.initial))
        seen.update(t.after_sha256 for t in inherited if t.accepted)
    changed = 0
    complete = False
    stop_reason = "step_budget"
    for local_step in range(max_steps):
        step = len(inherited) + local_step
        before_hash = sha256(current)
        decision = None
        candidate_hash = None
        validation = None
        try:
            decision = predictor(request, current, step, inherited + tuple(transitions))
            _check_decision(decision)
        except Exception as error:
            # Invalid objects cannot be serialized as a valid PatchDecision.
            transitions.append(Transition(step, before_hash, None, before_hash, None, False,
                                          "predictor_error", error=f"{type(error).__name__}: {error}"[:1000]))
            stop_reason = "predictor_error"
            break
        if decision.stop:
            transitions.append(Transition(step, before_hash, None, before_hash, decision,
                                          False, "stopped"))
            stop_reason = "predictor_stop"
            break
        try:
            candidate = apply_edits(current, decision.edits, max_output_bytes=max_output_bytes)
            candidate_hash = sha256(candidate)
            cost = sum(max(len(e.expected), len(e.replacement)) for e in decision.edits)
            if changed + cost > max_total_changed_bytes:
                raise ValueError("cumulative changed-byte budget exceeded")
            if candidate_hash in seen:
                raise ValueError("previous accepted state would be revisited")
            validation = (validator(request, current, candidate, decision) if validator else
                          ValidationDecision(True, reason="structural checks only"))
            if (not isinstance(validation, ValidationDecision) or type(validation.accept) is not bool
                    or type(validation.complete) is not bool or not isinstance(validation.reason, str)
                    or (validation.complete and not validation.accept)):
                raise ValueError("invalid validation decision")
            if validation.score is not None and (type(validation.score) not in (int, float)
                                                or not math.isfinite(validation.score)):
                raise ValueError("validation score must be finite")
            if validation.accept:
                current = candidate
                changed += cost
                seen.add(candidate_hash)
            transitions.append(Transition(step, before_hash, candidate_hash, sha256(current),
                                          decision, validation.accept,
                                          "accepted" if validation.accept else "rejected", validation))
            if validation.complete:
                complete = True
                stop_reason = "validator_complete"
                break
        except Exception as error:
            # The candidate is discarded.  The previous accepted state survives.
            transitions.append(Transition(step, before_hash, candidate_hash, before_hash,
                                          decision, False, "rejected_error", None,
                                          f"{type(error).__name__}: {error}"[:1000]))
    return EpisodeResult(request, initial, current, tuple(transitions), stop_reason, complete,
                         predictor_id, max_steps, max_total_changed_bytes, max_output_bytes,
                         parent.audit_sha256 if parent else None, inherited)


def replay_audit(initial: bytes, audit: dict) -> bytes:
    """Check and replay accepted byte transitions without calling the proposer.

    This verifies inheritance of program content.  It cannot re-establish the
    truth of a compiler/test result; keep the external validation logs as well.
    """
    if audit.get("schema") != "lightnest.iterative_patch.v1":
        raise ValueError("unsupported audit schema")
    if sha256(initial) != audit["initial_sha256"] or len(initial) != audit["initial_bytes"]:
        raise ValueError("initial audit state mismatch")
    current = initial
    accepted = 0
    for transition in audit["transitions"]:
        if sha256(current) != transition["before_sha256"]:
            raise ValueError("broken transition parent hash")
        record = transition["decision"]
        candidate = None
        if record and record["edits"]:
            edits = tuple(ByteEdit(e["offset"], bytes.fromhex(e["expected_hex"]),
                                   bytes.fromhex(e["replacement_hex"])) for e in record["edits"])
            try:
                candidate = apply_edits(current, edits, max_output_bytes=audit["bounds"]["max_output_bytes"])
            except ValueError:
                if transition["accepted"] or transition["candidate_sha256"] is not None:
                    raise ValueError("invalid edit marked as constructed or accepted") from None
            else:
                if sha256(candidate) != transition["candidate_sha256"]:
                    raise ValueError("candidate hash mismatch")
        if transition["accepted"]:
            if candidate is None or not (transition.get("validation") or {}).get("accept", False):
                raise ValueError("accepted transition has no valid candidate or acceptance record")
            current = candidate
            accepted += 1
        if sha256(current) != transition["after_sha256"]:
            raise ValueError("broken transition result hash")
    if accepted != audit["accepted_steps"] or sha256(current) != audit["final_sha256"] or len(current) != audit["final_bytes"]:
        raise ValueError("final audit state mismatch")
    return current
