"""Executable outcome feedback for reinforcement-style candidate learning.

The environment reward is computed only from visible program cases. Hidden cases,
clean target bytes and mutation metadata never enter this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .artifact import inspect_elf
from .execute import Case, same_result
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .iterate import Example


def transition_reward(passed_before: int, passed_after: int, valid_before: bool, valid_after: bool) -> float:
    """Dense scalar used by search and policy-gradient training."""
    reward = 1.5 * (passed_after - passed_before) - .75 * (passed_before - passed_after)
    reward += .5 * int(valid_after) - .5 * int(valid_before and not valid_after)
    return reward + 1.0 * int(valid_after and passed_after > 0)


@dataclass(frozen=True)
class Feedback:
    reward: float
    passed_before: int
    passed_after: int
    status_counts: dict[str, int]
    error_kinds: tuple[str, ...]
    valid_elf_before: bool
    valid_elf_after: bool


def _valid(raw: bytes) -> bool:
    try:
        info = inspect_elf(raw)
        return (info["machine"] == "EM_X86_64" and info["type"] in ("ET_EXEC", "ET_DYN")
                and info["entry_in_executable_segment"])
    except ValueError:
        return False


def _errors(result: dict) -> str:
    if result["status"] == "invalid_artifact": return "invalid_entry_or_elf"
    if result["status"] == "timeout": return "timeout"
    if result["status"] == "isolation_error": return "isolation_error"
    if result["returncode"] is not None and result["returncode"] < 0: return f"signal_{-result['returncode']}"
    if result["status"] != "ok": return result["status"]
    if result["stderr"]: return "stderr"
    if result["returncode"] not in (0,): return f"exit_{result['returncode']}"
    return "ok"


def outcome_reward(before: bytes, after: bytes, examples: list[Example], runner) -> Feedback:
    """Return dense progress reward for one candidate transition.

    A candidate gets credit for each newly passing visible case, a small penalty
    for lost cases, and a structural/safety bonus. Runtime failures are typed so
    the learner can distinguish malformed ELF, timeout, signal and diagnostic
    output rather than treating all failures as one scalar.
    """
    if not examples or len(examples) > 16: raise ValueError("visible examples must be 1..16")
    old = [runner.run(before, e.case) for e in examples]
    new = [runner.run(after, e.case) for e in examples]
    old_pass = sum(same_result(r, e.expected) for r, e in zip(old, examples, strict=True))
    new_pass = sum(same_result(r, e.expected) for r, e in zip(new, examples, strict=True))
    old_valid, new_valid = _valid(before), _valid(after)
    errors = tuple(sorted({_errors(r) for r in new}))
    statuses = {kind: sum(_errors(r) == kind for r in new) for kind in sorted({_errors(r) for r in new})}
    reward = transition_reward(old_pass, new_pass, old_valid, new_valid)
    reward -= 0.25 * sum(kind in {"timeout", "signal_11", "invalid_entry_or_elf"} for kind in errors)
    reward += 1.0 * int(new_pass == len(examples) and new_valid)
    return Feedback(reward, old_pass, new_pass, statuses, errors, old_valid, new_valid)


def reinforce_loss(output: dict, action: dict, reward: float, *, baseline: float = 0.) -> torch.Tensor:
    """REINFORCE term for an executed action, with detached scalar reward."""
    logits, bits = output["logits"], output["bits"]
    index = int(action["offset"])
    position_logprob = logits.log_softmax(0)[index]
    target = torch.tensor([(action["value"] >> bit) & 1 for bit in range(8)], device=bits.device, dtype=bits.dtype)
    bit_logprob = torch.distributions.Bernoulli(logits=bits[index]).log_prob(target).sum()
    stop_logprob = torch.nn.functional.logsigmoid(-logits[index]) if action["stop"] else torch.nn.functional.logsigmoid(logits[index])
    advantage = torch.as_tensor(float(reward - baseline), device=logits.device, dtype=logits.dtype)
    return -(advantage.detach() * (position_logprob + bit_logprob + stop_logprob))
