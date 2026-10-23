"""Bounded RGB edit/observe loop; temporary proposals never overwrite inputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

import torch

from .artifact import BytePatch, apply_patches, digest, inspect_elf
from .execute import Case, same_result
from .model import greedy_action
from .observe import Frame
from .feedback import transition_reward


@dataclass(frozen=True)
class Example:
    case: Case
    expected: dict

    def __post_init__(self):
        if not isinstance(self.case, Case) or set(self.expected) != {"status", "returncode", "stdout", "stderr"}:
            raise ValueError("visible examples require input and observable output only")
        if self.expected["status"] != "ok" or type(self.expected["returncode"]) is not int:
            raise ValueError("visible reference execution must be completed")
        bytes.fromhex(self.expected["stdout"]); bytes.fromhex(self.expected["stderr"])


def execution_frames(examples, results):
    """Same visible evidence representation during training and inference."""
    frames = []
    for i, (example, actual) in enumerate(zip(examples, results, strict=True)):
        base = i * (1 << 24)
        for kind, payload in (("stdin", example.case.stdin),
                              ("arguments", json.dumps(example.case.args).encode()),
                              ("stdout", bytes.fromhex(actual["stdout"])),
                              ("stderr", bytes.fromhex(actual["stderr"])),
                              ("required_stdout", bytes.fromhex(example.expected["stdout"])),
                              ("required_stderr", bytes.fromhex(example.expected["stderr"])),
                              ("process_result", json.dumps({k: actual[k] for k in ("status", "returncode")}, sort_keys=True).encode()),
                              ("required_result", json.dumps({k: example.expected[k] for k in ("status", "returncode")}, sort_keys=True).encode())):
            frames.append(Frame(kind, payload, base))
    return frames


class RGBEditor:
    def __init__(self, policy, runner, *, max_edits=8, max_candidates=32, wall_seconds=10., blue_risk=.5):
        if policy.training:
            raise ValueError("editor requires an eval policy")
        if not 1 <= max_edits <= 64 or not 1 <= max_candidates <= 4096 or not 0 < wall_seconds <= 120:
            raise ValueError("invalid iteration budget")
        if not 0 <= blue_risk <= 1:
            raise ValueError("invalid blue risk threshold")
        self.policy, self.runner = policy, runner
        self.max_edits, self.max_candidates, self.wall_seconds, self.blue_risk = max_edits, max_candidates, wall_seconds, blue_risk

    @torch.inference_mode()
    def repair(self, image: bytes, command: str, examples: list[Example], *, mode="routed", fixed_depth=None):
        if not examples or len(examples) > 16:
            raise ValueError("editor needs 1..16 visible examples")
        started = time.perf_counter(); original = image; current = image
        trace = []; seen = {digest(image)}; candidates = 0; committed = 0
        previous_steps = self.policy.recurrent_steps
        previous_encodes = self.policy.green.encoded_chunks
        previous_banks = self.policy.green.memory.bank_builds
        wrong_stop = 0

        def evaluate(raw):
            actual = [self.runner.run(raw, example.case) for example in examples]
            passed = {i for i, (r, e) in enumerate(zip(actual, examples, strict=True)) if same_result(r, e.expected)}
            try:
                info = inspect_elf(raw)
                valid = bool(info["entry_in_executable_segment"] and info["type"] in ("ET_EXEC", "ET_DYN"))
            except ValueError:
                valid = False
            return actual, passed, valid

        def finish(status, stop):
            final = current if stop else original
            return {"status": status, "stop": stop, "bytes": final, "source_sha256": digest(original),
                    "sha256": digest(final), "tentative_sha256": digest(current), "edits": committed,
                    "candidate_attempts": candidates, "wrong_stop_proposals": wrong_stop,
                    "recurrent_steps": self.policy.recurrent_steps - previous_steps,
                    "encoded_unique_chunks": self.policy.green.encoded_chunks - previous_encodes,
                    "bank_builds": self.policy.green.memory.bank_builds - previous_banks,
                    "wall_seconds": time.perf_counter() - started, "trace": trace,
                    "assurance": "visible_program_tests_only_not_proven"}

        actual, passed, valid = evaluate(current)
        if len(passed) == len(examples) and valid:
            trace.append({"action": "STOP", "origin": "visible_precheck", "image": digest(current)})
            return finish("observed_unchanged", True)
        for step in range(self.max_edits):
            if candidates >= self.max_candidates or time.perf_counter() - started >= self.wall_seconds:
                break
            frames = execution_frames(examples, actual)
            best = None
            for output, state in self.policy.thoughts(current, command, frames, mode=mode, fixed_depth=fixed_depth):
                action = greedy_action(output, current)
                trace.append({"action": "THINK", "step": step, "image": digest(current),
                              "origin": action["origin"], "depth": action["depth"], "risk": action["risk"],
                              "instances": state["observation"].instances,
                              "unique_chunks": len(state["observation"].payloads)})
                if action["stop"]:
                    wrong_stop += 1
                    continue
                if action["origin"] == "blue" and action["risk"] > self.blue_risk:
                    continue
                logits, bits = output["logits"], output["bits"]
                # Neural proposals only: no clean image, corruption offsets or
                # opcode-specific inverse-mutation rule is available here.
                positions = logits.topk(min(3, len(logits))).indices.tolist()
                found_progress = False
                for p in positions:
                    value = sum(int(v >= 0) << k for k, v in enumerate(bits[p]))
                    uncertain = bits[p].abs().topk(2, largest=False).indices.tolist()
                    values = [value, *(value ^ (1 << k) for k in uncertain)]
                    for replacement in values:
                        if replacement == current[p]:
                            continue
                        patch = BytePatch(p, current[p:p + 1], bytes([replacement]))
                        proposal = apply_patches(current, [patch], digest(current)); identity = digest(proposal)
                        if identity in seen:
                            continue
                        if candidates >= self.max_candidates or time.perf_counter() - started >= self.wall_seconds:
                            break
                        seen.add(identity); candidates += 1
                        results, proposal_passed, proposal_valid = evaluate(proposal)
                        reward = transition_reward(len(passed), len(proposal_passed), valid, proposal_valid)
                        if not passed <= proposal_passed:
                            continue
                        quality = (len(proposal_passed), int(proposal_valid))
                        record = (quality, reward, float(logits[p]), proposal, results, proposal_passed, proposal_valid, patch, action)
                        if best is None or record[:3] > best[:3]:
                            best = record
                        if len(proposal_passed) == len(examples) and proposal_valid:
                            before = digest(current)
                            current = proposal; committed += 1
                            trace.append({"action": "PATCH", "step": step, "before": before, "after": identity,
                                          "patch": patch.json(), "origin": action["origin"], "depth": action["depth"], "speculative": False})
                            trace.append({"action": "STOP", "origin": "visible_tests", "image": identity})
                            return finish("sample_checked_candidate", True)
                        if quality > (len(passed), int(valid)):
                            found_progress = True
                            break
                    if found_progress:
                        break
                if found_progress:
                    break  # Deeper recurrent states are not computed after acceptance.
            if best is None:
                break
            quality, reward, score, proposal, actual, proposal_passed, proposal_valid, patch, action = best
            before = digest(current)
            speculative = quality <= (len(passed), int(valid))
            current, passed, valid = proposal, proposal_passed, proposal_valid
            committed += 1
            trace.append({"action": "PATCH", "step": step, "before": before, "after": digest(current),
                          "patch": patch.json(), "origin": action["origin"], "depth": action["depth"],
                          "speculative": speculative, "passed_visible": sorted(passed), "reward": reward})
        return finish("budget" if candidates >= self.max_candidates or committed >= self.max_edits else "abstain", False)
