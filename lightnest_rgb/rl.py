"""On-policy executable-feedback fine-tuning for the RGB editor."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import torch

from .artifact import BytePatch, apply_patches, digest
from .corpus import damaged_image, external, rows, sha, write
from .execute import Case, ExecutableRunner
from .feedback import outcome_reward, reinforce_loss
from .iterate import Example, execution_frames
from .model import greedy_action
from .training import _plan, load_policy, save_policy, verify_corpus


def train_feedback(out, parent, seed=104732, *, steps=128, lr=1e-5):
    """Fine-tune from real executable outcomes; no clean target is read by RL."""
    out = external(out); verify_corpus(out)
    if steps < 1: raise ValueError("feedback updates must be positive")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    # The parent may predate the red abstraction/blue guidance heads. This is
    # an explicit warm-start migration, recorded in the plan rather than
    # pretending the old checkpoint matches the new math source.
    model, parent_meta = load_policy(parent, check_source=False); model.train()
    folder, plan = _plan(out, "policy-feedback", seed, {"parent_sha256": sha(parent), "steps": steps, "lr": lr,
        "reward": "visible execution status/returncode/stdout/stderr plus ELF preflight",
        "hidden_feedback": False, "target_bytes_input": False})
    programs = {p["name"]: p for p in json.loads((out / "executables.json").read_text())}
    tasks = rows(out / "tasks" / "train.jsonl"); sampler = random.Random(2026102443)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    runner = ExecutableRunner(folder / "execution", timeout=.4); started = time.perf_counter(); baseline = 0.; records=[]
    with (folder / "updates.jsonl").open("x") as log:
        for step in range(steps):
            task = sampler.choice(tasks); program = programs[task["program"]]; image, _target = damaged_image(out, task)
            examples = [Example(Case.parse(c["input"]), {k: c["expected"][k] for k in ("status", "returncode", "stdout", "stderr")}) for c in program["cases"][:2]]
            before = [runner.run(image, e.case) for e in examples]
            frames = execution_frames(examples, before)
            model.eval()
            with torch.no_grad(): blue, red, _ = model(image, "restore the executable to satisfy the visible program tests", frames, depth=model.config.red_depths[-1])
            action = greedy_action(red, image)
            if action["stop"] or action["value"] == image[action["offset"]]:
                candidate = image
            else:
                patch = BytePatch(action["offset"], image[action["offset"]:action["offset"]+1], bytes([action["value"]]))
                candidate = apply_patches(image, [patch], digest(image))
            feedback = outcome_reward(image, candidate, examples, runner)
            model.train(); optimizer.zero_grad(set_to_none=True)
            # Recompute the same state with gradients; execution is the environment,
            # so its scalar reward is deliberately detached from the network graph.
            _blue, differentiable, _ = model(image, "restore the executable to satisfy the visible program tests", frames, depth=model.config.red_depths[-1])
            loss = reinforce_loss(differentiable, action, feedback.reward, baseline=baseline)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            baseline = .95 * baseline + .05 * feedback.reward
            record = {"step": step + 1, "task": task["id"], "reward": feedback.reward, "baseline": baseline,
                      "passed_before": feedback.passed_before, "passed_after": feedback.passed_after,
                      "errors": feedback.error_kinds, "loss": float(loss.detach()), "action": action}
            log.write(json.dumps(record) + "\n"); log.flush(); records.append(record)
            if (step + 1) % 16 == 0: print(seed, record, flush=True)
    metadata = plan | {"updates": steps, "mean_reward": sum(r["reward"] for r in records) / len(records),
                       "positive_rewards": sum(r["reward"] > 0 for r in records), "native_calls": runner.native_calls,
                       "preflight_rejections": runner.preflight_rejections, "wall_seconds": time.perf_counter() - started,
                       "status": "executable_feedback_research_candidate"}
    path = out / "weights" / f"rgb-feedback-{seed}.pt"; save_policy(path, model, metadata)
    write(folder / "complete.json", metadata | {"checkpoint_sha256": sha(path)})
    print("FEEDBACK_TRAINED", seed, sha(path), flush=True)
