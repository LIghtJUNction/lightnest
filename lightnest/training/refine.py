"""A declared header-focused curriculum; no clean reference reaches inference."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import torch

from lightnest.training.corpus import damaged_image, external, rows, sha, write
from lightnest.runtime.execute import Case, ExecutableRunner
from lightnest.architecture.model import patch_loss
from lightnest.training.training import _frames, _plan, load_policy, save_policy, verify_corpus


def ranked_loss(output, image, target):
    loss = patch_loss(output, image, target)
    changed = torch.tensor([a != b for a, b in zip(image, target, strict=True)])
    if changed.any():
        # Optimize selection of any valid next edit, rather than averaging
        # thousands of irrelevant intact bytes into the action objective.
        loss = loss + 2 * (output["logits"].logsumexp(0) - output["logits"][changed].logsumexp(0))
    return loss


def refine_policy(out, parent, seed=104730, *, header_steps=4096, full_steps=128):
    if header_steps < 1 or full_steps < 1: raise ValueError("refinement budgets must be positive")
    out = external(out); verify_corpus(out); torch.set_num_threads(1)
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    model, metadata = load_policy(parent); model.train()
    folder, plan = _plan(out, "policy", seed, {"parent_sha256": sha(parent), "parent": metadata,
        "header_steps": header_steps, "full_steps": full_steps, "lr": .001,
        "curriculum": "first 256 bytes of real training programs, then complete training tasks",
        "ranking_loss_weight": 2., "refiner_sha256": sha(Path(__file__))})
    programs = {p["name"]: p for p in json.loads((out / "executables.json").read_text()) if p["split"] == "train"}
    images = {p["name"]: (out / "objects" / (p["sha256"] + ".elf")).read_bytes() for p in programs.values()}
    tasks = rows(out / "tasks" / "train.jsonl"); sampler = random.Random(2026102439)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    runner = ExecutableRunner(folder / "execution", timeout=.4); started = time.perf_counter(); updates = 0
    with (folder / "updates.jsonl").open("x") as file:
        for stage, steps in (("header", header_steps), ("full", full_steps)):
            losses = []
            for step in range(steps):
                if stage == "header":
                    target = sampler.choice(list(images.values()))[:256]; image = bytearray(target)
                    count = 0 if sampler.random() < .25 else sampler.choice((1, 2, 4, 8))
                    region = 120 if sampler.random() < .75 else len(image)
                    for offset in sampler.sample(range(region), count): image[offset] ^= 1 << sampler.randrange(8)
                    image = bytes(image); frames = ()
                else:
                    task = sampler.choice(tasks); image, target = damaged_image(out, task); program = programs[task["program"]]
                    actual = [runner.run(image, Case.parse(c["input"])) for c in program["cases"][:2]]
                    frames = _frames(program, actual)
                optimizer.zero_grad(set_to_none=True)
                blue, red, _ = model(image, "restore the executable to satisfy the visible program tests", frames,
                                      depth=model.config.red_depths[step % len(model.config.red_depths)])
                loss = ranked_loss(red, image, target) + .3 * ranked_loss(blue, image, target)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite ranked loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                losses.append(float(loss.detach())); updates += 1
                if (step + 1) % 64 == 0:
                    event = {"stage": stage, "step": step + 1, "mean_loss": sum(losses) / len(losses), "wall_seconds": time.perf_counter() - started}
                    file.write(json.dumps(event) + "\n"); file.flush(); losses = []; print(seed, event, flush=True)
            save_policy(folder / (stage + "-endpoint.pt"), model, plan | {"updates": updates})
    result = plan | {"updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
                     "wall_seconds": time.perf_counter() - started, "status": "header_focused_research_candidate"}
    path = out / "weights" / f"rgb-{seed}.pt"; save_policy(path, model, result)
    write(folder / "complete.json", result | {"checkpoint_sha256": sha(path)})
    print("REFINED", seed, sha(path), flush=True)
