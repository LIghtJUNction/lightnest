"""Emphasize condition-dependent bytes discovered from training files only."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from lightnest.training.corpus import external, rows, sha, write
from lightnest.runtime.execute import ExecutableRunner
from lightnest.architecture.generate import generation_loss
from lightnest.training.training import _generation_batch, _plan, generation_probe, load_generator, save_generator, verify_corpus


def variable_positions(records, out):
    masks = {}
    for mode in ("exit", "stdout"):
        images = [(out / "generation" / (r["sha256"] + ".elf")).read_bytes() for r in records if mode in r["spec"]]
        if len(set(map(len, images))) != 1: raise ValueError("variable-byte curriculum needs uniform length within request mode")
        values = torch.tensor([list(raw) for raw in images])
        masks[mode] = values.min(0).values.ne(values.max(0).values)
    return masks


def refine_generator(out, parent, seed=104730, *, steps=512):
    if steps < 1: raise ValueError("refinement updates must be positive")
    out = external(out); verify_corpus(out); torch.set_num_threads(1)
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    model, previous = load_generator(parent); records = rows(out / "generation" / "train.jsonl")
    masks = variable_positions(records, out)
    validation = rows(out / "generation" / "validation.jsonl")
    folder, plan = _plan(out, "generator", seed, {"parent_sha256": sha(parent), "parent": previous,
        "steps": steps, "batch": 8, "lr": .0005, "variable_byte_weight": 2.,
        "variable_positions": {k: v.nonzero().flatten().tolist() for k, v in masks.items()},
        "variable_positions_source": "training file differences within each request mode only",
        "refiner_sha256": sha(Path(__file__))})
    sampler = random.Random(2026102441); optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    runner = ExecutableRunner(folder / "probe", cache=False); started = time.perf_counter(); probes = []
    with (folder / "updates.jsonl").open("x") as file:
        losses = []
        for step in range(steps):
            selected = [sampler.choice(records) for _ in range(8)]
            previous_tokens, targets, valid, condition = _generation_batch(selected, out)
            important = torch.zeros_like(valid)
            for i, record in enumerate(selected):
                mask = masks["exit" if "exit" in record["spec"] else "stdout"]
                important[i, :len(mask)] = mask
            model.train(); optimizer.zero_grad(set_to_none=True); result = model(previous_tokens, condition)
            expected = ((targets[..., None] >> torch.arange(8)) & 1).float()
            loss = generation_loss(result, targets, valid) + 2 * F.binary_cross_entropy_with_logits(result["bits"][important], expected[important])
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite variable-byte loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 64 == 0:
                event = {"step": step + 1, "mean_loss": sum(losses) / len(losses), "wall_seconds": time.perf_counter() - started}
                file.write(json.dumps(event) + "\n"); file.flush(); losses = []; print(seed, event, flush=True)
            if (step + 1) % 128 == 0:
                probe = generation_probe(model, [validation[i] for i in (0, 1, 32, 33)], runner)
                probes.append({"step": step + 1, **probe}); print("REFINED_GENERATION_PROBE", step + 1, probe, flush=True)
                save_generator(folder / f"step-{step + 1}.pt", model, plan | {"updates": step + 1})
    result = plan | {"updates": steps, "validation_probes": probes,
                     "parameters": sum(p.numel() for p in model.parameters()), "wall_seconds": time.perf_counter() - started,
                     "status": "condition_byte_focused_research_candidate"}
    path = out / "weights" / f"generator-{seed}.pt"; save_generator(path, model, result)
    write(folder / "complete.json", result | {"checkpoint_sha256": sha(path)})
    print("GENERATOR_REFINED", seed, sha(path), flush=True)
