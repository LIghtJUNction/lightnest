"""Real-file RGB training and complete learned-output ELF generation training."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .artifact import digest, inspect_elf
from .corpus import ROOT, damaged_image, external, rows, sha, write
from .execute import Case, ExecutableRunner
from .generate import ELFGenerator, condition_tensor, generation_loss
from .model import ARCHITECTURE, RGBConfig, RGBPolicy, budget_target, greedy_action, patch_loss
from .iterate import Example, execution_frames


def math_sources(kind):
    names = ("generate.py",) if kind == "generator" else ("attention.py", "observe.py", "model.py")
    return {name: sha(Path(__file__).parent / name) for name in names}


def verify_corpus(out):
    protocol = json.loads((out / "corpus.json").read_text())
    if protocol["manifest_sha256"] != sha(out / "executables.json"):
        raise ValueError("real executable manifest changed")
    for split, value in protocol["tasks"].items():
        if sha(out / "tasks" / f"{split}.jsonl") != value:
            raise ValueError("real ELF task split changed")
    for split, value in protocol["generation"].items():
        if sha(out / "generation" / f"{split}.jsonl") != value:
            raise ValueError("generation split changed")
    return protocol


def save_policy(path, model, metadata):
    if Path(path).exists():
        raise FileExistsError(path)
    torch.save({"architecture": ARCHITECTURE, "kind": "policy", "config": asdict(model.config),
                "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_policy(path, *, check_source=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["architecture"] != ARCHITECTURE or payload["kind"] != "policy":
        raise ValueError("checkpoint is not an RGB ELF policy")
    if check_source and payload["metadata"]["math_sources"] != math_sources("policy"):
        raise ValueError("policy math source changed since this checkpoint")
    config = payload["config"]; config["red_depths"] = tuple(config["red_depths"])
    model = RGBPolicy(RGBConfig(**config))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {"red_to_blue.weight", "red_to_blue.bias", "red_abstraction.0.weight", "red_abstraction.0.bias",
                       "red_abstraction.2.weight", "red_abstraction.2.bias", "abstract_to_blue.weight", "abstract_to_blue.bias",
                       "blue_language.weight", "blue_language.bias"}
    allowed_missing |= {name for name in missing if name.startswith("budget.")}
    if set(unexpected) or not set(missing) <= allowed_missing:
        raise ValueError(f"policy checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval(), payload["metadata"]


def save_generator(path, model, metadata):
    if Path(path).exists():
        raise FileExistsError(path)
    torch.save({"architecture": ARCHITECTURE, "kind": "generator", "config": {"width": model.width, "capacity": model.capacity},
                "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_generator(path, *, check_source=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["architecture"] != ARCHITECTURE or payload["kind"] != "generator":
        raise ValueError("checkpoint is not an RGB full-image generator")
    if check_source and payload["metadata"]["math_sources"] != math_sources("generator"):
        raise ValueError("generation math source changed since this checkpoint")
    model = ELFGenerator(**payload["config"]); model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload["metadata"]


def _plan(out, kind, seed, settings):
    folder = out / "training" / f"{kind}-{seed}"
    folder.mkdir(exist_ok=False)
    plan = {"kind": kind, "seed": seed, "settings": settings, "corpus_sha256": sha(out / "corpus.json"),
            "math_sources": math_sources(kind), "trainer_sha256": sha(Path(__file__)),
            "uv_lock_sha256": sha(ROOT / "uv.lock"), "torch": str(torch.__version__), "device": "cpu", "test_model_queries": 0}
    write(folder / "plan.json", plan)
    # Preserve the exact implementation behind the recorded hashes, including
    # curriculum/controller helpers, even if later source work continues.
    snapshot = folder / "source"; snapshot.mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (snapshot / path.name).write_bytes(path.read_bytes())
    return folder, plan


def _frames(program, results, count=2):
    examples = [Example(Case.parse(c["input"]), {k: c["expected"][k] for k in ("status", "returncode", "stdout", "stderr")})
                for c in program["cases"][:count]]
    return execution_frames(examples, results)


def train_policy(out, seed=104729, *, warmup_steps=2048, full_steps=512, width=64):
    out = external(out); verify_corpus(out)
    if warmup_steps < 1 or full_steps < 1:
        raise ValueError("training budgets must be positive")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    config = RGBConfig(width=width)
    model = RGBPolicy(config)
    folder, plan = _plan(out, "policy", seed, {"warmup_steps": warmup_steps, "full_steps": full_steps, "config": asdict(config),
                                             "lr": .0003, "warmup_microbatch": 2, "full_microbatch": 1})
    programs = {p["name"]: p for p in json.loads((out / "executables.json").read_text())}
    train_programs = [p for p in programs.values() if p["split"] == "train"]
    images = {p["name"]: (out / "objects" / (p["sha256"] + ".elf")).read_bytes() for p in train_programs}
    for p in train_programs:
        if digest(images[p["name"]]) != p["sha256"]:
            raise ValueError("immutable executable changed")
    tasks = rows(out / "tasks" / "train.jsonl")
    sampler = random.Random(2026102423)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    runner = ExecutableRunner(folder / "training-execution", timeout=.4)
    started = time.perf_counter(); windows = set(); seen_tasks = set(); stream = __import__("hashlib").sha256()
    totals = {"warmup_updates": 0, "full_updates": 0, "sample_presentations": 0}
    with (folder / "updates.jsonl").open("x") as log:
        for stage, steps in (("warmup", warmup_steps), ("full", full_steps)):
            losses = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                microbatch = 2 if stage == "warmup" else 1
                for _ in range(microbatch):
                    if stage == "warmup":
                        program = sampler.choice(train_programs); raw = images[program["name"]]
                        low = 0 if sampler.random() < .5 else sampler.randrange(max(1, len(raw) // 64)) * 64
                        target = raw[low:low + 256]; image = bytearray(target)
                        count = 0 if sampler.random() < .25 else sampler.choice((1, 2, 4, 8))
                        for offset in sampler.sample(range(len(image)), min(count, len(image))):
                            image[offset] ^= 1 << sampler.randrange(8)
                        image = bytes(image); frames = ()
                        windows.add((program["sha256"], low)); stream.update(f"{program['sha256']}:{low}:{digest(image)}\n".encode())
                    else:
                        task = sampler.choice(tasks); program = programs[task["program"]]
                        image, target = damaged_image(out, task)
                        results = [runner.run(image, Case.parse(c["input"])) for c in program["cases"][:2]]
                        frames = _frames(program, results); seen_tasks.add(task["id"]); stream.update((task["id"] + "\n").encode())
                    depth = config.red_depths[step % len(config.red_depths)]
                    state = model.begin(image, "restore the executable to satisfy the visible program tests", frames)
                    model.advance(state, config.blue_depth)
                    blue = model.readout(state, blue=True)
                    candidates = [blue]
                    red = None
                    for candidate_depth in config.red_depths:
                        model.advance(state, candidate_depth)
                        candidates.append(model.readout(state, blue=True))
                        candidate_red = model.readout(state)
                        if candidate_depth == depth:
                            red = candidate_red
                    if red is None:
                        raise RuntimeError("training depth was not present in RGB schedule")
                    selected_budget = budget_target(candidates, image, target)
                    with torch.no_grad():
                        action = greedy_action(blue, image)
                        legal = (image == target) if action["stop"] else image[action["offset"]] != target[action["offset"]] and action["value"] == target[action["offset"]]
                    risk = F.binary_cross_entropy_with_logits(blue["risk_logit"], torch.tensor(float(not legal)))
                    budget_loss = F.cross_entropy(blue["budget_logits"], torch.tensor([selected_budget]))
                    loss = (.3 * patch_loss(blue, image, target) + patch_loss(red, image, target)
                            + .2 * risk + .1 * budget_loss)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite RGB policy loss")
                    (loss / microbatch).backward(); losses.append(float(loss.detach())); totals["sample_presentations"] += 1
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); totals[f"{stage}_updates"] += 1
                if (step + 1) % 64 == 0:
                    event = {"stage": stage, "step": step + 1, "mean_loss": sum(losses) / len(losses),
                             "gradient_norm": float(norm), "wall_seconds": time.perf_counter() - started}
                    log.write(json.dumps(event) + "\n"); log.flush(); losses = []; print(seed, event, flush=True)
            # Separate phase endpoints protect actual completed work on interruption.
            save_policy(folder / f"{stage}-endpoint.pt", model, plan | totals)
    metadata = plan | totals | {"unique_native_programs": len(train_programs), "unique_windows": len(windows),
                               "unique_full_tasks": len(seen_tasks), "stream_sha256": stream.hexdigest(),
                               "parameters": sum(p.numel() for p in model.parameters()), "wall_seconds": time.perf_counter() - started,
                               "trained_depths": list(config.red_depths), "status": "research_candidate"}
    path = out / "weights" / f"rgb-{seed}.pt"; save_policy(path, model, metadata)
    write(folder / "complete.json", metadata | {"checkpoint_sha256": sha(path)})
    print("POLICY_TRAINED", seed, sha(path), flush=True)


def _generation_batch(items, out):
    images = [(out / "generation" / (item["sha256"] + ".elf")).read_bytes() for item in items]
    length = max(map(len, images)) + 1
    previous = torch.zeros(len(images), length, dtype=torch.long)
    targets = torch.zeros_like(previous); valid = torch.zeros_like(previous, dtype=torch.bool)
    for i, image in enumerate(images):
        previous[i, :len(image) + 1] = torch.tensor([256, *image])
        targets[i, :len(image) + 1] = torch.tensor([*image, 256]); valid[i, :len(image) + 1] = True
    return previous, targets, valid, condition_tensor([item["spec"] for item in items])


def _raw_batch(images, sampler, batch=8, length=128):
    values = []
    for _ in range(batch):
        image = sampler.choice(images); start = sampler.randrange(max(1, len(image) - length))
        values.append(image[start:start + length])
    previous = torch.tensor([[256, *value] for value in values])
    target = torch.tensor([[*value, 256] for value in values])
    condition = torch.zeros(batch, 10, dtype=torch.long); condition[:, 0] = 2
    return previous, target, torch.ones_like(target, dtype=torch.bool), condition


@torch.inference_mode()
def generation_probe(model, items, runner):
    model.eval(); result = {"total": len(items), "elf_valid": 0, "behavior_correct": 0, "stopped": 0}
    for item in items:
        generated = model.generate(item["spec"])
        result["stopped"] += int(generated["stop"])
        try:
            info = inspect_elf(generated["bytes"])
            valid = info["type"] in ("ET_EXEC", "ET_DYN") and info["entry_in_executable_segment"]
        except ValueError:
            valid = False
        result["elf_valid"] += int(valid)
        if generated["stop"] and valid:
            actual = runner.run(generated["bytes"], Case())
            spec = item["spec"]
            correct = actual["status"] == "ok" and actual["returncode"] == spec.get("exit", 0) and actual["stdout"] == spec.get("stdout", "")
            result["behavior_correct"] += int(correct)
    return result


def train_generator(out, seed=104729, *, steps=1536, width=128):
    out = external(out); verify_corpus(out)
    if steps < 1:
        raise ValueError("training updates must be positive")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    model = ELFGenerator(width, 1024)
    folder, plan = _plan(out, "generator", seed, {"steps": steps, "batch": 8, "native_lm_batch": 8,
                                                 "native_lm_length": 128, "native_lm_loss_weight": .25,
                                                 "width": width, "capacity": 1024, "lr": .001})
    fit = rows(out / "generation" / "train.jsonl"); validation = rows(out / "generation" / "validation.jsonl")
    native = [p for p in json.loads((out / "executables.json").read_text()) if p["split"] == "train"]
    images = [(out / "objects" / (p["sha256"] + ".elf")).read_bytes() for p in native]
    sampler = random.Random(2026102427); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    runner = ExecutableRunner(folder / "generation-probe", cache=False)
    started = time.perf_counter(); probes = []; seen = set(); native_bytes = 0
    with (folder / "updates.jsonl").open("x") as log:
        losses = []
        for step in range(steps):
            selected = [sampler.choice(fit) for _ in range(8)]
            seen.update(item["id"] for item in selected)
            previous, target, valid, condition = _generation_batch(selected, out)
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(previous, condition, red_depth=2)
            loss = generation_loss(output, target, valid)
            previous_raw, target_raw, valid_raw, condition_raw = _raw_batch(images, sampler)
            raw_output = model(previous_raw, condition_raw, red_depth=2)
            native_loss = generation_loss(raw_output, target_raw, valid_raw)
            total = loss + .25 * native_loss
            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite generation loss")
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach())); native_bytes += int(valid_raw.sum()) - len(valid_raw)
            if (step + 1) % 64 == 0:
                event = {"step": step + 1, "mean_generation_loss": sum(losses) / len(losses),
                         "native_lm_loss": float(native_loss.detach()), "wall_seconds": time.perf_counter() - started}
                log.write(json.dumps(event) + "\n"); log.flush(); losses = []; print(seed, event, flush=True)
            if (step + 1) % 256 == 0:
                probe = generation_probe(model, [validation[i] for i in (0, 1, 32, 33)], runner)
                probes.append({"step": step + 1, **probe}); print("GENERATION_PROBE", seed, step + 1, probe, flush=True)
                save_generator(folder / f"step-{step + 1}.pt", model, plan | {"updates": step + 1})
    metadata = plan | {"updates": steps, "sample_presentations": steps * 8, "unique_generation_records": len(seen),
                       "native_lm_bytes_presented": native_bytes, "native_programs": len(native),
                       "validation_probes": probes, "parameters": sum(p.numel() for p in model.parameters()),
                       "wall_seconds": time.perf_counter() - started, "status": "research_candidate"}
    path = out / "weights" / f"generator-{seed}.pt"; save_generator(path, model, metadata)
    write(folder / "complete.json", metadata | {"checkpoint_sha256": sha(path)})
    print("GENERATOR_TRAINED", seed, sha(path), flush=True)
