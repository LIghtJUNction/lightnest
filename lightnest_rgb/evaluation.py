"""Frozen program-held-out evaluation; hidden outputs never enter the editor."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import torch

from .artifact import digest, inspect_elf
from .corpus import damaged_image, external, rows, sha, source_hashes, write
from .execute import Case, ExecutableRunner, same_result
from .iterate import Example, RGBEditor
from .training import load_generator, load_policy, verify_corpus

COMMAND = "restore the executable to satisfy the visible program tests"


def freeze(out, name, policy, generator, *, per_program=32, generation_count=32):
    out = external(out); verify_corpus(out)
    if not name.isidentifier() or not 1 <= per_program <= 128 or not 2 <= generation_count <= 96:
        raise ValueError("invalid evaluation name or sample count")
    load_policy(policy); load_generator(generator)
    sampler = random.Random(2026102431); groups = defaultdict(list)
    for task in rows(out / "tasks" / "test.jsonl"):
        groups[task["program"]].append(task)
    selected = []
    for program, tasks in sorted(groups.items()):
        if per_program > len(tasks): raise ValueError("sample count exceeds prepared tasks per program")
        strata = defaultdict(list)
        for task in tasks:
            strata[(task["fault_count"], task["band"])].append(task)
        for values in strata.values(): sampler.shuffle(values)
        picked = []
        while len(picked) < per_program:
            for key in sorted(strata):
                if strata[key] and len(picked) < per_program: picked.append(strata[key].pop())
        selected.extend(task["id"] for task in picked)
    generation = rows(out / "generation" / "test.jsonl")
    exits = [r for r in generation if "exit" in r["spec"]]; stdout = [r for r in generation if "stdout" in r["spec"]]
    sampler.shuffle(exits); sampler.shuffle(stdout)
    exit_count = min(len(exits), generation_count // 2)
    chosen = [r["id"] for r in exits[:exit_count] + stdout[:generation_count - exit_count]]
    if len(chosen) != generation_count: raise ValueError("requested too many generation examples of one kind")
    folder = out / "evaluation" / name; folder.mkdir(exist_ok=False)
    plan = {"policy": str(Path(policy).resolve()), "generator": str(Path(generator).resolve()),
            "policy_sha256": sha(policy), "generator_sha256": sha(generator), "corpus_sha256": sha(out / "corpus.json"),
            "sources": source_hashes(), "task_ids": selected, "generation_ids": chosen,
            "max_edits": 8, "max_candidates": 32, "wall_seconds": 15., "visible_cases": 4,
            "hidden_cases": 8, "seed": 2026102431, "repair_test_task_queries_before_freeze": 0,
            "generation_test_request_queries_before_freeze": 0,
            "prior_unlabeled_observation_benchmarks": {p.name: sha(p) for p in sorted((out / "evaluation").glob("observation*.json"))}}
    write(folder / "frozen.json", plan)
    return folder


def _frozen(out, name):
    folder = out / "evaluation" / name; plan = json.loads((folder / "frozen.json").read_text())
    if plan["sources"] != source_hashes(): raise ValueError("evaluation source changed after freeze")
    for key in ("policy", "generator"):
        if sha(plan[key]) != plan[key + "_sha256"]: raise ValueError("frozen checkpoint changed")
    if sha(out / "corpus.json") != plan["corpus_sha256"]: raise ValueError("frozen corpus changed")
    verify_corpus(out)
    return folder, plan


def evaluate_policy(out, name, *, mode="routed", fixed_depth=None):
    out = external(out); folder, plan = _frozen(out, name)
    torch.set_num_threads(1)
    label = f"{mode}-depth{fixed_depth or 'adaptive'}"
    model, _ = load_policy(plan["policy"])
    runner = ExecutableRunner(folder / (label + "-execution"), timeout=.4, cache=True)
    editor = RGBEditor(model, runner, max_edits=plan["max_edits"], max_candidates=plan["max_candidates"], wall_seconds=plan["wall_seconds"])
    tasks = {t["id"]: t for t in rows(out / "tasks" / "test.jsonl")}
    programs = {p["name"]: p for p in json.loads((out / "executables.json").read_text())}
    counts = Counter(); started = time.perf_counter()
    with (folder / (label + ".jsonl")).open("x") as file:
        for identity in plan["task_ids"]:
            task = tasks[identity]; program = programs[task["program"]]
            image, target = damaged_image(out, task)
            examples = [Example(Case.parse(c["input"]), {k: c["expected"][k] for k in ("status", "returncode", "stdout", "stderr")}) for c in program["cases"][:4]]
            result = editor.repair(image, COMMAND, examples, mode=mode, fixed_depth=fixed_depth)
            final = result.pop("bytes")
            hidden = [same_result(runner.run(final, Case.parse(c["input"])), c["expected"]) for c in program["cases"][4:]]
            precheck = bool(result["trace"] and result["trace"][0].get("origin") == "visible_precheck")
            hidden_correct = all(hidden)
            counts["total"] += 1; counts["damaged" if task["fault_count"] else "clean"] += 1
            counts["accepted"] += int(result["stop"]); counts["hidden_correct"] += int(result["stop"] and hidden_correct)
            counts["false_accept"] += int(result["stop"] and not hidden_correct)
            counts["precheck"] += int(precheck); counts["learned_repairs"] += int(result["stop"] and not precheck and hidden_correct)
            counts["multi_edit_repairs"] += int(result["stop"] and not precheck and hidden_correct and result["edits"] >= 2)
            counts["exact_restorations"] += int(final == target)
            for k in ("candidate_attempts", "recurrent_steps", "encoded_unique_chunks", "bank_builds", "wrong_stop_proposals"): counts[k] += result[k]
            event = {"id": identity, "program": task["program"], "fault_count": task["fault_count"], "band": task["band"],
                     "requested_band": task.get("requested_band", "clean"), "hidden_correct": hidden_correct, "exact_original": final == target, **result}
            file.write(json.dumps(event) + "\n"); file.flush()
            print(label, identity, result["status"], hidden_correct, result["edits"], flush=True)
    summary = {"variant": label, "counts": dict(counts), "wall_seconds": time.perf_counter() - started,
               "native_calls": runner.native_calls, "execution_cache_hits": runner.cache_hits,
               "preflight_rejections": runner.preflight_rejections,
               "frozen_sha256": sha(folder / "frozen.json"), "results_sha256": sha(folder / (label + ".jsonl")),
               "claim_scope": "program-held-out injected faults; visible tests are not a proof"}
    write(folder / (label + "-summary.json"), summary)
    return summary


def evaluate_generation(out, name):
    out = external(out); folder, plan = _frozen(out, name); torch.set_num_threads(1)
    model, _ = load_generator(plan["generator"])
    records = {r["id"]: r for r in rows(out / "generation" / "test.jsonl")}
    runner = ExecutableRunner(folder / "generation-execution", cache=False)
    counts = Counter(); started = time.perf_counter()
    with (folder / "generation.jsonl").open("x") as file:
        for identity in plan["generation_ids"]:
            record = records[identity]; result = model.generate(record["spec"]); raw = result.pop("bytes")
            try:
                info = inspect_elf(raw); valid = info["type"] in ("ET_EXEC", "ET_DYN") and info["entry_in_executable_segment"]
            except ValueError: valid = False
            actual = runner.run(raw, Case()) if valid and result["stop"] else None
            spec = record["spec"]
            correct = bool(actual and actual["status"] == "ok" and actual["returncode"] == spec.get("exit", 0)
                           and actual["stdout"] == spec.get("stdout", "") and actual["stderr"] == "")
            category = "exit" if "exit" in spec else "stdout"
            counts["total"] += 1; counts[category + "_total"] += 1; counts[category + "_correct"] += int(correct)
            counts["elf_valid"] += int(valid); counts["behavior_correct"] += int(correct); counts["stopped"] += int(result["stop"])
            counts["exact_reference"] += int(digest(raw) == record["sha256"])
            (folder / (identity.replace("/", "-") + ".elf")).write_bytes(raw)
            event = {"id": identity, "spec": spec, "sha256": digest(raw), "bytes": len(raw), "elf_valid": valid,
                     "behavior_correct": correct, "execution": actual, **result}
            file.write(json.dumps(event) + "\n"); file.flush(); print("GENERATED", identity, correct, flush=True)
    summary = {"counts": dict(counts), "wall_seconds": time.perf_counter() - started, "native_calls": runner.native_calls,
               "frozen_sha256": sha(folder / "frozen.json"), "results_sha256": sha(folder / "generation.jsonl"),
               "postprocessing": "none; every file byte emitted by learned decoder"}
    write(folder / "generation-summary.json", summary)
    return summary
