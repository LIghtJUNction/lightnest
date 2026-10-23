"""uv run --locked python -m lightnest_rgb --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .artifact import inspect_elf, write_image
from .corpus import external
from .execute import Case, ExecutableRunner
from .iterate import Example, RGBEditor


def main():
    parser = argparse.ArgumentParser(description="RGB complete-ELF research; PyTorch CPU")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("targets")
    p = commands.add_parser("decode"); p.add_argument("--architecture", required=True); p.add_argument("--hex", required=True)
    p = commands.add_parser("assemble"); p.add_argument("--architecture", required=True)
    p = commands.add_parser("inspect"); p.add_argument("image", type=Path)
    p = commands.add_parser("benchmark"); p.add_argument("image", type=Path); p.add_argument("--checkpoint", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--out", required=True); p.add_argument("--per-program", type=int, default=128)
    for action in ("train-policy", "train-generator"):
        p = commands.add_parser(action); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=104729)
        if action == "train-policy":
            p.add_argument("--warmup-steps", type=int, default=2048); p.add_argument("--full-steps", type=int, default=512)
        else: p.add_argument("--steps", type=int, default=1536)
    for action in ("refine-policy", "refine-generator"):
        p = commands.add_parser(action); p.add_argument("--out", required=True); p.add_argument("--parent", required=True)
        p.add_argument("--seed", type=int, default=104730)
        if action == "refine-policy":
            p.add_argument("--header-steps", type=int, default=4096); p.add_argument("--full-steps", type=int, default=128)
        else: p.add_argument("--steps", type=int, default=512)
    for action in ("generate", "repair"):
        p = commands.add_parser(action); p.add_argument("--checkpoint", required=True); p.add_argument("--request", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True); p.add_argument("--workdir", required=True)
        if action == "repair": p.add_argument("image", type=Path)
    p = commands.add_parser("freeze"); p.add_argument("--out", required=True); p.add_argument("--name", required=True)
    p.add_argument("--policy", required=True); p.add_argument("--generator", required=True)
    p.add_argument("--per-program", type=int, default=32); p.add_argument("--generation-count", type=int, default=32)
    p = commands.add_parser("evaluate"); p.add_argument("--out", required=True); p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=("policy", "generation"), required=True); p.add_argument("--mode", choices=("routed", "compressed", "dense"), default="routed")
    p.add_argument("--fixed-depth", type=int)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.action == "targets":
        from .isa import llvm_targets
        result = llvm_targets()
    elif args.action == "decode":
        from dataclasses import asdict
        from .isa import decode
        result = [{k: v.hex() if isinstance(v, bytes) else v for k, v in asdict(i).items()} for i in decode(args.architecture, bytes.fromhex(args.hex))]
    elif args.action == "assemble":
        from .isa import assemble
        result = {"hex": assemble(args.architecture, sys.stdin.read()).hex()}
    elif args.action == "inspect": result = inspect_elf(args.image.read_bytes())
    elif args.action == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.checkpoint, args.image.read_bytes())
    elif args.action == "prepare":
        from .corpus import prepare
        prepare(args.out, per_program=args.per_program); return
    elif args.action == "train-policy":
        from .training import train_policy
        train_policy(args.out, args.seed, warmup_steps=args.warmup_steps, full_steps=args.full_steps); return
    elif args.action == "train-generator":
        from .training import train_generator
        train_generator(args.out, args.seed, steps=args.steps); return
    elif args.action == "refine-policy":
        from .refine import refine_policy
        refine_policy(args.out, args.parent, args.seed, header_steps=args.header_steps, full_steps=args.full_steps); return
    elif args.action == "refine-generator":
        from .refine_generation import refine_generator
        refine_generator(args.out, args.parent, args.seed, steps=args.steps); return
    elif args.action == "freeze":
        from .evaluation import freeze
        result = {"folder": str(freeze(args.out, args.name, args.policy, args.generator, per_program=args.per_program, generation_count=args.generation_count))}
    elif args.action == "evaluate":
        from .evaluation import evaluate_policy, evaluate_generation
        result = evaluate_generation(args.out, args.name) if args.kind == "generation" else evaluate_policy(args.out, args.name, mode=args.mode, fixed_depth=args.fixed_depth)
    else:
        from .training import load_generator, load_policy
        workdir = external(args.workdir); workdir.mkdir(parents=True, exist_ok=False)
        output = external(args.output); runner = ExecutableRunner(workdir / "execution", cache=False)
        request = json.loads(args.request.read_text())
        if args.action == "generate":
            model, _ = load_generator(args.checkpoint); result = model.generate(request); raw = result.pop("bytes")
            if not result["stop"]: raise ValueError("learned decoder did not stop; output not published")
            inspect_elf(raw); actual = runner.run(raw, Case())
            if not (actual["status"] == "ok" and actual["returncode"] == request.get("exit", 0)
                    and actual["stdout"] == request.get("stdout", "") and actual["stderr"] == ""):
                raise ValueError("generated image failed requested behavior; output not published")
            result["execution"] = actual
        else:
            if set(request) != {"command", "examples"}: raise ValueError("repair request requires command and visible examples only")
            examples = []
            for value in request["examples"]:
                if set(value) != {"input", "expected"}: raise ValueError("example requires input and expected observable output")
                examples.append(Example(Case.parse(value["input"]), value["expected"]))
            model, _ = load_policy(args.checkpoint)
            result = RGBEditor(model, runner).repair(args.image.read_bytes(), request["command"], examples)
            raw = result.pop("bytes")
            if not result["stop"]: raise ValueError("repair abstained; output not published")
        result["artifact"] = write_image(output, raw)
        (workdir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
