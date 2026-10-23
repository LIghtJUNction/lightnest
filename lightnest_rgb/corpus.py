"""Reproducible real-executable corruption tasks and learned ELF-generation targets."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

from .artifact import BytePatch, apply_patches, digest, inspect_elf
from .execute import Case, ExecutableRunner, same_result

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = {
    "base64": (), "wc": ("-c",), "head": ("-n", "3"), "tail": ("-n", "3"),
    "uniq": (), "cut": ("-d,", "-f2"), "tr": ("a-z", "A-Z"), "sha256sum": (), "sort": (),
    "gzip": ("-n", "-c"), "bzip2": ("-c",), "xz": ("-c", "-T1"),
    "grep": ("-F", "foo"), "zstd": ("-q", "--single-thread", "--no-asyncio", "-c"),
    "sed": ("s/foo/bar/g",), "jq": ("-c", ".items | map(.value) | add"),
    "basename": ("--", "/usr/bin/lightnest"), "dirname": ("--", "/usr/bin/lightnest"),
    "rev": (), "fold": ("-w", "4"), "od": ("-An", "-tx1"),
    "strings": ("-n", "4"), "realpath": ("-m", "/tmp/lightnest"),
}
VALIDATION = {"grep", "zstd", "dirname", "realpath"}
TEST = {"sed", "jq", "basename", "od"}


def external(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("research artifacts must be outside the checkout")
    return path


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write(path, obj):
    with Path(path).open("x") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, allow_nan=False); file.write("\n")


def rows(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file]


def jsonl(path, items):
    with Path(path).open("x") as file:
        for row in items:
            file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "lightnest_rgb").glob("*.py"))}


def cases_for(name, args, seed=20261024):
    rng = random.Random(seed)
    data = [b"", b"foo,a,9\nbar,b,3\nfoo,a,9\n", b"z\na\na\n", b"1234567890\n" * 2]
    for i in range(8):
        lines = [f'{rng.choice(("foo", "bar", "other"))},{rng.randrange(1000)},{rng.randrange(1000)}\n' for _ in range(3 + i)]
        data.append("".join(lines).encode())
    if name == "jq":
        data = [json.dumps({"items": [{"value": rng.randrange(-10000, 10000)} for _ in range(i + 1)]}).encode() for i in range(12)]
    return [Case(tuple(args), value) for value in data]


def compile_generation_target(spec):
    """Reference-data construction only. The learned decoder never calls this."""
    if set(spec) == {"exit"}:
        assembly = f".global _start\n.text\n_start:\n mov ${spec['exit']},%edi\n mov $60,%eax\n syscall\n"
    else:
        payload = bytes.fromhex(spec["stdout"])
        if len(payload) != 8:
            raise ValueError("generation stdout reference requires eight bytes")
        assembly = (".global _start\n.text\n_start:\n mov $1,%edi\n lea message(%rip),%rsi\n mov $8,%edx\n"
                    " mov $1,%eax\n syscall\n xor %edi,%edi\n mov $60,%eax\n syscall\n.section .rodata\nmessage:\n.byte " +
                    ",".join(map(str, payload)) + "\n")
    assembly += '.section .note.GNU-stack,"",@progbits\n'
    with tempfile.TemporaryDirectory(prefix="lightnest-link-target-") as temp:
        path = Path(temp); (path / "target.s").write_text(assembly)
        subprocess.run(["as", "--64", "-o", str(path / "target.o"), str(path / "target.s")], check=True, capture_output=True)
        subprocess.run(["ld", "--nmagic", "--build-id=none", "--strip-all", "-o", str(path / "target"), str(path / "target.o")],
                       check=True, capture_output=True)
        return (path / "target").read_bytes()


def prepare(out, *, per_program=128):
    out = external(out)
    if type(per_program) is not int or not 16 <= per_program <= 4096:
        raise ValueError("per-program task count must be 16..4096")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("objects", "tasks", "generation", "weights", "training", "evaluation"):
        (out / name).mkdir()
    started = time.perf_counter(); runner = ExecutableRunner(out / "construction-execution", timeout=.35)
    manifest = []; task_splits = {name: [] for name in ("train", "validation", "test")}; counters = Counter()
    rng = random.Random(2026102419)
    for name, args in PROGRAMS.items():
        source = Path(shutil.which(name) or "")
        if not source.is_file():
            raise RuntimeError(f"required real executable missing: {name}")
        raw = source.read_bytes(); info = inspect_elf(raw)
        object_path = out / "objects" / (digest(raw) + ".elf")
        if not object_path.exists(): object_path.write_bytes(raw); object_path.chmod(0o400)
        split = "test" if name in TEST else "validation" if name in VALIDATION else "train"
        cases = cases_for(name, args); reference = [runner.run(raw, case) for case in cases]
        if any(r["status"] != "ok" for r in reference):
            raise RuntimeError(f"clean real executable failed isolation preflight: {name}")
        if name not in ("grep",) and any(r["returncode"] for r in reference):
            raise RuntimeError(f"clean corpus command failed: {name}")
        try:
            package = subprocess.check_output(["pacman", "-Qoq", str(source)], stderr=subprocess.DEVNULL).decode().strip()
            package_version = subprocess.check_output(["pacman", "-Q", package]).decode().strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            package_version = "unavailable; artifact hash is authoritative"
        manifest.append({"name": name, "source": str(source.resolve()), "sha256": digest(raw), "size": len(raw),
                         "package": package_version, "split": split, "elf": info,
                         "cases": [{"input": c.json(), "expected": {k: r[k] for k in ("status", "returncode", "stdout", "stderr")}}
                                   for c, r in zip(cases, reference, strict=True)]})
        text = next(s for s in info["sections"] if s["name"] == ".text")
        rodata = next((s for s in info["sections"] if s["name"] == ".rodata"), text)
        bands = {"header": (0, min(120, len(raw))), "text": (text["offset"], text["offset"] + text["size"]),
                 "rodata": (rodata["offset"], rodata["offset"] + rodata["size"])}
        entry_segment = next(s for s in info["segments"] if s["type"] == "PT_LOAD" and
                             s["address"] <= info["entry"] < s["address"] + s["file_size"])
        entry_offset = entry_segment["offset"] + info["entry"] - entry_segment["address"]
        for index in range(per_program):
            if index % 8 == 7:
                task_splits[split].append({"id": f"{name}/{index}", "program": name, "image": digest(raw),
                                           "patches": [], "fault_count": 0, "band": "clean", "observed_failures": []})
                continue
            faults = (1, 2, 4, 8)[(index // 4) % 4]
            requested_band = ("header", "text", "rodata", "mixed")[index % 4]
            accepted = None
            for attempt in range(24):
                # Hard regions remain explicit. A bounded fallback produces a
                # real failing task instead of quietly fabricating an effect.
                band = requested_band if attempt < 8 else "header"
                positions = set()
                while len(positions) < faults:
                    choice = rng.choice(tuple(bands)) if band == "mixed" else band
                    low, high = bands[choice]
                    if choice == "text" and rng.random() < .5:
                        low, high = entry_offset, min(entry_offset + 256, high)
                    positions.add(rng.randrange(low, high))
                patches = [BytePatch(p, raw[p:p + 1], bytes([raw[p] ^ (1 << rng.randrange(8))])) for p in sorted(positions)]
                damaged = apply_patches(raw, patches, digest(raw)); failed = []
                for ci in (1, 2, 3, 0):
                    case = cases[ci]
                    result = runner.run(damaged, case)
                    if not same_result(result, reference[ci]):
                        failed.append(ci)
                        break
                counters["mutation_attempts"] += 1
                if failed:
                    accepted = {"id": f"{name}/{index}", "program": name, "image": digest(raw),
                                "patches": [p.json() for p in patches], "fault_count": faults,
                                "band": band, "requested_band": requested_band, "observed_failures": failed,
                                "damaged_sha256": digest(damaged)}
                    counters[f"accepted_{band}"] += 1
                    break
            if accepted is None:
                raise RuntimeError(f"could not construct an observed failure: {name}/{index}")
            task_splits[split].append(accepted)
        print("REAL_EXECUTABLE", name, split, len(raw), per_program, flush=True)
    write(out / "executables.json", manifest)
    for split, items in task_splits.items(): jsonl(out / "tasks" / f"{split}.jsonl", items)
    generation = {split: [] for split in task_splits}
    specs = []
    for status in range(256):
        split = "test" if status % 8 == 4 else "validation" if status % 8 == 0 else "train"
        specs.append((split, {"exit": status}))
    for split, count, seed in (("train", 512, 11), ("validation", 32, 19), ("test", 64, 29)):
        gr = random.Random(seed)
        for _ in range(count):
            payload = bytes(gr.choice(b"abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*") for _ in range(8))
            specs.append((split, {"stdout": payload.hex()}))
    for index, (split, spec) in enumerate(specs):
        raw = compile_generation_target(spec); info = inspect_elf(raw)
        result = runner.run(raw, Case())
        expected_exit, expected_stdout = (spec["exit"], "") if "exit" in spec else (0, spec["stdout"])
        if result["status"] != "ok" or result["returncode"] != expected_exit or result["stdout"] != expected_stdout:
            raise ValueError("linker-built generation reference failed actual OS execution")
        path = out / "generation" / (digest(raw) + ".elf")
        if not path.exists(): path.write_bytes(raw); path.chmod(0o400)
        generation[split].append({"id": f"generation/{index}", "spec": spec, "sha256": digest(raw), "size": len(raw)})
    for split, items in generation.items(): jsonl(out / "generation" / f"{split}.jsonl", items)
    protocol = {"id": "rgb-real-elf-v2", "program_count": len(manifest), "per_program": per_program,
                "split_programs": {split: [p["name"] for p in manifest if p["split"] == split] for split in task_splits},
                "task_counts": {split: len(items) for split, items in task_splits.items()},
                "fault_histogram": dict(Counter(str(t["fault_count"]) for ts in task_splits.values() for t in ts)),
                "construction": dict(counters), "native_calls": runner.native_calls, "exact_cache_hits": runner.cache_hits,
                "preflight_rejections": runner.preflight_rejections,
                "generation_counts": {split: len(items) for split, items in generation.items()},
                "manifest_sha256": sha(out / "executables.json"),
                "tasks": {split: sha(out / "tasks" / f"{split}.jsonl") for split in task_splits},
                "generation": {split: sha(out / "generation" / f"{split}.jsonl") for split in generation},
                "wall_seconds": time.perf_counter() - started, "test_model_queries": 0,
                "visible_cases": 4, "hidden_cases": 8,
                "scope": "real installed ELF fault injection; artificial faults, not naturally occurring bugs",
                "generation_scope": "complete learned-output ELF for exit-status/eight-byte-stdout curricula; no arbitrary-program claim"}
    write(out / "corpus.json", protocol)
    print("CORPUS", json.dumps({k: protocol[k] for k in ("task_counts", "generation_counts", "native_calls", "wall_seconds")}), flush=True)


def damaged_image(out, task):
    raw = (Path(out) / "objects" / (task["image"] + ".elf")).read_bytes()
    patches = [BytePatch(p["offset"], bytes.fromhex(p["before"]), bytes.fromhex(p["after"])) for p in task["patches"]]
    return apply_patches(raw, patches, task["image"]), raw
