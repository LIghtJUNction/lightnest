"""Prepare audited historical TRAIN data and freeze separate evaluation groups."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
LEGACY = ROOT / "archive/legacy"


def sha(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def write_rows(path: Path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def read_rows(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def legacy_runtime():
    base = LEGACY / "baseline/LightNest_RGB_v2"
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    from rgb.runtime import common, configure_torch
    from lightnest_data.core import neutral_fingerprint
    from lightnest_data.generator import generate_pair, FAMILIES
    from study.prepare import simple_pair
    return common, configure_torch, neutral_fingerprint, generate_pair, FAMILIES, simple_pair


def output_directory(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("research outputs must be outside the repository")
    return path


def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}


def verify_preparation(out: Path):
    protocol = json.loads((out / "protocol.json").read_text())
    for name, digest in protocol["data"].items():
        if sha(out / "data" / name) != digest:
            raise ValueError(f"prepared data changed: {name}")
    for name, digest in protocol["parents"].items():
        if sha(out / "parents" / name) != digest:
            raise ValueError(f"parent checkpoint changed: {name}")
    return protocol


def prepare(hybrid: Path, dataset: Path, parents: Path, out: Path,
            train_rows=32768, validation_rows=512, test_groups=16):
    out = output_directory(out)
    if train_rows < 2 or validation_rows < 2 or test_groups < 1:
        raise ValueError("preparation sizes must be positive")
    if sha(hybrid) != "234ee0f38c546207195a3fb1ac968d45592fb35b51f84d1777d90c2c4051f079":
        raise ValueError("Hybrid source archive does not match its recorded SHA256")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("data", "parents", "models", "training", "evaluation"):
        (out / name).mkdir()
    common, _, neutral, generate_pair, families, simple_pair = legacy_runtime()
    known, exact = set(), set()
    with zipfile.ZipFile(hybrid) as archive:
        regression_name = next(n for n in archive.namelist() if n.endswith("baseline_data/regression.jsonl"))
        regression = [json.loads(line) for line in archive.read(regression_name).decode().splitlines()]
        # All accessible historical records protect the new test, regardless of split.
        for name in archive.namelist():
            if name.endswith(".jsonl") and ("/data/" in name or "/baseline_data/" in name):
                for line in archive.read(name).decode().splitlines():
                    row = json.loads(line)
                    for key in ("binary", "target"):
                        if key in row:
                            code = bytes.fromhex(row[key]); known.add(neutral(code))
                            exact.add(hashlib.sha256(code).hexdigest())
        for seed in (104729, 130363):
            name = next(n for n in archive.namelist() if n.endswith(f"models/rgb-{seed}.pt"))
            (out / "parents" / f"c64-{seed}.pt").write_bytes(archive.read(name))
    regression_neutral = {neutral(bytes.fromhex(row[key])) for row in regression for key in ("binary", "target")}
    with zipfile.ZipFile(parents) as archive:
        for seed in (104729, 130363):
            name = next(n for n in archive.namelist() if n.endswith(f"runs/expanded-{seed}/model.pt"))
            (out / "parents" / f"expanded-{seed}.pt").write_bytes(archive.read(name))
    selected = {}
    counts = Counter()
    with zipfile.ZipFile(dataset) as archive:
        for name in archive.namelist():
            if "/data/index/" in name and name.endswith(".jsonl.gz"):
                with gzip.GzipFile(fileobj=archive.open(name)) as file:
                    for line in file:
                        row = json.loads(line)
                        known.add(row["neutral_sha256"])
                        exact.update((row["binary_sha256"], row["target_sha256"]))
        for split, limit in (("train", train_rows), ("validation_public", validation_rows)):
            heaps = {False:[], True:[]}
            names = sorted(n for n in archive.namelist() if f"/data/index/{split}/" in n and n.endswith(".jsonl.gz"))
            if not names:
                raise ValueError(f"missing normalized dataset split: {split}")
            for name in names:
                def rows(kind):
                    with gzip.GzipFile(fileobj=archive.open(name.replace("/index/", f"/{kind}/"))) as file:
                        yield from (json.loads(line) for line in file)
                for index, inp, label in zip(rows("index"), rows("inputs"), rows("labels"), strict=True):
                    if not index["record_id"] == inp["record_id"] == label["record_id"] or index["split"] != split:
                        raise ValueError("normalized archive join/split mismatch")
                    if index["neutral_sha256"] in regression_neutral:
                        counts[f"{split}_excluded_regression_structure"] += 1
                        continue
                    code, target = bytes.fromhex(inp["program_hex"]), bytes.fromhex(label["target_hex"])
                    if hashlib.sha256(code).hexdigest() != index["binary_sha256"] or hashlib.sha256(target).hexdigest() != index["target_sha256"]:
                        raise ValueError("normalized payload hash mismatch")
                    changed = code != target
                    row = {"id":index["record_id"], "group_id":index["group_id"],
                           "command":inp["request"], "binary":code.hex(), "target":target.hex(),
                           "family":index["families"][0], "neutral_sha256":index["neutral_sha256"]}
                    rank = int(hashlib.sha256(("rpn3-train-selection:"+row["id"]).encode()).hexdigest(), 16)
                    heap = heaps[changed]; item = (-rank, row["id"], row)
                    capacity = limit // 2
                    if len(heap) < capacity:
                        heapq.heappush(heap, item)
                    elif item[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, item)
                    counts[f"{split}_eligible_{'edit' if changed else 'stop'}"] += 1
            selected[split] = [item[2] for changed in (False, True) for item in sorted(heaps[changed], reverse=True)]
    train, validation = selected["train"], selected["validation_public"]
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise ValueError("TRAIN/validation groups overlap")
    # No scores or model queries influence the following test construction.
    rg = random.Random(2026102407)
    cases = [[0,0,0],[1,2,3],[-1,2,-3],[2**31-1,1,-1],[-2**31,-1,1],[7,7,0]]
    cases += [[rg.randint(-100000,100000) for _ in range(3)] for _ in range(6)]
    test, collisions = [], 0
    for fi, family in enumerate(("simple", *families, "branch_spill", "loop_spill")):
        got = attempt = 0
        while got < test_groups:
            seed = 202610240000 + fi*100000 + attempt; attempt += 1
            if attempt > 100000:
                raise RuntimeError("test group isolation exhausted")
            length = (3,8) if got < test_groups//2 else (9,14)
            rows = simple_pair(seed, got%2, "rpn3-frozen") if family == "simple" else generate_pair(family, seed, got%2, length, "rpn3-frozen")
            fps = {neutral(bytes.fromhex(r[k])) for r in rows for k in ("binary", "target")}
            digests = {hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in rows for k in ("binary", "target")}
            if fps & known or digests & exact:
                collisions += 1; continue
            known.update(fps); exact.update(digests)
            for row in rows:
                row.pop("source", None)
                row["cases"] = [[x, common.execute_a(bytes.fromhex(row["target"]), x)] for x in cases]
                from .isa import execute_b
                if any(execute_b(bytes.fromhex(row["target"]), x) != y for x, y in row["cases"]):
                    raise ValueError("new test reference interpreters disagree")
            test.extend(rows); got += 1
    for name, rows in (("train",train),("validation",validation),("regression",regression),("test",test)):
        write_rows(out / "data" / f"{name}.jsonl", rows)
    protocol = {"architecture":"lightnest-rpn3-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
                "archives":{p.name:sha(p) for p in (hybrid,dataset,parents)},
                "data":{p.name:sha(p) for p in sorted((out/"data").glob("*.jsonl"))},
                "parents":{p.name:sha(p) for p in sorted((out/"parents").glob("*.pt"))},
                "counts":{k:len(v) for k,v in (("train",train),("validation",validation),("regression",regression),("test",test))},
                "selection_counts":dict(counts), "test_collisions_rejected":collisions,
                "train_groups":len({r["group_id"] for r in train}), "test_model_queries":0,
                "test_scope":"new seeds in existing synthetic families; semantic deduplication not proved",
                "acceptance":"exact bytes + explicit STOP + two bounded interpreters; report lost successes and false edits",
                "promotion":"both seeds must beat expanded parents without losing old successes or adding false edits",
                "training":{"seeds":[104729,130363],"steps":2048,"batch":16,"lr":0.0003,"threshold":0.5,
                            "optimizer":"AdamW", "weight_decay":0.01,"clip":1.0,
                            "selection":"fixed final endpoint; validation is diagnostic, never select using final test"}}
    write_json(out / "protocol.json", protocol)
    print(json.dumps(protocol["counts"]), flush=True)
