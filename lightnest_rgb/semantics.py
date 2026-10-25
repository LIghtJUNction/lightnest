"""Code/description alignment on external projects, with no parser at inference."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import warnings

import torch
from torch.nn import functional as F

from .any2any import ArtifactCore
from .corpus import external

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE = "external-code-description-contrastive-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def sources():
    return {name: sha(Path(__file__).parent / name) for name in ("any2any.py", "semantics.py")}


class _RemoveDocs(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def examples(path, root, group, *, max_bytes=1024):
    """AST supplies training pairs only; neither docs nor comments remain in code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            raw = path.read_bytes()
            if len(raw) > 300000:
                return []
            tree = ast.parse(raw)
    except (OSError, SyntaxError, ValueError):
        return []
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node) or ""
        description = " ".join(doc.split("\n\n", 1)[0].split())
        if not 6 <= len(description.split()) <= 48 or len(description.encode()) > 384:
            continue
        # Clone the subtree; stripping a parent must not erase nested labels.
        code_node = ast.parse(ast.unparse(node)).body[0]
        code = ast.unparse(_RemoveDocs().visit(code_node)).encode()
        if not 48 <= len(code) <= max_bytes:
            continue
        result.append({"id": f"{group}/{path.relative_to(root)}:{node.lineno}", "group": group,
                       "path": str(path.resolve()), "file_sha256": hashlib.sha256(raw).hexdigest(),
                       "code": code.decode(), "description": description,
                       "code_sha256": hashlib.sha256(code).hexdigest()})
    return result


def prepare(out, *, max_bytes=1024, from_corpus=None):
    out = external(out)
    if not 256 <= max_bytes <= 4096:
        raise ValueError("invalid complete-function capacity")
    if from_corpus:
        parent = external(from_corpus)
        protocol = json.loads((parent / "protocol.json").read_text())
        if protocol["max_bytes"] != max_bytes or any(sha(parent / f"{name}.jsonl") != digest
                                                    for name, digest in protocol["splits"].items()):
            raise ValueError("parent corpus capacity or identity changed")
        splits = {name: read(parent, f"{name}_pool" if f"{name}_pool" in protocol["splits"] else name)
                  for name in ("train", "validation", "test", "self")}
        return freeze(out, splits, protocol["training_roots"], max_bytes, parent=sha(parent / "protocol.json"))
    splits = {name: [] for name in ("train", "validation", "test", "self")}
    roots = [Path("/usr/share"), *sorted(Path("/usr/lib").glob("python3.*"))]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            parts = path.relative_to(root).parts
            if root.name.startswith("python3."):
                group = parts[1].split(".")[0] if parts[0] == "site-packages" and len(parts) > 1 else "python-stdlib"
            else:
                group = parts[0]
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            splits[split].extend(examples(path, root, group, max_bytes=max_bytes))
    for path in sorted((ROOT / "lightnest_rgb").glob("*.py")):
        splits["self"].extend(examples(path, ROOT, "lightnest", max_bytes=max_bytes))
    return freeze(out, splits, [str(root) for root in roots], max_bytes)


def freeze(out, splits, roots, max_bytes, *, parent=None):
    out.mkdir(parents=True, exist_ok=False)
    # Give held-out records priority, eliminating exact code and label leakage.
    seen_code, seen_text = set(), set()
    counts = {}
    files = []
    for name in ("self", "test", "validation", "train"):
        kept = []
        for row in splits[name]:
            label = " ".join(row["description"].lower().split())
            if row["code_sha256"] in seen_code or label in seen_text:
                continue
            seen_code.add(row["code_sha256"]); seen_text.add(label); kept.append(row)
        for row in kept:
            nodes = ast.walk(ast.parse(row["code"]))
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and ast.get_docstring(node) for node in nodes):
                raise ValueError(f"description remains in model input: {row['id']}")
        if len(kept) < (32 if name != "self" else 8):
            raise ValueError(f"insufficient independent {name} examples: {len(kept)}")
        available = len(kept)
        if name in {"validation", "test"} and len(kept) > 512:
            with (out / f"{name}_pool.jsonl").open("x") as file:
                for row in kept:
                    file.write(json.dumps(row) + "\n")
            files.append(f"{name}_pool")
            groups = {}
            for row in sorted(kept, key=lambda row: row["code_sha256"]):
                groups.setdefault(row["group"], []).append(row)
            kept = []
            while len(kept) < 512:
                for group in sorted(groups):
                    if groups[group] and len(kept) < 512:
                        kept.append(groups[group].pop())
        with (out / f"{name}.jsonl").open("x") as file:
            for row in kept:
                file.write(json.dumps(row) + "\n")
        files.append(name)
        counts[name] = {"examples": len(kept), "available": available, "groups": sorted({row["group"] for row in kept})}
    snapshot = out / "source"; snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    write(out / "protocol.json", {"objective": OBJECTIVE, "counts": counts, "max_bytes": max_bytes,
          "splits": {name: sha(out / f"{name}.jsonl") for name in files}, "sources": sources(),
          "uv_lock_sha256": sha(ROOT / "uv.lock"), "training_roots": roots, "parent_protocol_sha256": parent,
          "excluded_repository": str(ROOT), "test_model_queries_at_freeze": 0,
          "selection": "validation recall@1 only; test and self read after checkpoint freeze",
          "evaluation_sampling": "at most 512 per split, round robin over projects after code-hash ordering; full pools retained",
          "scope": "matching descriptions, not free-form explanation or executable behavior"})
    return counts


def verify(out):
    out = external(out)
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["sources"] != sources() or protocol["uv_lock_sha256"] != sha(ROOT / "uv.lock"):
        raise ValueError("frozen semantic implementation or dependencies changed")
    if any(sha(out / f"{name}.jsonl") != digest for name, digest in protocol["splits"].items()):
        raise ValueError("frozen semantic data changed")
    return out, protocol


def read(out, split):
    return [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("objective") not in {OBJECTIVE, 'external-binary-language-v1'}:
        raise ValueError("checkpoint lacks semantic training")
    model = ArtifactCore(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


@torch.inference_mode()
def embed(model, values, batch_size=8):
    # Length sorting changes only padding work, never truncates a source.
    order = sorted(range(len(values)), key=lambda index: len(values[index]))
    chunks = []
    for start in range(0, len(order), batch_size):
        chunks.append(model.semantic_embedding([values[index] for index in order[start:start + batch_size]]))
    embeddings = torch.cat(chunks)
    inverse = torch.tensor(order).argsort()
    return embeddings[inverse]


def metrics(scores):
    target = torch.arange(scores.shape[0])
    ranks = scores.argsort(dim=1, descending=True, stable=True).argsort(dim=1).diag() + 1
    return {"examples": len(target), "candidates": scores.shape[1],
            "recall_at_1": float((scores.argmax(1) == target).float().mean()),
            "recall_at_5": float((ranks <= min(5, scores.shape[1])).float().mean()),
            "mrr": float(ranks.float().reciprocal().mean())}


def lexical_scores(rows):
    def words(text):
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower().replace("_", " ")
        return Counter(re.findall(r"[a-z]{2,}", text))
    codes = [words(row["code"]) for row in rows]
    docs = [words(row["description"]) for row in rows]
    vocabulary = sorted(set().union(*(set(row) for row in docs)))
    df = Counter(word for row in docs for word in row)
    idf = torch.tensor([math.log((1 + len(rows)) / (1 + df[word])) + 1 for word in vocabulary])
    def vectors(values):
        return F.normalize(torch.tensor([[row[word] for word in vocabulary] for row in values]).float() * idf, dim=1)
    return vectors(codes) @ vectors(docs).T


def train(out, *, steps=1024, seed=20261026, batch_size=8, width=64):
    out, protocol = verify(out)
    if steps < 1 or batch_size < 2:
        raise ValueError("invalid semantic training budget")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    folder = out / f"training-{seed}"; folder.mkdir(exist_ok=False)
    config = {"width": width, "heads": 4, "max_bytes": protocol["max_bytes"],
              "sequence_limit": max(256, protocol["max_bytes"]), "note_slots": 16,
              "request_encoding": "gru", "semantic_chunk_size": 8}
    model = ArtifactCore(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    rows = read(out, "train"); validation = read(out, "validation")
    rng = random.Random(seed); started = time.perf_counter(); best = -1; losses = []
    write(folder / "plan.json", {"steps": steps, "seed": seed, "batch_size": batch_size,
          "config": config, "protocol_sha256": sha(out / "protocol.json"), "sources": sources(),
          "parameters": sum(p.numel() for p in model.parameters()), "test_model_queries": 0})
    with (folder / "updates.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            chosen = rng.sample(rows, batch_size)
            model.train()
            code = model.semantic_embedding([row["code"].encode() for row in chosen])
            text = model.semantic_embedding([row["description"].encode() for row in chosen])
            scores = code @ text.T / .07
            target = torch.arange(batch_size)
            loss = (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
            if step % 32 == 0 or step == steps:
                event = {"step": step, "loss": sum(losses) / len(losses), "wall_seconds": time.perf_counter() - started}
                if step % 256 == 0 or step == steps:
                    model.eval()
                    a = embed(model, [row["code"].encode() for row in validation])
                    b = embed(model, [row["description"].encode() for row in validation])
                    event["validation"] = metrics(a @ b.T)
                    payload = {"objective": OBJECTIVE, "config": config, "state_dict": model.state_dict(),
                               "step": step, "protocol_sha256": sha(out / "protocol.json"), "sources": sources()}
                    torch.save(payload | {"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                                          "sampler_rng": rng.getstate()}, folder / f"step-{step}.pt")
                    if event["validation"]["recall_at_1"] > best:
                        best = event["validation"]["recall_at_1"]
                        # The selection receipt binds the exact checkpoint, not a mutable best.pt.
                        torch.save(payload, folder / f"candidate-{step}.pt")
                        with (folder / "selection.json").open("w") as receipt:
                            json.dump({"checkpoint": f"candidate-{step}.pt", "recall_at_1": best}, receipt)
                log.write(json.dumps(event) + "\n"); log.flush(); print(json.dumps(event), flush=True); losses = []
    return json.loads((folder / "selection.json").read_text())


def evaluate(out, checkpoint, *, baseline=None):
    out, protocol = verify(out)
    model, payload = load(checkpoint)
    if payload["protocol_sha256"] != sha(out / "protocol.json") or payload["sources"] != sources():
        raise ValueError("semantic checkpoint/protocol mismatch")
    report = {"checkpoint_sha256": sha(checkpoint), "protocol_sha256": sha(out / "protocol.json"),
              "selection_step": payload["step"], "splits": {}, "scope": protocol["scope"]}
    old = None
    if baseline:
        old_payload = torch.load(baseline, map_location="cpu", weights_only=True)
        config = dict(old_payload["config"]); config.setdefault("request_encoding", "mean")
        old = ArtifactCore(**config).eval(); old.load_state_dict(old_payload["state_dict"], strict=True)
        report["baseline_sha256"] = sha(baseline)
    for split in ("test", "self"):
        rows = read(out, split)
        code = [row["code"].encode() for row in rows]; docs = [row["description"].encode() for row in rows]
        a, b = embed(model, code), embed(model, docs); scores = a @ b.T
        result = {"neural": metrics(scores), "lexical": metrics(lexical_scores(rows)), "chance_at_1": 1 / len(rows)}
        if old:
            result["byte_reconstruction_checkpoint"] = metrics(embed(old, code, 1) @ embed(old, docs, 1).T)
        shuffled = list(range(1, len(rows))) + [0]
        result["wrong_source_control"] = metrics(scores[shuffled])
        result["rows"] = [{"id": row["id"], "expected": row["description"],
                          "prediction": rows[int(scores[i].argmax())]["description"],
                          "correct": int(scores[i].argmax()) == i} for i, row in enumerate(rows)]
        report["splits"][split] = result
    destination = Path(checkpoint).with_suffix(".evaluation.json")
    write(destination, report)
    return {split: {k: v for k, v in result.items() if k != "rows"} for split, result in report["splits"].items()}


@torch.inference_mode()
def match(model, raw, descriptions):
    if not isinstance(descriptions, (list, tuple)) or not descriptions or any(not isinstance(text, str) or not text for text in descriptions):
        raise ValueError("nonempty candidate descriptions required")
    if model.semantic_bpe is not None:
        from .binary_semantics import rank
        return rank(model, raw, descriptions)
    vectors = embed(model, [raw, *[text.encode() for text in descriptions]])
    scores = vectors[0] @ vectors[1:].T
    order = scores.argsort(descending=True).tolist()
    return [{"description": descriptions[index], "score": float(scores[index])} for index in order]
