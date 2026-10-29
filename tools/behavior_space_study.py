"""Reproduce the optional finite-behavior research experiment.

Inputs and raw results remain outside the repository. Requires a v6 archive
root with native reference tables and the train/dev/test splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightnest_rgb.behavior_space import BehaviorSpace, InconsistentEvidence, fit_penalty, transitions


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(v6: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    train = np.load(v6 / "data/train.npz", allow_pickle=False)["tables"]
    raw = (v6 / "data/native_reference_tables.bin").read_bytes()
    if len(raw) != 13*2*256*256:
        raise ValueError("unexpected reference table shape")
    tables = np.frombuffer(raw, np.uint8).reshape(13, 2, 256, 256)
    if not np.isin(tables, (0, 1)).all():
        raise ValueError("reference labels are not Boolean")
    seen = set()
    for folder in ("data", "phase2/data"):
        for split in ("train", "dev", "test"):
            for row in np.load(v6/folder/f"{split}.npz", allow_pickle=False)["tables"]:
                seen.add(row.tobytes())
    targets, manifest = [], []
    for family in (9, 10, 11, 12):
        candidates = []
        for sign in range(2):
            for offset in range(256):
                row = tables[family, sign, offset]
                if row.tobytes() not in seen:
                    seen.add(row.tobytes())
                    candidates.append((sign, offset, row.copy()))
        chosen = np.random.default_rng(71000 + family).choice(len(candidates), 16, replace=False)
        for index in chosen:
            sign, offset, row = candidates[index]
            targets.append(row)
            manifest.append(dict(family=family, sign=sign, offset=offset,
                                 sha256=digest(row.tobytes())))
    return train, np.stack(targets), manifest


def nearest(evidence: np.ndarray) -> np.ndarray:
    """Circular nearest-observed rule; ties choose the lower input index."""
    ids = np.flatnonzero(evidence >= 0)
    if not len(ids):
        raise ValueError("nearest requires at least one observation")
    d = np.abs(np.arange(len(evidence))[:, None] - ids[None, :])
    d = np.minimum(d, len(evidence)-d)
    return evidence[ids[d.argmin(axis=1)]].astype(np.uint8)


def evaluate(targets: np.ndarray, beta: float, output: Path) -> list[dict]:
    rows = []
    spaces = {"entropy_learned": BehaviorSpace(256, 12, beta),
              "entropy_uniform": BehaviorSpace(256, 12, 0),
              "random": BehaviorSpace(256, 12, beta)}
    log = (output/"traces.jsonl").open("w", encoding="utf-8")
    try:
        for target_id, truth in enumerate(targets):
            for mask_id in range(4):
                order = np.random.default_rng(880000 + target_id*17 + mask_id).permutation(256)
                for initial in (32, 128):
                    base = np.full(256, -1, dtype=np.int8)
                    base[order[:initial]] = truth[order[:initial]]
                    for policy, space in spaces.items():
                        known = base.copy()
                        queries = []
                        cpu0 = time.process_time()
                        wall0 = time.perf_counter()
                        for step in range(33):
                            if step in (0, 8, 16, 32):
                                post = space.posterior(known)
                                pred = space.decode(known)
                                row = dict(target=target_id, mask=mask_id, initial=initial,
                                           policy=policy, extra=step, success=bool(np.array_equal(pred, truth)),
                                           nearest_success=bool(np.array_equal(nearest(known), truth)),
                                           mismatched_bits=int(np.count_nonzero(pred != truth)),
                                           log2_count=post.log2_behavior_count,
                                           conditional_unique=post.conditional_unique,
                                           structural_uncertainty=int(post.possible.all(axis=1).sum()),
                                           queried_labels=len(queries), cpu_seconds=time.process_time()-cpu0,
                                           wall_seconds=time.perf_counter()-wall0,
                                           prediction_hex=np.packbits(pred).tobytes().hex())
                                rows.append(row)
                            if step == 32:
                                break
                            if policy == "random":
                                unseen = order[initial:]
                                query = int(unseen[step])
                            else:
                                post = space.posterior(known)
                                query = post.choose_query()
                                if query is None:
                                    # This is class-conditional agreement, not a global guarantee.
                                    # No extra oracle calls are made after it.
                                    continue
                            if known[query] >= 0:
                                raise AssertionError("query policy requested an already observed label")
                            answer = int(truth[query])  # the ONLY target access used for acquisition
                            known[query] = answer
                            queries.append([query, answer])
                        log.write(json.dumps(dict(target=target_id, mask=mask_id, initial=initial,
                                                  policy=policy, initial_inputs=order[:initial].tolist(),
                                                  queries=queries)) + "\n")
            if target_id % 8 == 7:
                print(f"Completed {target_id+1}/{len(targets)} behaviors", flush=True)
        return rows
    finally:
        log.close()


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for policy in ("entropy_learned", "entropy_uniform", "random"):
        for initial in (32, 128):
            for extra in (0, 8, 16, 32):
                subset = [r for r in rows if (r["policy"],r["initial"],r["extra"]) ==
                          (policy,initial,extra)]
                out.append(dict(policy=policy, initial=initial, extra=extra, n=len(subset),
                                successes=sum(r["success"] for r in subset),
                                nearest_successes=sum(r["nearest_success"] for r in subset),
                                conditional_unique=sum(r["conditional_unique"] for r in subset),
                                mean_queries=float(np.mean([r["queried_labels"] for r in subset])),
                                mean_cpu_seconds=float(np.mean([r["cpu_seconds"] for r in subset])),
                                mean_log2_count=float(np.mean([r["log2_count"] for r in subset]))))
    return out


def misspecification(targets: np.ndarray, beta: float) -> list[dict]:
    out = []
    space = BehaviorSpace(256, 4, beta)
    for i, truth in enumerate(targets):
        order = np.random.default_rng(991000+i).permutation(256)
        known = np.full(256, -1, np.int8)
        known[order[:32]] = truth[order[:32]]
        first_failure = None
        unique_wrong = False
        for step in range(33):
            try:
                p = space.posterior(known)
                pred = space.decode(known)
            except InconsistentEvidence:
                first_failure = step
                break
            unique_wrong |= p.conditional_unique and not np.array_equal(pred, truth)
            q = p.choose_query()
            if q is None or step == 32:
                break
            known[q] = truth[q]
        out.append(dict(target=i, true_transitions=transitions(truth),
                        contradiction_step=first_failure, conditional_unique_but_wrong=bool(unique_wrong)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    train, targets, manifest = prepare(args.v6_root)
    beta = fit_penalty(train, 12)
    protocol = dict(schema="lightnest-behavior-space-v7", date_utc="2026-10-29",
                    train_count=len(train), test_count=len(targets), mask_count=4,
                    initial_budgets=[32,128], max_new_labels=32, transition_cap=12,
                    penalty_fit="finite-ring maximum likelihood, training-only",
                    fitted_penalty=beta, policies=["entropy_learned","entropy_uniform","random"],
                    primary="paired success difference: learned entropy vs random at 32+32",
                    decoder="identical minimum-transition consistent decoding for all policies",
                    native_reference_sha256=digest((args.v6_root/"data/native_reference_tables.bin").read_bytes()),
                    train_sha256=digest(train.tobytes()), test_sha256=digest(targets.tobytes()),
                    source_sha256={p.name:digest(p.read_bytes()) for p in [
                        Path(__file__), Path(__file__).resolve().parents[1]/"lightnest_rgb/behavior_space.py"]},
                    preregistration="local freeze before measurement, not external preregistration",
                    not_neural_training=True, oracle_access="only requested labels; full domain scoring is read-only",
                    assumptions="target in declared class; deterministic noiseless oracle",
                    inherited_training="v6 training data, no inherited weights in structured method",
                    cpu_budget="equal label budgets; inference compute is NOT equal",
                    python=sys.version, platform=platform.platform(), numpy=np.__version__)
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.savez_compressed(args.out/"data.npz", train=train, test=targets)
    rows = evaluate(targets,beta,args.out)
    (args.out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary = summarize(rows)
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.out/"misspecification.json").write_text(json.dumps(misspecification(targets,beta),indent=2)+"\n")
    (args.out/"runtime.json").write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start),indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
