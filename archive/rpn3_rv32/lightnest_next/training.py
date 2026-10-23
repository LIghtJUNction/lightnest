"""Fixed-budget training and endpoint evaluation with per-record comparisons."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from .data import (legacy_runtime, output_directory, read_rows, sha, source_hashes,
                   verify_preparation, write_json)
from .isa import execute_b
from .model import Batch, Config, PatchNetwork, load_model, save_model
from .model import supervised_loss


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def requests(rows):
    return [{"command":row["command"], "binary":row["binary"]} for row in rows]


@torch.inference_mode()
def validation_score(model, rows, batch_size=16):
    model.eval()
    score = {"total":len(rows), "exact":0, "edit_exact":0, "edits":0, "false_edits":0}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        for row, pred in zip(chunk, model.predict(requests(chunk)), strict=True):
            edit = row["binary"] != row["target"]
            exact = pred["stop"] and pred["final"] == row["target"]
            score["exact"] += int(exact)
            score["edits"] += int(edit)
            score["edit_exact"] += int(exact and edit)
            score["false_edits"] += int(not edit and bool(pred["patches"]))
    return score


def train(out: Path, seed: int):
    out = output_directory(out)
    protocol = verify_preparation(out)
    settings = protocol["training"]
    if seed not in settings["seeds"]:
        raise ValueError("seed was not frozen in preparation")
    configure(); torch.manual_seed(seed)
    rows = read_rows(out / "data/train.jsonl")
    validation = read_rows(out / "data/validation.jsonl")
    pools = [[r for r in rows if (r["binary"] != r["target"]) == changed] for changed in (False, True)]
    if not all(pools):
        raise ValueError("training must contain correct programs and edits")
    config = Config()
    model = PatchNetwork(config)
    plan = {"seed":seed, "config":asdict(config), "training":settings,
            "source":source_hashes(), "protocol_sha256":sha(out/"protocol.json"),
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "python":__import__("sys").version, "torch":torch.__version__, "device":"cpu",
            "sampling":"equal EDIT/STOP; identical per-step input stream across both model seeds",
            "test_model_queries":0}
    write_json(out / "training" / f"{seed}-plan.json", plan)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    sampler = random.Random(2026102409)
    stream = __import__("hashlib").sha256()
    started = time.perf_counter(); losses = []; history = []; seen = set()
    with (out / "training" / f"{seed}-updates.jsonl").open("x") as log:
        for step in range(settings["steps"]):
            selected = [sampler.choice(pools[j%2]) for j in range(settings["batch"])]
            sampler.shuffle(selected)
            for row in selected:
                stream.update((row["id"]+"\n").encode()); seen.add(row["id"])
            batch = Batch.encode(requests(selected), config)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits, bits = model(batch)
            loss = supervised_loss(logits, bits, batch, selected)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip"], error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
            if (step+1) % 64 == 0:
                event = {"step":step+1, "mean_loss":sum(losses)/len(losses),
                         "gradient_norm":float(norm), "wall_seconds":time.perf_counter()-started}
                log.write(json.dumps(event)+"\n");log.flush();losses=[]
                print(json.dumps({"seed":seed, **event}), flush=True)
            if (step+1) % 512 == 0:
                diagnostic = validation_score(model, validation)
                history.append({"step":step+1, **diagnostic})
                print("VALIDATION",seed,step+1,diagnostic,flush=True)
    elapsed = time.perf_counter()-started
    metadata = {"seed":seed, "updates":settings["steps"], "stream_sha256":stream.hexdigest(),
                "presentations":settings["steps"]*settings["batch"], "unique_records":len(seen),
                "wall_seconds_including_validation":elapsed, "protocol_sha256":sha(out/"protocol.json"),
                "source":plan["source"], "validation_history":history,
                "status":"research_candidate_not_promoted"}
    path = out / "models" / f"rpn3-{seed}.pt"
    save_model(path, model.eval(), metadata)
    # Optimizer and RNG state are separate from inference weights; never pickle-load untrusted files.
    torch.save({"optimizer":optimizer.state_dict(), "torch_rng":torch.get_rng_state(),
                "sampler_rng":sampler.getstate(), "updates":settings["steps"]},
               out / "training" / f"{seed}-optimizer.pt")
    metadata["checkpoint_sha256"] = sha(path)
    write_json(out / "training" / f"{seed}-complete.json", metadata)
    print("TRAINED",seed,metadata["checkpoint_sha256"],flush=True)


class ParentPredictor:
    def __init__(self, checkpoint):
        common, _, _, _, _, _ = legacy_runtime()
        from rgb.scripted_core import LoopCore
        self.common = common
        self.model = common.load_model(checkpoint).eval()
        self.loop = torch.jit.script(LoopCore(self.model).eval())

    @torch.inference_mode()
    def __call__(self, rows):
        from study.evaluate import action
        inputs, rel, ctx = self.model.encode_inputs(*self.common.encode(rows, self.model.config))
        h = self.loop(inputs, inputs, rel, ctx.valid, 64)
        scores, bits = self.model.decode_state(h, ctx)
        return [action(scores[j],bits[j],self.model.none_index) | {"blue":False,"rounds":64} for j in range(len(rows))]


def score_predictions(rows, predictions, execute_a):
    result = []
    for row, pred in zip(rows, predictions, strict=True):
        code = bytes.fromhex(pred["final"])
        numeric, agree = True, True
        for inputs, expected in row["cases"]:
            actual = []
            for vm in (execute_a, execute_b):
                try:
                    actual.append(vm(code, inputs))
                except (ValueError,IndexError):
                    actual.append(None)
            agree &= actual[0] == actual[1]
            numeric &= actual[0] == expected and actual[1] == expected
        need = bool(row["edits"])
        result.append({"id":row["id"], "initial_sha256":__import__("hashlib").sha256(bytes.fromhex(row["binary"])).hexdigest(),
                       "family":row["family"], "need_edit":need,
                       "strict":pred["stop"] and pred["final"] == row["target"] and numeric and (need or not pred["patches"]),
                       "false_edit":not need and bool(pred["patches"]), "numeric":numeric, "vm_agreement":agree,
                       "final":pred["final"], "stop":pred["stop"], "status":pred["status"], "patches":pred["patches"]})
    return result


def summary(records):
    return {"total":len(records), "strict_total":sum(r["strict"] for r in records),
            "edits":sum(r["need_edit"] for r in records),
            "strict_edits":sum(r["strict"] and r["need_edit"] for r in records),
            "false_edits":sum(r["false_edit"] for r in records),
            "all_vm_agree":all(r["vm_agreement"] for r in records)}


def comparison(parent, candidate):
    by_id = {r["id"]:r for r in parent}
    if len(by_id) != len(parent) or set(by_id) != {r["id"] for r in candidate}:
        raise ValueError("comparison task IDs differ or repeat")
    for row in candidate:
        if row["initial_sha256"] != by_id[row["id"]]["initial_sha256"]:
            raise ValueError("comparison inputs differ")
    return {"lost_success_ids":[r["id"] for r in candidate if by_id[r["id"]]["strict"] and not r["strict"]],
            "added_success_ids":[r["id"] for r in candidate if not by_id[r["id"]]["strict"] and r["strict"]],
            "added_false_edit_ids":[r["id"] for r in candidate if not by_id[r["id"]]["false_edit"] and r["false_edit"]]}


def freeze(out):
    protocol = verify_preparation(out)
    endpoints = {}
    for seed in protocol["training"]["seeds"]:
        metadata = json.loads((out/"training"/f"{seed}-complete.json").read_text())
        if metadata["source"] != source_hashes():
            raise ValueError("source changed after training")
        p = out/"models"/f"rpn3-{seed}.pt"
        if metadata["checkpoint_sha256"] != sha(p):
            raise ValueError("trained checkpoint changed")
        endpoints[p.name] = sha(p)
    write_json(out/"final-freeze.json", {"source":source_hashes(),"models":endpoints,
               "data":protocol["data"], "parents":protocol["parents"], "test_model_queries":0,
               "protocol_sha256":sha(out/"protocol.json")})


def evaluate(out: Path, seed: int):
    out = output_directory(out); configure()
    protocol = verify_preparation(out)
    frozen = json.loads((out/"final-freeze.json").read_text())
    if frozen["source"] != source_hashes() or frozen["protocol_sha256"] != sha(out/"protocol.json"):
        raise ValueError("evaluation no longer matches frozen source/protocol")
    checkpoint = out/"models"/f"rpn3-{seed}.pt"
    if sha(checkpoint) != frozen["models"][checkpoint.name]:
        raise ValueError("evaluation checkpoint changed")
    model = load_model(checkpoint)
    common, _, _, _, _, _ = legacy_runtime()
    from study.evaluate import rollout
    report = {"seed":seed, "protocol_sha256":sha(out/"protocol.json"), "splits":{}}
    for split in ("regression","test"):
        rows = read_rows(out/"data"/f"{split}.jsonl")
        predictions = []; started = time.perf_counter()
        for start in range(0,len(rows),16):
            predictions.extend(model.predict(requests(rows[start:start+16])))
        inference_seconds = time.perf_counter()-started
        records = score_predictions(rows,predictions,common.execute_a)
        write_json(out/"evaluation"/f"{seed}-{split}-rpn3.json", {"summary":summary(records),"records":records,
                   "inference_wall_seconds":inference_seconds,"checkpoint_sha256":sha(checkpoint)})
        result = {"rpn3":summary(records),"inference_wall_seconds":inference_seconds,"parents":{}}
        for kind in ("c64","expanded"):
            predictor = ParentPredictor(out/"parents"/f"{kind}-{seed}.pt")
            started = time.perf_counter()
            parent_records = rollout(predictor, rows, batch=16)
            wall = time.perf_counter()-started
            # Re-score saved endpoints through the same pair of interpreters used for RPN3.
            parent_predictions = [{"final":r["final"],"stop":r["explicit_stop"],"status":r["status"],
                                   "patches":[{"offset":a["position"],"value":a["value"]} for a in r["actions"] if not a["stop"]]}
                                  for r in parent_records]
            rescored = score_predictions(rows,parent_predictions,common.execute_a)
            if [r["strict"] for r in rescored] != [r["strict"] for r in parent_records]:
                raise ValueError("parent evaluator and independent re-score disagree")
            cmp = comparison(rescored,records)
            write_json(out/"evaluation"/f"{seed}-{split}-{kind}.json", {"summary":summary(rescored),"records":rescored,
                       "wall_seconds_including_original_score":wall,"comparison_to_rpn3":cmp})
            result["parents"][kind] = {"summary":summary(rescored),
                                      "lost_successes":len(cmp["lost_success_ids"]),
                                      "added_successes":len(cmp["added_success_ids"]),
                                      "added_false_edits":len(cmp["added_false_edit_ids"])}
            del predictor
        report["splits"][split] = result
        print("EVALUATED",seed,split,json.dumps(result),flush=True)
    report["qualified"] = all(s["parents"]["expanded"]["lost_successes"] == 0 and
                               s["parents"]["expanded"]["added_false_edits"] == 0 and
                               s["rpn3"]["strict_edits"] >= s["parents"]["expanded"]["summary"]["strict_edits"]
                               for s in report["splits"].values())
    write_json(out/"evaluation"/f"{seed}-report.json",report)
    return report
