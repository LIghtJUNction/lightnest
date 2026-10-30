"""Runtime for the self-contained LightNest ELF.

The executable observes its own bytes with the embedded Any2Any model and
passes those observations to the embedded language model.  No sentence is
assembled from a Python template at runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib import request

import torch

from lightnest.architecture.any2any import ArtifactCore, ArtifactRequest, KINDS
from lightnest.runtime.artifact import inspect_elf


WRAPPER_MAGIC = b"LIGHTNEST_WRAPPER_V1"


def _embedded(name):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    path = root / name
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def _read_self_payload():
    """Read the embedded base ELF and locate the raw GGUF payload."""
    executable = Path(os.environ.get("LIGHTNEST_SELF_EXE", "/proc/self/exe"))
    total = executable.stat().st_size
    footer_size = len(WRAPPER_MAGIC) + 16
    if total <= footer_size:
        raise RuntimeError("self executable is missing the wrapper footer")
    with executable.open("rb") as source:
        source.seek(total - footer_size)
        footer = source.read(footer_size)
        if footer[:len(WRAPPER_MAGIC)] != WRAPPER_MAGIC:
            raise RuntimeError("self executable is missing the wrapper footer")
        base_size, model_size = struct.unpack("<QQ", footer[len(WRAPPER_MAGIC):])
        payload_start = total - footer_size - base_size - model_size
        model_start = payload_start + base_size
        if payload_start <= 0 or model_start <= payload_start:
            raise RuntimeError("invalid embedded payload sizes")
        source.seek(payload_start)
        elf_bytes = source.read(base_size)
    return executable, elf_bytes, total, model_start, model_size


def _materialize_language_model(executable, model_start, model_size):
    temporary = tempfile.NamedTemporaryFile(prefix="lightnest-self-", suffix=".gguf", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with executable.open("rb") as source:
            source.seek(model_start)
            remaining = model_size
            while remaining:
                chunk = source.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("truncated embedded GGUF")
                temporary.write(chunk)
                remaining -= len(chunk)
        temporary.flush()
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.close()


def _sample(raw, limit):
    if len(raw) <= limit: return raw
    window = max(256, limit // 3)
    return raw[:window] + raw[len(raw) // 2:len(raw) // 2 + window] + raw[-window:]


@torch.inference_mode()
def _neural_observation(raw, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config.setdefault("request_encoding", "mean")
    if "sequence_limit" not in config: config["sequence_limit"] = payload["state_dict"]["position.weight"].shape[0]
    if "note_slots" not in config: config["note_slots"] = 32
    model = ArtifactCore(**config)
    model.load_state_dict(payload["state_dict"], strict=False); model.eval()
    window = _sample(raw, model.max_bytes)
    encoded = model.encode_chunks(window, ArtifactRequest("elf", "understand", "self"), max_chunks=32)
    encoded, context = model.contextualize(encoded)
    abstract, guidance, budget = model.think_budgeted(encoded)
    heads = model.heads(encoded, guidance, budget_logits=budget["logits"])
    kind = int(heads["kind_logits"].argmax(-1)[0])
    return {"predicted_kind": KINDS[kind] if kind < len(KINDS) else "custom",
            "summary": [round(float(value), 6) for value in heads["summary"][0]],
            "compute_budget_depth": budget["depth"],
            "context_budget_index": int(context["budget_logits"].argmax(-1)[0]),
            "context_slots": int((context["weights"][0] > 0).sum()),
            "sampled_bytes": len(window), "weights_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def _free_port():
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close(); return port


def _language_elf_observation(elf):
    """Keep repeated ELF metadata out of the language model's context."""
    fields = ("sha256", "size", "bits", "machine", "type", "entry",
              "entry_in_executable_segment", "little_endian", "interpreter", "needed")
    view = {key: elf[key] for key in fields}
    view.update(segment_count=len(elf["segments"]), section_count=len(elf["sections"]),
                import_count=len(elf["imports"]), export_count=len(elf["exports"]))
    return view


def _ask_language_model(server, prompt):
    payload = json.dumps({"model": "lightnest-embedded", "input": prompt,
                          "temperature": 0.1, "max_output_tokens": 512}).encode()
    req = request.Request(f"http://127.0.0.1:{server}/v1/responses", data=payload,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as response:
        value = json.loads(response.read())
    text_parts = [part["text"] for item in value.get("output", [])
                  for part in item.get("content", []) if part.get("type") == "output_text"]
    text = "".join(text_parts)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    return text.strip()


def main():
    checkpoint = _embedded("self-any2any.pt")
    llama = _embedded("llama-server")
    executable, raw, container_bytes, model_start, model_size = _read_self_payload()
    configured_language_model = os.environ.get("LIGHTNEST_LANGUAGE_MODEL")
    owns_language_model = not configured_language_model
    language_model = (Path(configured_language_model) if configured_language_model else
                      _materialize_language_model(executable, model_start, model_size))
    elf = inspect_elf(raw)
    neural = _neural_observation(raw, checkpoint)
    port = _free_port()
    root = str(Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = root + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["GGML_BACKEND_PATH"] = root
    process = subprocess.Popen([str(llama), "--model", str(language_model), "--host", "127.0.0.1",
                                "--port", str(port), "--ctx-size", "4096", "--threads", "4", "--no-warmup",
                                "--alias", "lightnest-embedded"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(120):
            try:
                with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200: break
            except Exception: time.sleep(.5)
        else:
            detail = ""
            if process.poll() is not None and process.stderr is not None:
                detail = process.stderr.read().strip()[-2000:]
            raise RuntimeError("embedded language runtime did not become ready" + (f": {detail}" if detail else ""))
        language_elf = _language_elf_observation(elf)
        language_elf.update(elf_payload_bytes=len(raw), container_bytes=container_bytes,
                             embedded_language_bytes=model_size,
                             elf_payload_mib=round(len(raw) / 2**20, 3),
                             container_gib=round(container_bytes / 2**30, 3),
                             embedded_language_gib=round(model_size / 2**30, 3))
        prompt = ("你是内嵌在 LightNest 可执行文件中的语言推理模型。请根据观测解释你正在运行的文件。"
                  "用中文写，最多 300 字，严格分三段，并且分别以“已知事实：”“模型推断：”“不能声称的内容：”开头。"
                  "第一段说明 ELF、架构、ELF主体大小、单文件总大小、入口是否在可执行段；大小优先使用观测给出的 MiB/GiB，"
                  "不要自行改写长字节数字。第二段必须逐项写出："
                  "Any2Any预测类型、采样字节数、上下文槽数、思考深度、内嵌权重和语言模型；这些都是直接观测到的事实，"
                  "不能在第三段否定。第三段只说明无法从这些观测推出的业务功能、训练语料和源代码意图。"
                  "不要复述 JSON，不要编造观测没有给出的功能。\n\n"
                  f"ELF观测：{json.dumps(language_elf, ensure_ascii=False)}\n"
                  f"神经观测：{json.dumps(neural, ensure_ascii=False)}\n"
                  f"内嵌语言模型文件大小：{language_model.stat().st_size} 字节")
        explanation = _ask_language_model(port, prompt)
        if len(explanation) < 40: raise RuntimeError("embedded model produced no usable explanation")
        print(explanation)
        print(f"\n[embedded_any2any_sha256={neural['weights_sha256']} self_bytes={container_bytes}]")
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill()
        if owns_language_model:
            language_model.unlink(missing_ok=True)
        base_executable = os.environ.get("LIGHTNEST_BASE_EXE")
        if base_executable:
            Path(base_executable).unlink(missing_ok=True)


if __name__ == "__main__": main()
