"""Durable WGT artifacts and compatibility payloads for existing host entrypoints."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from weightlang.modules import load_module, save_module, restore_parameters
from safetensors.torch import load_file


def model_kwargs(model):
    name = type(model).__name__
    if name == "RGBPolicy":
        return {"config": model.config}
    if name in {"ELFGenerator", "HNetELFGenerator"}:
        result = {"width": model.width, "capacity": model.capacity}
        if name == "HNetELFGenerator":
            result.update(chunk_mode=model.decode.mode, target_ratio=model.decode.target_ratio)
        return result
    if name == "ArtifactCore":
        return {"width": model.width, "heads": model.encoder.layers[0].self_attn.num_heads,
                "max_bytes": model.max_bytes, "sequence_limit": model.sequence_limit,
                "red_depths": model.red_depths, "note_slots": model.notebook.slots,
                "request_encoding": model.request_encoding, "semantic_chunk_size": model.semantic_chunk_size,
                "semantic_bpe": model.semantic_bpe}
    raise ValueError(f"provide explicit constructor kwargs for {name}")


def save_model(model, path, *, metadata=None, kwargs=None, origin="edited", training=None):
    """Save a new complete model directory without overwriting retained outputs."""
    if training is not None:
        training = dict(training)
    if metadata is not None:
        # Provenance is caller reported, separate from whether training ran.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
    artifact = save_module(model, path, kwargs=model_kwargs(model) if kwargs is None else kwargs,
                                  origin=origin, training=training, metadata=metadata)
    return artifact


def load_model(path):
    # The project loader explicitly opts into its executable WGT model format.
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if manifest.get("format") == "weightlang.tensor-program":
        return _load_legacy_model(Path(path), manifest)
    artifact = load_module(path, trusted=True)
    metadata = artifact.manifest.get("application_metadata", {})
    return artifact.module, metadata


def checkpoint_digest(path):
    path = Path(path)
    return hashlib.sha256((path / "manifest.json" if path.is_dir() else path).read_bytes()).hexdigest()


def load_payload(path):
    """Expose a WGT model as the legacy metadata/state view for host consumers."""
    path = Path(path)
    if not path.is_dir():
        return torch.load(path, map_location="cpu", weights_only=True)
    model, metadata = load_model(path)
    config = model_kwargs(model)
    name = type(model).__name__
    if name == "RGBPolicy":
        config = asdict(model.config)
    elif name == "ArtifactCore":
        pass
    else:
        config = dict(config)
    return {"architecture": "lightnest-any2any-v1" if name == "ArtifactCore" else "lightnest-rgb-full-elf-v1",
            "kind": "artifact-core" if name == "ArtifactCore" else "policy" if name == "RGBPolicy" else "generator",
            "config": config, "state_dict": model.state_dict(), "metadata": metadata}


def load_any2any(path, *, strict=False):
    if Path(path).is_dir():
        model, metadata = load_model(path)
        if type(model).__name__ != "ArtifactCore":
            raise ValueError("checkpoint is not an Any2Any artifact core")
        return model.eval(), metadata
    from lightnest.architecture.any2any import ArtifactCore
    payload = load_payload(path)
    if payload.get("architecture") != "lightnest-any2any-v1" or payload.get("kind") != "artifact-core":
        raise ValueError("checkpoint is not an Any2Any artifact core")
    config = dict(payload["config"])
    config.setdefault("request_encoding", "mean")
    config.setdefault("sequence_limit", payload["state_dict"]["position.weight"].shape[0])
    config.setdefault("note_slots", 32)
    model = ArtifactCore(**config)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    allowed = {name for name in missing if name.startswith(("copy_query.", "budget.", "notebook."))}
    if unexpected or set(missing) != allowed:
        raise ValueError(f"Any2Any checkpoint mismatch: {missing} {unexpected}")
    return model.eval(), payload.get("metadata", {})


# Exact source snapshots supported by the previous LightNest migration. Legacy
# source is checked as data, then its tensors bind to the migrated architecture.
_LEGACY_SOURCE_HASHES = {
    "lightnest.architecture." + Path(name).stem: digest
    for name, digest in json.loads(
        (Path(__file__).with_name("wgt") / "legacy-source-hashes.json").read_text()
    )["sha256"].items()
    if name.endswith(".wgt")
}


def _legacy_config(value):
    if isinstance(value, list):
        return [_legacy_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$tuple"}:
        return tuple(_legacy_config(item) for item in value["$tuple"])
    if "$dataclass" in value:
        if set(value) != {"$dataclass", "fields"} or value["$dataclass"] != "lightnest.architecture.model:RGBConfig":
            raise ValueError("unsupported legacy model configuration")
        from lightnest.architecture.model import RGBConfig
        return RGBConfig(**_legacy_config(value["fields"]))
    return {name: _legacy_config(item) for name, item in value.items()}


def _load_legacy_model(path, manifest):
    if manifest.get("format_version") != 1 or manifest.get("args"):
        raise ValueError("unsupported legacy model format or positional arguments")
    sources = manifest.get("sources", {})
    if not sources or manifest.get("host") not in sources:
        raise ValueError("legacy model lacks its complete source snapshots")
    for host, source in sources.items():
        actual = hashlib.sha256(source.encode()).hexdigest()
        if _LEGACY_SOURCE_HASHES.get(host) != actual or manifest.get("source_hashes", {}).get(host) != actual:
            raise ValueError(f"unrecognized legacy source snapshot: {host}")
    host = manifest["host"]
    if hashlib.sha256((path / "source.wgt").read_bytes()).hexdigest() != _LEGACY_SOURCE_HASHES[host]:
        raise ValueError("legacy source digest mismatch")
    if manifest.get("weights", {}).get("file") != "weights.safetensors":
        raise ValueError("invalid legacy weight file")
    weight_file = path / "weights.safetensors"
    if hashlib.sha256(weight_file.read_bytes()).hexdigest() != manifest["weights"]["sha256"]:
        raise ValueError("legacy weight digest mismatch")
    import importlib
    module = importlib.import_module(host)
    cls = getattr(module, manifest["model"], None)
    if cls is None or getattr(cls, "_wgt_definition", {}).get("torch_module", {}).get("host") != host:
        raise ValueError("unsupported legacy model class")
    values = load_file(str(weight_file))
    specs = manifest["weights"]["tensors"]
    if values.keys() != specs.keys():
        raise ValueError("legacy weight keys mismatch")
    for name, value in values.items():
        spec = specs[name]
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"legacy tensor contract mismatch: {name}")
    kwargs = _legacy_config(manifest["kwargs"])
    buffers = {}
    if manifest["model"] == "X86World":
        from lightnest.architecture.elf_tensor_loader import ELFTensorLoader
        from lightnest.architecture.x86_internal import X86Internal
        memory_size = kwargs.get("memory_size", 4096)
        output_size = kwargs.get("output_size", 64)
        buffers["output_offsets"] = torch.arange(output_size, dtype=torch.long)
        loader = ELFTensorLoader(memory_size=memory_size)
        machine = X86Internal(memory_size=memory_size, output_size=output_size)
        buffers.update({"loader." + name: value for name, value in loader.named_buffers()})
        buffers.update({"x86." + name: value for name, value in machine.named_buffers()})
    model = restore_parameters(cls, values, kwargs=kwargs, buffers=buffers)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(specs[name].get("requires_grad", True))
    return model, manifest.get("application_metadata", {})
