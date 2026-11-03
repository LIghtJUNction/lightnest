"""Durable WGT artifacts and compatibility payloads for existing host entrypoints."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from weightlang.tensor_program import load_tensor_artifact, save_tensor_module


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
    artifact = save_tensor_module(model, path, kwargs=model_kwargs(model) if kwargs is None else kwargs,
                                  origin=origin, training=training, metadata=metadata)
    return artifact


def load_model(path):
    # The project loader explicitly opts into its executable WGT model format.
    artifact = load_tensor_artifact(path, trusted=True, warn_untrained=False)
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
        return model, metadata
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
