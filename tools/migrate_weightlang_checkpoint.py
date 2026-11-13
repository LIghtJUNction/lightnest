"""Convert an existing LightNest checkpoint to a new complete WGT model directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lightnest.architecture.any2any import ArtifactCore
from lightnest.architecture.weightlang_io import (
    checkpoint_digest, load_model, load_payload, model_kwargs, save_model,
)
from lightnest.training.training import load_generator, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new weights/<model> directory")

    kwargs = None
    training = None
    if args.source.is_dir():
        model, metadata = load_model(args.source)
        manifest = json.loads((args.source / "manifest.json").read_text())
        source = model.state_dict()
        training = manifest.get("training")
        if type(model).__name__ in {"RGBPolicy", "ELFGenerator", "HNetELFGenerator", "ArtifactCore"}:
            kwargs = model_kwargs(model)
        else:
            # The remaining project models have JSON scalar constructor fields.
            kwargs = manifest["kwargs"]
    elif args.source.is_file():
        payload = load_payload(args.source)
        kind = payload["kind"]
        if kind == "generator":
            model, metadata = load_generator(args.source)
        elif kind == "policy":
            model, metadata = load_policy(args.source)
        elif kind == "artifact-core":
            config = dict(payload["config"])
            config.setdefault("request_encoding", "mean")
            config.setdefault("sequence_limit", payload["state_dict"]["position.weight"].shape[0])
            model = ArtifactCore(**config)
            model.load_state_dict(payload["state_dict"], strict=True)
            metadata = payload.get("metadata", {})
        else:
            parser.error("unsupported checkpoint kind: " + kind)
        source = payload["state_dict"]
    else:
        parser.error("source must be a checkpoint file or complete model directory")

    target = model.state_dict()
    if source.keys() != target.keys():
        parser.error("checkpoint lacks current parameters; strict migration refuses random completion")
    for name, value in source.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype:
            parser.error("tensor contract differs: " + name)
    record = {
        "source": str(args.source.resolve()),
        "sha256": checkpoint_digest(args.source),
        "mapping": {name: name for name in source},
        "transformations": [],
        "claim": "exact parameter transfer; task quality requires separate evaluation",
    }
    artifact = save_model(
        model, args.output, kwargs=kwargs, metadata=metadata | {"migration": record},
        origin="imported", training=training,
    )
    restored, _ = load_model(artifact.path)
    for name, value in source.items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    print(json.dumps({
        "path": str(artifact.path), "mapped_tensors": len(source),
        "weights_sha256": artifact.manifest["weights"]["sha256"],
    }))


if __name__ == "__main__":
    main()
