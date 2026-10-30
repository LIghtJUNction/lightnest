"""Measure actual complete-image observation and repeated evidence reads on CPU."""
from __future__ import annotations

import statistics
import time

import torch

from lightnest.runtime.artifact import digest
from lightnest.training.training import load_policy


@torch.inference_mode()
def benchmark(checkpoint, image, *, repeats=5, depth=16):
    torch.set_num_threads(1)
    model, _ = load_policy(checkpoint)
    results = {}
    for mode in ("dense", "compressed", "routed"):
        builds = []; reads = []; pairs = None
        for _ in range(repeats):
            model.green._content_cache.clear()
            start = time.perf_counter(); state = model.begin(image, "inspect and restore this executable")
            built = time.perf_counter(); model.advance(state, depth, mode)
            end = time.perf_counter(); builds.append(built - start); reads.append(end - built)
            n = state["observation"].instances
            fanout = n if mode == "dense" else model.config.landmarks
            if mode == "routed": fanout += model.config.landmarks + min(model.config.detail_k, n)
            pairs = {"construction": state["bank"].construction_pairs,
                     "repeated_read": depth * model.config.heads * n * fanout,
                     "routing_included_in_fanout": mode == "routed"}
        results[mode] = {"median_common_build_seconds": statistics.median(builds),
                         "median_recurrent_read_seconds": statistics.median(reads),
                         "median_total_seconds": statistics.median([a + b for a, b in zip(builds, reads, strict=True)]),
                         "score_pairs": pairs}
    model.green._content_cache.clear(); before = model.green.encoded_chunks
    first = model.begin(image, "inspect this executable"); cold = model.green.encoded_chunks - before
    before = model.green.encoded_chunks; model.begin(image, "inspect this executable"); unchanged = model.green.encoded_chunks - before
    edited = bytearray(image); edited[len(image) // 2] ^= 1
    before = model.green.encoded_chunks; model.begin(bytes(edited), "inspect this executable"); changed = model.green.encoded_chunks - before
    return {"image_sha256": digest(image), "image_bytes": len(image), "instances": first["observation"].instances,
            "unique_contents": len(first["observation"].payloads), "depth": depth, "repeats": repeats,
            "content_encodes": {"cold": cold, "unchanged_image": unchanged, "single_byte_edit": changed},
            "modes": results, "scope": "same RGB weights and common bank construction, not independently optimized dense baseline; CPU single thread; no execution validation in timing"}
