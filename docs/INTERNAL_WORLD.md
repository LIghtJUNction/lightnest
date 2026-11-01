# Real-architecture internal world experiment

LightNest is exploring a model that proposes a complete Linux x86-64 ELF, evaluates it using tensor operations inside its forward pass, and uses the result to propose another candidate. There is no custom instruction set and no disk image. This is a small research prototype, not a complete computer inside a language model.

```text
structured request + expected output
       ↓
learned ELF byte proposal
       ↓
ELF segment loading → x86-64 subset → limited Linux predictions
       ↓                         (tensor operations on CPU or GPU)
pass/fail feedback → learned next proposal → select passing bytes
       ↓
independent native execution for final evaluation
```

The model's internal loop does not start an external emulator or a subprocess. It does use a fixed Python loop to launch tensor operations, so it is not a CPU implemented entirely in neural-network weights and it still incurs host and GPU kernel-launch overhead. An optional CUDA Graph captures the loader and execution subset for a separate benchmark; it is not integrated into the natural-language commands.

## What exists

- `architecture/elf_tensor_loader.py` maps a bounded static x86-64 ELF into tensors, including load permissions and entry point. It rejects unsupported files. The experiment uses at most 1,024 output bytes and a bounded memory buffer, not a general Linux process image.
- `architecture/x86_internal.py` implements a small set of actual x86-64 instructions, including register moves, XOR, RIP-relative LEA, add, jumps and `syscall`. Write to standard output/error and exit have deterministic subset behavior. Unsupported instructions or invalid state do not count as success. There is no stack setup, argv, dynamic linker, general Linux kernel, or complete instruction/flag coverage.
- `architecture/ptx_internal.py` implements the semantics of actual PTX `add.u32` and `mul.lo.u32` on tensor lanes. Tests compare these operations with PTX run on a physical NVIDIA GPU. This is not a general GPU, PTX parser, or kernel runtime and it is not used by the ELF generation experiment.
- `architecture/linux_environment.py` contains a learned predictor for a narrow `openat` experiment. It predicts one of fd 3, ENOENT or EACCES for a controlled set of file names and flags. It has no file descriptor table, file contents, general filesystem state or comprehensive permission model. Its outputs are predictions, not actual Linux guarantees. With no predictor configured, this syscall is unsupported.
- `architecture/x86_world.py` combines a learned complete-file proposer, the loader and x86 subset, and a trainable feedback controller. Each round compares predicted output and exit code with a visible expected case and retains the first candidate that passes. The request is a structured vector, not natural language or source code. Argmax and execution are discrete: the experiments train byte predictions with supervised losses and do not backpropagate through machine execution.

These modules are independent of `create`, `explain` and `edit`, which currently call Ollama. Qwen3 has not been trained with this internal world, and the internal world is not yet a backend for those commands.

## Local evidence and its limits

The CPU and CUDA tests compare hand-written x86 programs and 32 randomly parameterized programs from the same skeleton against separately executed native Linux ELFs. They also test unsupported and invalid states. A test checks that two model rounds do not call a subprocess. PTX integer operations are checked against the physical GPU when CUDA is available.

A CUDA experiment trained on eight two-character requests, used four for validation and held out four combinations. All programs used the same 744-byte ELF skeleton. Two supervised proposal rounds received internal execution feedback; the feedback controller received nonzero gradients. Three of four held-out combinations passed both the internal check and final isolated native execution; all three matched the reference bytes. The first and second rounds each passed the same three cases, so this experiment does **not** demonstrate learned repair. It is too small to establish generalization to other programs, instructions, real applications or natural language.

Separately, a small `openat` classifier scored 60/60 on held-out names with the same predictable naming pattern used in its training fixture. This is not an independent test of arbitrary filesystems. The model and experiments are research code; internal acceptance is not a guarantee that a generated executable behaves correctly outside the limited test case.

A separate local timing measured only the ELF tensor loader and a 16-step x86 rollout, without the neural proposal or data transfer. On this shared RTX 4060 laptop, the median of 10 runs for batch 32 was 38.01 ms on CPU, 31.06 ms with ordinary GPU launches and 5.60 ms with a captured CUDA Graph; the graph capture itself took 52.21 ms. These are small, noisy measurements, not end-to-end application latency or a comparison against an external emulator. The benchmark checks that a replay reads changed input.

The experiment artifacts, checkpoints, source snapshots and reports from this machine are outside the repository under `~/Documents/LightNest-artifacts/`. Their datasets, dependencies and prior test inspection limit what can be concluded. Prior experiment reports are not a substitute for an unseen benchmark.

## Reproduce a small GPU experiment

Use the separate CUDA environment described in the [README](../README.md#gpu-generator-training). The following creates fixture observations once and trains the model on GPU; final evaluation runs candidates independently in the project's isolated native runner. `OUT` must be a new directory outside the checkout, and the native runner requires bubblewrap and libseccomp.

```bash
GPU_PY=/path/outside/checkout/lightnest-cuda-env/bin/python
OUT=/path/outside/checkout/lightnest-internal-experiment
"$GPU_PY" -m lightnest.training.openat_experiment --out "$OUT-openat" --device cuda
"$GPU_PY" -m lightnest.training.x86_world_feedback_experiment \
  --out "$OUT-world" --device cuda --steps 200 --alphabet LMNO \
  --environment-checkpoint "$OUT-openat/environment.pt" \
  --observations "$OUT-openat/observations.json"
```

These commands use a tiny generated fixture and take GPU time; they do not create a general-purpose trained model. The environment labels are collected before training from controlled host files. During the model's iterative forward pass, no external execution is used; native execution is an independent final check.
