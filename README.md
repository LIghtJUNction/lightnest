<div align="center">
  <h1>LightNest</h1>
  <p><strong>Research on learning to inspect, edit, and generate complete executable files.</strong></p>
  <p>RGB recurrent architecture · full ELF images · isolated execution · evidence-first evaluation</p>

  [![Python](https://img.shields.io/badge/Python-3.13-111111?style=flat-square&logo=python)](pyproject.toml)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1%20CPU-111111?style=flat-square&logo=pytorch)](pyproject.toml)
  [![Target](https://img.shields.io/badge/training%20target-Linux%20x86--64-111111?style=flat-square)](docs/RGB_DESIGN.md)
  [![Status](https://img.shields.io/badge/status-research-111111?style=flat-square)](docs/RESULTS.md)

  [Run](#run-the-tools) · [RGB](#rgb-in-plain-terms) · [Current evidence](#current-evidence) · [Reproduce](docs/RGB_REPRODUCE.md) · [Design](docs/RGB_DESIGN.md)
</div>

> [!IMPORTANT]
> LightNest is an active research repository. It does **not** currently demonstrate arbitrary binary understanding, arbitrary program generation, complete ISA coverage, or safe execution of unknown code.

## What the project is actually testing

The current mainline gives a model the **complete bytes of an ELF executable**, not only a disassembled function, then measures four separate capabilities:

```text
inspect structure and behavior
             ↓
propose a bounded byte edit
             ↓
execute the candidate in isolation
             ↓
learn from visible outcomes

or

generate every byte of a complete executable image
```

Deterministic tools parse ELF headers, segments, sections, entry points, dependencies and imports. Those parser results are recorded separately from anything the model learns. A parser correctly locating an entry point is not evidence that the neural model understands the program.

The generator must emit the complete file. The writer does not attach an ELF header, copy a template, invoke a compiler, or silently repair malformed output after generation.

## RGB in plain terms

RGB is a division of work, not three mysterious kinds of intelligence:

| Part | Simple responsibility |
| --- | --- |
| **Green** | Read repeated byte/log content once, keep every location, and build a compact evidence bank with routes back to original detail |
| **Blue** | Make a cheap first proposal and decide whether deeper computation is justified |
| **Red** | Reuse the same evidence and recurrent state for more reasoning steps, then stop or propose another action |
| **Notebook** | Keep a learned, content-addressed working set for long files instead of truncating only by position |

```text
complete file
    ↓
Green: compress global evidence + retain exact locations
    ↓
Blue: cheap candidate / STOP decision
    ↓ difficult case
Red: query the same evidence repeatedly
    ↓
edit, execute, observe, update state
```

A committed edit invalidates affected evidence before the next observation. Compute depth is trained from task utility rather than hard-coded file-size thresholds.

The attention design is informed by [Mixture-of-Top-k Attention](https://arxiv.org/html/2602.01219v6), especially its combination of compressed global information and routed original details. LightNest is a CPU-oriented adaptation for executable files, not a reproduction of the paper's vision training or throughput claims.

Full architecture and boundaries: [RGB design](docs/RGB_DESIGN.md).

## Current evidence

The current real-data curriculum contains:

| Material | Current scope |
| --- | ---: |
| Installed command-line programs | 16 |
| Complete ELF edit tasks | 2,048 |
| Complete ELF generation samples | 864 |
| Primary training architecture | Linux x86-64 |
| Any2Any input capacity | 1 MiB |
| Local attention window | 4,096 tokens |

Faults are artificially injected. The current generation curriculum asks for very small observable programs, such as a specified exit code or an eight-byte output. These are useful controlled tasks, not a claim that the model can generate general software.

Multi-architecture encode/decode interfaces reuse iced-x86 and LLVM, but actual model training and execution currently target x86-64.

## A negative result kept visible

The external code-to-description alignment experiment changed the objective from byte reconstruction to matching complete functions with their documentation. It also fixed a request encoder that previously lost word order.

The frozen validation result did **not** pass:

| Method | Top-1 | Top-5 |
| --- | ---: | ---: |
| Previous byte-reconstruction checkpoint | 2.73% | 4.69% |
| New learned alignment model | 10.74% | 22.46% |
| Lexical TF-IDF baseline | **49.41%** | **72.07%** |

Because the model remained far below the lexical baseline, the candidate was not promoted and the external test and LightNest blind-test sets were not queried. This prevents a failed validation result from being converted into a success claim by repeatedly checking held-out data.

The result proves only that the new objective improved over the previous checkpoint on this frozen validation set while still failing the acceptance threshold. It does not establish source-code semantics or binary behavior understanding.

Protocol, hashes, sampling limits and reproduction commands: [semantic alignment experiment](docs/SEMANTIC_ALIGNMENT.md).

## Artifact and execution boundary

Generated or edited files are written to new paths; source executables remain unchanged.

```text
candidate bytes
      ↓
ELF preflight
      ↓
bubblewrap namespaces
      ↓
no network + resource limits + process-creation filter
      ↓
typed result: pass / mismatch / invalid ELF / timeout / signal / isolation failure
```

The runner uses an isolated filesystem, no network, resource limits and a seccomp policy that denies process creation. Exact result caches bind the image hash, input, arguments and execution policy.

Isolation reduces risk; it does not mathematically prove arbitrary machine code safe. Tests passing on visible inputs also do not prove full program correctness.

## Editing loop

The editor receives only:

- the original file;
- the requested change;
- visible training or evaluation examples;
- typed execution feedback from visible cases.

It does not receive hidden-test results or clean target bytes as reward. Fixed-length patches include expected-image and expected-before-byte preconditions. Before/after hashes make the action replayable.

```pseudo
candidate = model(file, request, visible_evidence)
result = isolated_execute(candidate, visible_cases)
reward = new_passes - regressions - typed_failure_penalties
update(action_that_created_candidate, reward)
```

This is bounded reinforcement from an execution environment, not unrestricted self-modifying code.

## Run the tools

Dependencies are locked for Python 3.13 and PyTorch 2.12.1 CPU:

```bash
uv sync --locked
uv run --locked python tools/check.py
```

Explore deterministic interfaces:

```bash
uv run --locked python -m lightnest_rgb --help
uv run --locked python -m lightnest_rgb inspect /usr/bin/sort
uv run --locked python -m lightnest_rgb targets
uv run --locked python -m lightnest_rgb decode \
  --architecture x86_64 \
  --hex 4889f84801f0c3
```

Training and execution tests additionally require GCC, Clang, LLD, LLVM, bubblewrap and libseccomp. Corpus construction requires the referenced local command-line programs.

Weights, original executables, caches, training logs and per-example results remain outside the repository.

## Reproduction map

| Goal | Document |
| --- | --- |
| Reproduce the current RGB pipeline | [RGB reproduction](docs/RGB_REPRODUCE.md) |
| Understand the architecture and gates | [RGB design](docs/RGB_DESIGN.md) |
| Inspect historical measured results | [Results](docs/RESULTS.md) |
| Review the failed semantic experiment | [Semantic alignment](docs/SEMANTIC_ALIGNMENT.md) |
| Inspect the self-contained ELF experiment | [Self-hosting ELF](docs/SELF_HOSTING.md) |
| Review older research branches | [Archive](archive/README.md) |
| Contribute without mixing evidence levels | [Contributing](CONTRIBUTING.md) |

## Repository map

```text
lightnest_rgb/              current RGB mainline and complete-ELF tools
archive/universal_native/   previous Universal Machine v1 work
archive/rpn3_rv32/          earlier RV32 architecture
archive/legacy/             historical experiments and evidence
tests/                      current and archived behavior tests
tools/                      unified checks and research scripts
docs/                       design, results and reproduction protocols
```

## Source and licensing notice

This repository republishes selected research files from owner-provided project archives whose original license status has not been fully verified. No new project-wide license is asserted. Review [NOTICE.md](NOTICE.md) and the archived provenance before redistribution.

iced-x86, LLVM, PyTorch, pyelftools and other dependencies remain under their own licenses.
