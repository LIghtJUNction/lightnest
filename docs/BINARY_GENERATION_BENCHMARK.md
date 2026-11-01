# Comparing complete executable generation

The external approaches have different outputs: [PromptToBinaryExecutable](https://github.com/sanketkalwar/PromptToBinaryExecutable) and [Prompt2ELF](https://github.com/faustinoaq/prompt2elf) target complete x86-64 Linux ELF files; [llmtobinary](https://llmtobinary.com/) targets ANMA-1; [L-Zero](https://l-zero.yuxu.ge/) uses a textual instruction format and runtime. Do not compare their execution rates as though the environments or tasks were interchangeable.

PromptToBinaryExecutable's repository contains a `qwen_binary_generator_lora_generic/adapter_model.safetensors` LoRA adapter (Git blob `99a90a9aad77b68eb90f7fb2e5eb65f5ef463218`, 73,911,112 bytes when checked on 2026-11-01). This establishes that an adapter file is published, not that it loads, matches the documented base model, or reproduces any result. No weights were downloaded or evaluated here.

## Executable evaluation

`tools/evaluate_binary.py` evaluates a complete file using the existing isolated runner, without modifying, repairing, compiling, or attaching a container to it. Its JSON report includes image and suite SHA-256, runner policy hash, ELF parse metadata, and a per-case execution result. It reports the pass fraction and exits nonzero if any case fails. A successful parse alone is not execution success. An `isolation_error` is infrastructure failure, not evidence about generated program behavior.

Create `cases.json` (the byte strings are hexadecimal):

```json
{
  "name": "cat-example",
  "cases": [
    {"input": {"args": [], "stdin": "610a"},
     "expected": {"returncode": 0, "stdout": "610a", "stderr": ""}},
    {"input": {"args": [], "stdin": "6200"},
     "expected": {"returncode": 0, "stdout": "6200", "stderr": ""}}
  ]
}
```

```sh
uv run --locked python tools/evaluate_binary.py /usr/bin/cat cases.json > result.json
```

The runner limits resources, denies network and process creation, and supplies a restricted filesystem. Consequently a network server, a program requiring unmounted files, or software needing subprocesses cannot be assessed as functionally correct with this profile. Isolation reduces risk but is not a proof of safety. The case file is visible to the evaluator; this tool does not freeze a model or enforce a hidden-test boundary. Freeze model, prompts, case selection, generator configuration and budgets separately before a real held-out comparison. Identical tasks and comparable generation budgets are needed; multiple examples of a hand-authored program do not establish model generalization.

## External smoke check, 2026-11-01

Decoded `hex/hello.hex` from Prompt2ELF commit `803ea21a2290b145d3311c381262e1fa4e07f529` directly as hex. The resulting 167-byte image SHA-256 was `9af5552eb1aa279d770e05a0ea12d4556a2e444dfdc2b0d289aaad835a13229e`. The local runner parsed an x86-64 ET_EXEC ELF and passed 2/2 manually specified cases: no arguments and empty input, and an ignored argument with input; both require stdout `Hello, world!\n`, empty stderr and exit 0. This verifies that one published example runs under this policy. It does not evaluate a new generation, a trained model, or a held-out benchmark.

A useful next experiment is to preregister diverse input-dependent tasks, collect independent candidate files from each native-ELF method under a fixed generation and repair budget, then run a frozen suite with separately held-out inputs. Report format validity, execution failures, visible and held-out behavior, time, and generation cost separately. Do not train on or tune against the held-out suite.
