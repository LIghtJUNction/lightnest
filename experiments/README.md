# Architecture search

[Results](RESULTS.md) · [Lessons](LESSONS.md) · [Repository structure](STRUCTURE.md) describes application, experiment and artifact ownership.

`STATUS.json` is the single current-state file. Replace its contents at phase
boundaries; never append history to it.

- `active/<experiment>/`: current protocol, version implementations and runner.
- `archive/<round>/`: frozen source, protocol, results and lessons from a completed round.
- `weights/experiments/<round>/`: durable selected model and reproducibility metadata.
- `.cache/experiments/<round>/`: disposable training work; removed after archiving.

Run from the repository root:

```sh
uv run --locked python -m experiments.loop --limit 1
```

The [active protocol](active/elf_generation/PROTOCOL.md) defines the questions and candidate versions.

The loop and repository check tool explicitly select the sibling checkout compiler
when present, while respecting an existing `WGT_COMPILER`. For direct runner or
unittest commands, export `WGT_COMPILER="$PWD/../WeightLang/WeightLang/target/debug/wgt"`.

The runner freezes data and code hashes before training, uses validation to select
an architecture and checkpoint, and evaluates the blind test after selection.
Equal steps and examples are not equal FLOPs. Compare quality, parameters and time.
Record failed hypotheses as well as successes. Retain only the winning checkpoint,
compact per-case results and source snapshots; delete this round's other temporary
weights after archiving. New rounds must state their change before looking at results.
Once exposed, a test set is an audit set, never a new blind test. Each round uses
fresh stdout requests; exit values come from a finite 256-value population.

These initial tasks cover controlled complete ELF generation, not arbitrary binary
understanding or editing. No application replacement is justified solely by them.

Retained models can emit bytes without a compiler or reference generator:

```sh
WGT_COMPILER=/nonexistent uv run --locked python -m experiments.infer \
  --round r003 --exit 7 --output /tmp/lightnest-candidate-7.elf
```

This writes a new file and reports model stopping; it does not execute it or claim
its behavior is correct. The archived audit compares every byte to independently
verified reference programs.

The selected r007 instance also has a request-bit interval audit:

```sh
WGT_COMPILER="$PWD/../WeightLang/WeightLang/target/debug/wgt" \
  uv run --locked python -m experiments.active.elf_generation.certify --round r007
```

Its scope is the train-inferred contract for two fixed ELF layouts, with an explicit
numerical margin. It does not prove general binary semantics.
