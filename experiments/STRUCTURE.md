# Repository layout

```text
lightnest/
  architecture/          application configuration, adapters and runtime kernels
    wgt/*/model.wgt      application model entrypoints
  training/              application training and data preparation
  runtime/               ELF inspection and isolated execution
  rl/                    feedback and learning loops
  self_hosting/          application integration
experiments/
  STATUS.json            one current state, replaced rather than appended
  PLAN.json              declared hypotheses and budgets for the next rounds
  loop.py                runs declared rounds and skips completed archives
  active/elf_generation/ current runner, question builder and candidate versions
  archive/rNNN/          compact frozen source, questions, results and lessons
weights/experiments/      retained complete selected models, ignored by Git
.cache/experiments/       disposable round checkpoints, removed after archival
```

Commands from the repository root:

```sh
uv run --locked python -m experiments.loop --limit 1
uv run --locked python -m unittest tests.experiments.test_protocol
```

`loop` never repeats or overwrites a completed round. Add a new ID and hypothesis
in `PLAN.json` to continue. An incomplete archive blocks automatic continuation
for that ID so that failure evidence is not silently overwritten.

Active code is reusable; completed versions live only in compact archive snapshots.
No experiment data, binaries or optimizer checkpoints belong in Git. Reports include
failed outcomes and the specific lesson that motivates the next hypothesis.

A research candidate becomes an application architecture only after passing the
application's inference, editing, state, serialization and execution contracts.
