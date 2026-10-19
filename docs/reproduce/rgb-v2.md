# Verify, run and reproduce

Use the pinned environment in README. Extract to a new directory. Never edit
original source archives. Run all commands from `LightNest_RGB_v2/`.

## File integrity and code tests

```bash
python verify_release.py
python -m unittest discover -s tests -v
```

22 v2 tests cover budget sharing, wrong STOP, discarded suffixes, fresh fallback,
nonfinite blue outputs, compiled red checks, and unchanged tensor counts. The
52 inherited v1 tests were run in a separate original extraction; their log is
included. Neither test count is a capability score.

## Read results without training

`results/summary-v2.json` combines all methods and both seeds.
`results/final-*.json` stores committed actions and the raw scratch draft.
`results/risk-decision-replay.json` stores all risk probabilities and decisions.
`results/backend-*.json` records additional actual compiled inference.

The audits use a separate count/replay routine and the supplied second RV32
interpreter, not an external audit service. They only run bounded interpreted
code. Existing reports are write-once. To rerun an audit, use a working copy and
remove ONLY its output receipt there:

```bash
rm results/training-audit-v2.json
python -m experiment.audit train
rm results/evaluation-audit-v2.json
python -m experiment.audit final
```

The training audit replays each sampling stream and performs two restored next
steps per head. It does not overwrite trained checkpoints. The evaluation audit
rebuilds committed candidate bytes, runs the second interpreter and checks the
metrics. It does not rerun the neural networks. The later risk-decision replay
DOES rerun all blue drafts and checks every learned gate decision.

## Clean deterministic replay

```bash
python -m experiment.new_run --out ../rgb-v2-replay
cd ../rgb-v2-replay
python -m experiment.run_study fit --seed 104729
python -m experiment.run_study fit --seed 130363
python -m experiment.run_study calibrate --seed 104729
python -m experiment.run_study calibrate --seed 130363
python -m experiment.run_study freeze
python -m experiment.run_study final --seed 104729
python -m experiment.run_study final --seed 130363
python -m experiment.audit train
python -m experiment.audit final
python -m experiment.risk_diagnostics
python -m experiment.benchmark
```

This reuses the old test data to reproduce a result. It is not a new blind test.
The optional old v1 metadata and module names are preserved as source components;
use the v2 commands above, not the old v1 study launchers.

Each risk head has 512 updates with batch32. RGB red/green/blue and the old router
remain frozen. All four trained heads, AdamW state, random state, exact sampling
indices and fitting features are included. There is no omitted feature cache
needed to reproduce risk training. The initial RGB checkpoints are also present.

The generator source is saved in `experiment/prepare.py`. Its recorded external
paths identify the expanded v2 metadata archive and the original RGB v1 source
used for collision checks. Supplied `data/*.jsonl` are the exact generated inputs
and targets, so network/audit reproduction above does not require those external
archives. To repeat collision checking in a different filesystem, use the prior
archives at the two paths documented in that source, or copy it into a separate
new experiment with new path arguments and a new source commitment.

## Timing and optional compiled loop

Timing uses live proposals, five repeats, rotated method order, batch1 and
batch16, simple and equally balanced public examples. Parser cache is warmed;
result cache is absent. Model load, VM grading and first compile are outside
the steady-state measurement. Drafts, the risk head, verifier and fallback ARE
inside. Risk routes may produce different candidates, so their speed is not a
pure faster computation of an identical function. Verified drafts and compiled
red have separate action-equality checks.

`experiment.backend_study validate --seed <seed>` reruns complete inference on
consumed cases. `experiment.backend_study benchmark` runs the optional backend
comparison. These refuse to replace existing output reports; use a working copy
with only the relevant `results/backend-*.json` removed. Do not remove or rewrite
the frozen protocols, source hashes or input data and still call it the same run.

`experiment/cli_audit_recipe.py` preserves the original local CLI smoke-check
recipe and paths; use the portable prediction commands for normal operation.
The gate and verified routes stay optional. Backend equivalence was evaluated
for red and risk_full; other explicit combinations are not claimed to have the
same full evaluation coverage.
