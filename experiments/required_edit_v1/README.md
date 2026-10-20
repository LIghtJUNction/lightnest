# Required-edit feedback experiment

This experiment changes only how training feedback is shared between remaining
required edit positions. It does not add a network, alter inference, or enable
any unaccepted model. See `docs/experiments/2026-10-20-required-edit-mechanism.md`
for all results and limitations.

`loss.py` contains three rules: `set_mass` (the unchanged baseline), `coverage`
(equal supervision for all required edits), and `weakest` (bottleneck control).
They share identical byte supervision. STOP and single-edit examples have
identical objectives. The labels must mean that **every edit is required**, not
that edits are mutually exclusive candidate solutions.

## Source tests

From the repository root, with the existing CPU PyTorch dependencies:

```bash
python -m unittest discover -s tests -p 'test_required_edit.py' -v
```

## Reproduce a fresh run

Use local copies of the already-delivered, verified archives. Extract the
Hybrid v1 archive into a directory outside the repository. From the scale v2
source/models archive, copy each `runs/expanded-<seed>/model.pt` to a separate
parents directory as `expanded-104729.pt` and `expanded-130363.pt`. Do not
substitute the older accepted C64 or randomly initialized weights.

```bash
python experiments/required_edit_v1/run.py prepare \
  --workspace ../LightNest_Hybrid_v1 \
  --parents ../expanded-parents \
  --dataset-archive ../LightNest_scale_v2_10k_dataset.zip \
  --rgb1-archive ../LightNest_RGB_v1_source_models_results.zip \
  --out ../required-edit-reproduction

python experiments/required_edit_v1/run.py train \
  --workspace ../LightNest_Hybrid_v1 --out ../required-edit-reproduction
python experiments/required_edit_v1/run.py evaluate \
  --workspace ../LightNest_Hybrid_v1 --out ../required-edit-reproduction
python experiments/required_edit_v1/run.py audit \
  --workspace ../LightNest_Hybrid_v1 --out ../required-edit-reproduction
```

Run ordinary Python, without `-O`/`-OO`: the research audits use assertions.
The first command records hashes and freezes the new run before training.
The source freeze records absolute paths in that run. For a different machine
or path, create a new run with `prepare`; do not edit a frozen receipt to make
an old run appear unchanged. All stages refuse to replace existing outputs.
Reproduction of the same published test is not a new blind evaluation.

The archived experiment used Python 3.13.5 and PyTorch 2.10.0+cpu. Different
hardware or library versions need not produce bitwise-identical retraining.
The complete result archive separately preserves all fixed model endpoints,
input records, action traces, sampling logs, and verification receipts.
