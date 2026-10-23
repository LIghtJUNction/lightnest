# Reproduce LightNest Hybrid v1

The recorded run used Python 3.13.5 and PyTorch 2.10.0+cpu. See environment.json
for the actual environment. A different library, hardware, or batch shape may
change floating point results. Do not claim exact training replay without a check.

## Verify and test

Extract the archive. In its LightNest_Hybrid_v1 directory, run:

```bash
python verify_release.py
python -m hybrid.test_mechanisms
```

The runtime needs PyTorch and the Python standard library. It makes no model API
calls and needs no API key. The 12 mechanism tests use unchanged baseline weights.

## Use a candidate

The following known public example is included in the package:

```bash
python -m hybrid.predict \
  --binary results/cli-source.rv32 \
  --command "change every add that contributes to the output to sub" \
  --out ../Hybrid_parent_candidate
```

This uses the unchanged parent. To request an experimental policy explicitly:

```bash
python -m hybrid.predict --seed 104729 --strategy rl0 \
  --binary results/cli-source.rv32 \
  --command "change every add that contributes to the output to sub" \
  --out ../Hybrid_research_candidate
```

An output directory must not exist. The command writes candidate.rv32 and
trace.json with mode 0600. It never runs or deploys the candidate. It accepts
aligned raw RV32, at most 1,024 bytes, not arbitrary ELF or executable formats.
The trace shows discrete mode decisions and action text, not translated thoughts.

## Reproduce training in a new directory

Do not overwrite this experiment. Copy only the unchanged inputs and source:

```bash
python - <<'PY'
from pathlib import Path
import shutil
source=Path.cwd();dest=source.parent/'LightNest_Hybrid_retrain'
dest.mkdir(exist_ok=False)
for name in ('hybrid','baseline','data'):
    shutil.copytree(source/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('protocol.json','source-freeze.json'):
    shutil.copy2(source/name,dest/name)
(dest/'models').mkdir();(dest/'results').mkdir()
print(dest)
PY
cd ../LightNest_Hybrid_retrain
python -m hybrid.train --seed 104729
python -m hybrid.train --seed 130363
python -m hybrid.evaluate freeze
python -m hybrid.evaluate evaluate --seed 104729
python -m hybrid.evaluate evaluate --seed 130363
python -m hybrid.audit
python -m hybrid.public_checks
python -m hybrid.final_checks
python -m hybrid.report
```

This reuses the recorded training rows and consumed tests. It is a reproduction,
not a fresh holdout experiment. Output files use exclusive creation. Run each
phase once per new directory. Full score and RL logs remain available in results.

`hybrid.prepare` was used to generate the included data. Rebuilding the historical
novelty check also needs the original RGB v1 archive and scale-v2 dataset at the
paths named in that script. They are not needed to train with the included rows.
Do not rerun prepare on this release: it refuses existing dataset files.

## Interpretation

`rl0` has no compute penalty. `rlcost` has a penalty of 0.1 times normalized
internal rounds. Both train mode decisions only. The actor and latent transition
stay frozen during RL; saved rollout features are replayed at those decisions.
This is not GRPO over a language model and not a reproduction of HybridAgent.

The link and action head were supervised on 192 batches per seed. The controller
then had 256 supervised updates and 96 policy-gradient updates per RL branch.
The independent audit reconstructs actions and scores with a second existing VM.
It is a separate script from the same session, not external certification.

The original project and default model are not replaced. Zero wrong edits and
no loss of old successful cases were required for promotion; no new policy met
those requirements. New research must reserve different test groups.
