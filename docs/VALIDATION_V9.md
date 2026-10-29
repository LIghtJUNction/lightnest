# LightNest v9: finite risk planning and verified-state-model research

This **optional** research extension is based on the completed 2026-10-29 v9
validation. It does **not** change the production model, the v7 research
hypothesis class, or existing experiment checkpoints.

## Implemented code in Git

- `lightnest_rgb/risk_planner.py`: exact one-/two-query candidate-identification
  value, greedy entropy, two-step risk, and horizon-matched two-step entropy.
  **The candidate catalogue is supplied; this is not general program synthesis.**
- `lightnest_rgb/automata.py`: finite-state learner with membership and external
  equivalence queries, product-automaton equivalence checker, independent
  relation-closure validator, and a fixed x86-64 machine-code table interpreter.
- `tools/validation_v9_demo.py`: repeat the UTF-8 state-machine experiment
  in a new folder using CPython as a membership oracle and a hand-encoded
  RFC 3629 reference table as an exact counterexample teacher. Native execution
  uses the inherited bounded runner, not arbitrary direct execution.
- `tests/test_validation_v9.py`: 16 **data-free** regression tests for core
  mathematics, state-machine learning, bounded native execution, and the
  documented command-line entry point.

The complete original experiment orchestration, 19 historical v9 tests,
three frozen neural models, musl source labels, full native traces, and
all evaluation outputs remain in the separately archived source/evidence bundle.
The Git-only demo does not reproduce the neural training arm or all 69,924
historical native tests; run the archived `reproduce.py` for that.

## Important measurements (archived v9, not new CI measurements)

The musl-derived supplied-catalogue identification task has eight candidates
and up to two external answers. Mean success under the **declared target prior**:

| Selection | Weighted success |
|---|---:|
| Single-step entropy | 87.0838% |
| Exact two-step entropy | 88.6501% |
| Exact two-step whole-program risk | **88.7231%** |
| Three neural value approximators (average) | 86.2416% |

Most of the apparent advantage over single-step entropy came from searching
an additional step, **not** changing the utility. The neural approximation
did not generalize well to these held-out musl-derived sets. The latter are
not independent projects, and the target prior was synthetic.

In a separate experiment, CPython strict UTF-8 membership outputs plus an
**external exact reference DFA** led to a nine-state learned automaton. Its
state relation was exhaustively checked (9 reachable pairs, 2,304 byte
transitions). A fixed x86-64 backend passed 69,924 actual native tests on
strings up to 131,072 bytes. This proves equivalence of the **two provided
transition tables**, not the correctness of arbitrary binary files or all
CPython behavior. It is not a neural program generator.

The v8 prefix-sum posterior optimization was also tested: 96 workloads,
864 paired timings; probability discrepancy at most 2.22e-16, without clear
end-to-end speedup on the measured workloads. No timing results or learned
weights were fabricated or replaced in this Git extraction.

## Run the committed smoke tests and independently learn the DFA

Requires Linux x86-64, GCC and NumPy in the repository environment:

```bash
uv run --locked python -m unittest discover -s tests -p 'test_validation_v9.py' -v
uv run --locked python tools/validation_v9_demo.py --out /tmp/lightnest-v9-demo-fresh
```

The demo output destination must not already exist. It writes a learned DFA,
a trusted-backend function, and an actual native test summary. The demo's
case count is **not** the historical 69,924; compare the recorded result
rather than combining these evaluations.

## Reproduce all original v9 experiments

The complete authored source and frozen data are stored in Google Drive:

- [v9 project folder](https://drive.google.com/drive/folders/1dYJCho2EIb6x1ru_zhjD8d2B2LlQXavW)
- [Source and model archive](https://drive.google.com/file/d/12VqUchM-aQN56SnPDcG0UcmHKrNAht51/view)
- [Full v9 report](https://drive.google.com/file/d/1eLoO3xOW5BX8_hQsOC5LvmhTNLaAUvuE/view)

Archive SHA-256: `8764a586f9275f14b18ee064c5a4b0e6a989c1aedf41812b52309500e9bc40c3`.
The archive contains `reproduce.py`, `inputs/`, the full `src/` scripts and
`tests/test_validation.py`. Run that copy with a new absolute `--out` path,
under the recorded environment. The artifact is separate from the main Git
tree to keep large datasets, weights, executable binaries, and raw traces out
of normal source control.

## Critical boundaries

- The exact equivalence teacher *knows* the reference machine; replacing it
  with a finite test suite removes the equivalence guarantee.
- Relation-closure verification is a mathematical result about supplied
  finite transition tables; the trusted ELF backend remains empirically tested.
- Passing native tests does not certify security or memory safety outside the
  bounded input contract.
- Neither self-hosting nor recursive self-improvement (RSI) is implemented.
- These additions do not fix previously failing full-repository CI jobs by
  ignoring them; PR and workflow results must be evaluated independently.
