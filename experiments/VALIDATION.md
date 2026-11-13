# Local validation

- Official `tools/check.py`: 343 tests, 3 skipped, no failures on the final run. This includes 9 experiment protocol checks.
- All 11 current WGT source files pass the same sibling debug compiler used for training.
- Latest wheel contains all 10 flat application WGT files and the current candidate WGT source; archived experiments are excluded.
- All 7 retained complete models load and execute in fresh processes with `WGT_COMPILER=/nonexistent`.
- r007 CLI generation for exit 7 and eight-byte stdout `LightNes` exactly matches independently assembled, natively checked references. Generated unknown outputs are never directly executed.
- r007 selected-model request-bit interval check passes both train-inferred layouts at numerical tolerance 0.001. Minimum bit margins: exit 3.958, stdout 1.977. This is not a general ELF semantics proof.
- An earlier overloaded run exceeded an old demo's 15-second startup timeout. A later retry exposed concurrent editable SDK compiler-lookup changes. The check/loop now explicitly select the unchanged sibling debug binary, respecting an existing WGT_COMPILER. The final complete run passes.
- Later-round editable SDK source was not fully snapshotted at process start. Compiled definitions and trained parameters are preserved in complete selected artifacts; compiler identity is recorded.

All work is local. Application entrypoints retain their existing model contracts; this controlled two-layout experiment has not replaced the application's inspection/editing architecture.
