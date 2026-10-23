# LightNest Lean contracts

Pinned Lean 4.35.0; only the bundled Std library is required.

Run from the repository root:

```bash
python tools/check_lean.py
```

This builds the proofs in a fresh directory, audits transitive axioms, rechecks
compiled proofs with `leanchecker`, checks live Python/Lean agreement, and
requires known-invalid proof controls to fail. Missing Lean is an error.

For interactive development only:

```bash
cd formal
lake build
lake env lean Audit.lean
lake env leanchecker LightNest
```

These individual commands do not run the Python conformance tests. See
[the verification guide](../docs/FORMAL_VERIFICATION.md) for exact assumptions,
trust boundaries, and the candidate-retention tool. No theorem in this package
asserts that a learned model is accurate or that its arbitrary patches are safe.
