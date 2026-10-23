# RGB over real ELF executables

The next mainline uses complete Linux x86-64 executables, not extracted function
bodies. Four separate gates cover structure/understanding, efficient editing,
learned complete-image generation and multi-step execution. Original system
executables are read-only; all generated artifacts remain outside the checkout.

## MiTA research anchor

Reference: Wen et al., [Mixture-of-Top-k Attention: Efficient Attention via
Scalable Fast Weights, v6](https://arxiv.org/html/2602.01219v6), especially §3.2.
The official [implementation](https://github.com/QishuaiWen/MiTA) was inspected at
`95460f703ef5bd5d814c45a6ff52d9582207d01c`.

Attention's keys/values can be interpreted as input-conditioned MLP weights.
MiTA combines compressed global evidence and routed access to original details.
Its shared and routed key/value pairs participate in one normalized attention;
averaging two independently normalized answers changes the operator.

Our adaptation is a CPU-compatible PyTorch implementation, not a claimed
reproduction of the paper's vision training or GPU throughput. ELF addresses,
duplicate occurrence maps and incremental edit invalidation are additional
project-specific requirements. Paper fast weights are not synonymous with
blue's fast decisions, and they do not supply red's recurrent state by themselves.

## RGB responsibilities

- Green: encode repeated byte/log content once, retain every location and
  occurrence, build a compact global expert and reusable top-k detail experts.
  Compression never authorizes erasing a byte range or treating distinct
  addresses as semantically identical.
- Blue: shallow next-action proposal and calibrated routing; it must actually
  skip deeper computation when accepted. Quality and wrong STOP are measured.
- Red: shared recurrent state with a configurable depth schedule. It queries the
  same current-image evidence bank repeatedly; it does not rebuild the entire
  context for each thought step. A committed edit invalidates affected state.
- Controller: one global action/candidate/time budget, explicit STOP, replayable
  before/after hashes, actual executable feedback, no hidden-test feedback.

## Artifact and execution boundary

`artifact.py` parses complete ELF headers, segments, sections, dynamic imports
and entry points. Fixed-length patches have image and before-byte preconditions;
edits create new files atomically, using copy-on-write when available.

`execute.py` runs complete executables under bubblewrap namespaces, an isolated
filesystem, no network, a syscall filter denying process creation, and resource
limits. Inputs and expected outputs are separate. Exact result caches bind the
image hash, input, arguments and execution policy.

Generation must output the complete image from trained model outputs. The writer
does not append an ELF header, copy a template or silently repair invalid output.
Linker-built examples are training targets, not the generation implementation.

## Evaluation protocol

Use installed open-source CLI executables and separate program-level splits.
Record package/artifact identity, real clean baselines, corruption difficulty,
actual failing cases, hidden cases and multiple-edit states. Mere file-count
growth does not demonstrate increased independent task diversity.

Compare dense observation, compression-only and compression-plus-routing under
the same tasks; include expert construction, gathering and invalidation in
timing. Adaptive versus fixed-depth red must retain old successes and expose
wrong STOP. Complete-image generation additionally requires ELF validation and
successful OS execution of a request not used for fitting.

## Invalid entry preflight

Fault injection initially sent malformed entry addresses to the OS, causing
expected sandbox-child SIGSEGV reports in the host journal. The 2026-10-24
`cut` case with entry `0x40002cd0` is now rejected before child creation. The
runner validates x86-64 executable type and a file-backed executable entry,
records `invalid_artifact` separately, and does not count rejections as native
executions. This does not prove arbitrary instructions safe or prevent all
runtime faults. Existing negative-example execution records remain unchanged.
