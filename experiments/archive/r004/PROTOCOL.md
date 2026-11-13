# Complete ELF generation — versioned protocol

## Questions

Each round contains 544 independently assembled and natively verified references:
192 training / 32 validation / 32 audit exit codes, and 192 training / 32 validation /
64 audit eight-byte stdout requests. Request values and output hashes do not overlap
between splits. The manifest records each request, split and reference hash.
Exit codes have only 256 possible values: across rounds they cannot provide an
indefinitely fresh blind population. Random stdout payloads use all 256 byte values.

Models receive only the request and learned position embeddings. They emit all
bytes and stop decisions simultaneously, without an ELF template, target prefix,
reference length, compiler, assembler or repair at inference. Training capacity is
a shared maximum file-size limit; the first predicted stop chooses actual length.
The two reference layouts are intentionally small. No unsupported generalization
claim follows from performance on these questions.

## Versions

| Version | Computation | Purpose |
| --- | --- | --- |
| mlp | shared position/request MLP | cheap independent-position baseline |
| gru | two-layer GRU | sequential computation |
| conv | two width-5 convolutions | local interactions |
| transformer | two encoder layers, four attention heads | global interactions |

The GRU consumes position and request representations, **not** previous target
bytes. The other models may mix future position features; no target data exists in
those features. All use the same parallel emission interface.

## Selection and integrity

Three initialization seeds, same sampled minibatches, AdamW, gradient clipping,
fixed step budget, fixed width, CPU/f32 and deterministic Torch algorithms.
This matches examples and updates, not parameter counts, FLOPs or elapsed time.
Validation is checked every 80 steps and at the final step. Select checkpoints,
then architecture by macro family exact-file accuracy; break ties by accuracy on
train-derived variable byte positions, then weighted loss. The architecture ranking
uses seed means; the retained instance is its best validation seed.

Freeze `selection.json` before any test scoring. The test is an audit and cannot
change the winner. Report all seeds, families, full-file exactness, EOS correctness,
parameter counts, training time and generation time. Exact identity transfers the
independently checked reference behavior; arbitrary mismatching bytes are not
executed. The split is a controlled within-layout generalization test.

## Lifecycle

1. State hypothesis and budget; freeze source and protocol before training.
2. Validate protocol mechanics, run all candidates and select on validation.
3. Audit the frozen candidates once; write the report and concrete lessons.
4. Save the selected model, snapshot compact source/results, delete only this
   round's disposable checkpoints. Never delete user data or old model weights.
5. Overwrite `STATUS.json` with the present state and next hypothesis.
6. Start a new named round with fresh stdout questions and a declared change.

A completed round does not automatically justify application promotion. Replacing
inspection, editing, state handling or checkpoint APIs requires corresponding tasks.

## Round r002 predeclared follow-up

Hypothesis: request bytes are diluted by hundreds of fixed bytes. Compare `mlp`,
`conv`, `mlp_copy`, `conv_copy` at a shared 1,200-step budget with train-derived
variable-byte weight 128, data seed 201, and the same three initialization seeds.
Suffix `_copy` adds learned position/mode queries over the nine request-byte slots
and a learned gate. It does not prescribe any byte location or attach ELF data.
The no-copy variants isolate the effect of this head under the same loss/budget.
The increased budget and weighting mean r001 versus r002 is not an architecture-only
comparison. Within r002 the copy ablation uses identical data and schedules.

## Round r003 predeclared follow-up

Compare width 16 (`_small`) and width 32 for MLP/conv with the learned copy
head, 1,200 updates, variable-byte weight 128, data seed 301 and three seeds.
All four candidates share the questions and minibatch schedules. Selection uses
validation macro exact, variable-byte accuracy, **parameter count**, then loss.
For checkpoints of the same model, parameters are constant. This declares the
efficiency preference before test access; r001/r002 used loss as the last tie-break.
Only stdout requests are a fresh random population; finite exit values recur.

## Round r004 predeclared follow-up

The width-16 model can copy request bytes but still miss fixed bytes. Add a
zero-initialized learned mode/position output bias, comparing width 8 and 16
against the width-32 MLP/conv copy controls. Each model emits the entire file;
the bias is a trained parameter table, not supplied ELF bytes. Use data seed 401,
1,200 updates, weight 128 and three seeds. Include all bias-table parameters in
the parameter count. This deliberately tests within-layout memorization versus
shared-network capacity; success still does not establish unseen-layout synthesis.
