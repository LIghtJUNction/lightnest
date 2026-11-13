# Architecture search lessons

1. **Mean loss is insufficient.** In [r001](archive/r001/REPORT.md), all four
   architecture families across three seeds generated zero fully correct files.
   Hundreds of predictable bytes hid failure on the requested payload.
2. **Request transfer mattered more than backbone choice.** The paired controls in
   [r002](archive/r002/REPORT.md) reached 286/288 exact files for MLP/copy and
   287/288 for conv/copy, well above their no-copy controls at identical budget.
3. **Shrinking width can damage full-file quality.** In
   [r003](archive/r003/REPORT.md), the width-16 MLP reached 70.83% mean validation
   macro exact versus 95.83% at width 32. Correct variable bytes do not guarantee
   the rest of the file or stopping is correct.
4. **Extra learned tables are not free.** [r004](archive/r004/REPORT.md) counts all
   mode/position bias parameters. The width-8 bias model remained far behind the
   width-32 convolution; a plausible mechanism still needs measured acceptance.
5. **Validation and audit can disagree.** Rank candidates on validation before
   examining test results. Preserve every seed and family. Report raw exact-file
   counts as well as a family-balanced macro mean.
6. **Repeated layouts limit the conclusion.** These tasks teach two assembler
   layouts, including complete headers and section data. They do not measure
   arbitrary program synthesis, understanding, editing, or unseen ELF layouts.
7. **Artifacts are evidence.** Each completed round keeps source, request manifest,
   source/toolchain hashes, protocol, validation selection and per-case audit.
   Only the selected complete model survives; temporary losing checkpoints go.
8. **State is not history.** `STATUS.json` contains only present state and pointers.
   Protocols, results and lessons belong in compact round archives.

The current winner and the next action are in [STATUS.json](STATUS.json). New
hypotheses and budgets must be declared in [PLAN.json](PLAN.json) before execution.

9. **More validation is not a structural fix.** [r005](archive/r005/REPORT.md)
   selected a 16,458-parameter model with 282/288 audit successes. Expanding
   validation in [r006](archive/r006/REPORT.md) still yielded 474/480 audit successes.
10. **Separate fixed and variable computation when the task contract permits it.**
    In [r007](archive/r007/REPORT.md), mode/position-only MLP plus learned copying
    achieved 480/480 audit files across three seeds, with 13,786 parameters. The
    selected instance passed request-bit interval bounds for both train-inferred
    layouts. This is stronger evidence for those layouts, not general synthesis.
11. **Pin the actual compiler used for a round.** The editable SDK changed its
    default compiler lookup during this session. Explicitly selecting the same
    sibling debug binary restored local verification; do not alter another
    checkout's ongoing work. SDK source provenance for later rounds is partial;
    retained complete models include their compiled definitions.
