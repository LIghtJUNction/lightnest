# r003 — complete ELF generation

Paired width-16 versus width-32 comparison with learned request-slot copying; prefer fewer parameters only after validation exact and variable accuracy tie.

Budget: 1200 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 64, 'test': 96}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp_copy_small | 14938 | 70.833% | 72.396% | 183/288 | 6.73 |
| mlp_copy | 32426 | 95.833% | 97.656% | 279/288 | 12.03 |
| conv_copy_small | 16458 | 95.833% | 97.396% | 278/288 | 8.33 |
| conv_copy | 38538 | 98.438% | 100.000% | 288/288 | 15.16 |

Validation-selected architecture: **conv_copy**; retained instance: `conv_copy-s11`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| mlp_copy_small-s11 | 31/32 | 31/64 | 62/96 |
| mlp_copy-s11 | 32/32 | 62/64 | 94/96 |
| conv_copy_small-s11 | 32/32 | 62/64 | 94/96 |
| conv_copy-s11 | 32/32 | 64/64 | 96/96 |
| mlp_copy_small-s23 | 32/32 | 56/64 | 88/96 |
| mlp_copy-s23 | 32/32 | 62/64 | 94/96 |
| conv_copy_small-s23 | 32/32 | 60/64 | 92/96 |
| conv_copy-s23 | 32/32 | 64/64 | 96/96 |
| mlp_copy_small-s37 | 32/32 | 1/64 | 33/96 |
| mlp_copy-s37 | 32/32 | 59/64 | 91/96 |
| conv_copy_small-s37 | 32/32 | 60/64 | 92/96 |
| conv_copy-s37 | 32/32 | 64/64 | 96/96 |

## Lesson

Shrinking width is not a free optimization at the same number of updates. Width-16 MLP quality fell substantially. A correct payload still does not imply that every fixed byte and the stopping decision is correct.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r003/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Learn mode/position output biases from training to relieve fixed-byte memorization; compare width-8/16 biased MLP with width-32 copy-head controls. No headers are attached at inference.
