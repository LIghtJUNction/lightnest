# r005 — complete ELF generation

Paired doubled-update-budget test: can narrower models recover full-file quality? All four controls receive 2400 updates; fresh stdout questions and validation-first efficiency ranking.

Budget: 2400 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 64, 'test': 96}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv_copy_small | 16458 | 100.000% | 98.438% | 282/288 | 28.76 |
| mlp_copy_bias_small | 28492 | 100.000% | 100.000% | 288/288 | 24.26 |
| mlp_copy | 32426 | 100.000% | 99.219% | 285/288 | 38.30 |
| conv_copy | 38538 | 100.000% | 100.000% | 288/288 | 46.28 |

Validation-selected architecture: **conv_copy_small**; retained instance: `conv_copy_small-s11`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| conv_copy_small-s11 | 32/32 | 64/64 | 96/96 |
| mlp_copy_bias_small-s11 | 32/32 | 64/64 | 96/96 |
| mlp_copy-s11 | 32/32 | 61/64 | 93/96 |
| conv_copy-s11 | 32/32 | 64/64 | 96/96 |
| conv_copy_small-s23 | 32/32 | 63/64 | 95/96 |
| mlp_copy_bias_small-s23 | 32/32 | 64/64 | 96/96 |
| mlp_copy-s23 | 32/32 | 64/64 | 96/96 |
| conv_copy-s23 | 32/32 | 64/64 | 96/96 |
| conv_copy_small-s37 | 32/32 | 59/64 | 91/96 |
| mlp_copy_bias_small-s37 | 32/32 | 64/64 | 96/96 |
| mlp_copy-s37 | 32/32 | 64/64 | 96/96 |
| conv_copy-s37 | 32/32 | 64/64 | 96/96 |

## Lesson

All candidates reached perfect mean validation macro exact. The predeclared parameter tie-break selected the narrow convolution, but its audit produced only 282/288 exact files. A perfect small validation set was insufficient to establish equivalent reliability. Do not retroactively change this selection using the audit.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r005/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Independent confirmation with 4x more stdout validation questions: distinguish smaller-model rare failures before efficiency tie-breaking. Three seeds; identical within-round training examples and 2400 updates.
