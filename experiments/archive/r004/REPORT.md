# r004 — complete ELF generation

Learn mode/position output biases from training to relieve fixed-byte memorization; compare width-8/16 biased MLP with width-32 copy-head controls. No headers are attached at inference.

Budget: 1200 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 64, 'test': 96}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp_copy | 32426 | 97.917% | 99.219% | 285/288 | 13.67 |
| conv_copy | 38538 | 98.958% | 100.000% | 288/288 | 16.83 |
| mlp_copy_bias_small | 28492 | 94.792% | 97.917% | 280/288 | 8.86 |
| mlp_copy_bias_tiny | 20708 | 39.062% | 36.458% | 70/288 | 7.52 |

Validation-selected architecture: **conv_copy**; retained instance: `conv_copy-s23`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| mlp_copy-s11 | 32/32 | 62/64 | 94/96 |
| conv_copy-s11 | 32/32 | 64/64 | 96/96 |
| mlp_copy_bias_small-s11 | 32/32 | 62/64 | 94/96 |
| mlp_copy_bias_tiny-s11 | 32/32 | 0/64 | 32/96 |
| mlp_copy-s23 | 32/32 | 64/64 | 96/96 |
| conv_copy-s23 | 32/32 | 64/64 | 96/96 |
| mlp_copy_bias_small-s23 | 32/32 | 62/64 | 94/96 |
| mlp_copy_bias_tiny-s23 | 32/32 | 0/64 | 32/96 |
| mlp_copy-s37 | 32/32 | 63/64 | 95/96 |
| conv_copy-s37 | 32/32 | 64/64 | 96/96 |
| mlp_copy_bias_small-s37 | 32/32 | 60/64 | 92/96 |
| mlp_copy_bias_tiny-s37 | 6/32 | 0/64 | 6/96 |

## Lesson

Learned mode/position output biases must count toward model size. Width-8 remained weak; width-16 bias MLP improved the smaller-model option but did not match the larger convolution at this budget. Plausible mechanisms need measured acceptance.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r004/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Paired doubled-update-budget test: can narrower models recover full-file quality? All four controls receive 2400 updates; fresh stdout questions and validation-first efficiency ranking.
