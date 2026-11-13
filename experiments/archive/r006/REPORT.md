# r006 — complete ELF generation

Independent confirmation with 4x more stdout validation questions: distinguish smaller-model rare failures before efficiency tie-breaking. Three seeds; identical within-round training examples and 2400 updates.

Budget: 2400 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 160, 'test': 160}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv_copy_small | 16458 | 100.000% | 99.219% | 474/480 | 20.83 |
| mlp_copy_bias_small | 28492 | 100.000% | 98.828% | 471/480 | 19.34 |
| conv_copy | 38538 | 100.000% | 99.870% | 479/480 | 42.63 |

Validation-selected architecture: **conv_copy_small**; retained instance: `conv_copy_small-s11`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| conv_copy_small-s11 | 32/32 | 128/128 | 160/160 |
| mlp_copy_bias_small-s11 | 32/32 | 124/128 | 156/160 |
| conv_copy-s11 | 32/32 | 127/128 | 159/160 |
| conv_copy_small-s23 | 32/32 | 124/128 | 156/160 |
| mlp_copy_bias_small-s23 | 32/32 | 128/128 | 160/160 |
| conv_copy-s23 | 32/32 | 128/128 | 160/160 |
| conv_copy_small-s37 | 32/32 | 126/128 | 158/160 |
| mlp_copy_bias_small-s37 | 32/32 | 123/128 | 155/160 |
| conv_copy-s37 | 32/32 | 128/128 | 160/160 |

## Lesson

Quadrupling stdout validation questions did not eliminate the selection gap: narrow convolution validation was perfect but audit was 474/480. Increasing a finite validation set alone did not resolve the rare errors. The next experiment removes payload input from fixed-byte and stop computation.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r006/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Separate request payload from fixed-byte computation: mode/position-only backbone plus learned request-copy branch. Compare with original narrow conv on identical data and budget; test conditional-affine invariants.
