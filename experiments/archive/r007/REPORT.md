# r007 — complete ELF generation

Separate request payload from fixed-byte computation: mode/position-only backbone plus learned request-copy branch. Compare with original narrow conv on identical data and budget; test conditional-affine invariants.

Budget: 2400 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 160, 'test': 160}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv_copy_small | 16458 | 99.479% | 99.349% | 475/480 | 22.10 |
| conv_copy_mode_small | 15306 | 100.000% | 100.000% | 480/480 | 20.57 |
| mlp_copy_mode_small | 13786 | 100.000% | 100.000% | 480/480 | 16.06 |

Validation-selected architecture: **mlp_copy_mode_small**; retained instance: `mlp_copy_mode_small-s37`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| conv_copy_small-s11 | 32/32 | 125/128 | 157/160 |
| conv_copy_mode_small-s11 | 32/32 | 128/128 | 160/160 |
| mlp_copy_mode_small-s11 | 32/32 | 128/128 | 160/160 |
| conv_copy_small-s23 | 32/32 | 126/128 | 158/160 |
| conv_copy_mode_small-s23 | 32/32 | 128/128 | 160/160 |
| mlp_copy_mode_small-s23 | 32/32 | 128/128 | 160/160 |
| conv_copy_small-s37 | 32/32 | 128/128 | 160/160 |
| conv_copy_mode_small-s37 | 32/32 | 128/128 | 160/160 |
| mlp_copy_mode_small-s37 | 32/32 | 128/128 | 160/160 |

## Lesson

The mode-only intervention separates fixed content and stopping from variable request data. Compare it against the original narrow convolution on identical examples and update budgets. The interval check, when applicable, audits all bit combinations under the train-inferred two-layout contract; it does not prove arbitrary executable correctness.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r007/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Expand the question contract to unseen ELF layouts and independent inspect/edit tasks before application promotion. Freeze the new protocol before training.

## Request-bit margin audit

Applicable: True; passed: True. See `copy-margin-certificate.json` for bounds, tolerance and scope.
