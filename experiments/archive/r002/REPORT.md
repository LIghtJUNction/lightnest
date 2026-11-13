# r002 — complete ELF generation

Paired copy-head ablation after increasing request-byte weight and budget; shared examples and schedules within round.

Budget: 1200 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 64, 'test': 96}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp | 31017 | 27.604% | 23.958% | 46/288 | 11.47 |
| conv | 37129 | 40.104% | 35.938% | 69/288 | 13.74 |
| mlp_copy | 32426 | 100.000% | 99.479% | 286/288 | 15.06 |
| conv_copy | 38538 | 100.000% | 99.740% | 287/288 | 19.35 |

Validation-selected architecture: **conv_copy**; retained instance: `conv_copy-s23`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| mlp-s11 | 13/32 | 0/64 | 13/96 |
| conv-s11 | 25/32 | 0/64 | 25/96 |
| mlp_copy-s11 | 32/32 | 63/64 | 95/96 |
| conv_copy-s11 | 32/32 | 64/64 | 96/96 |
| mlp-s23 | 10/32 | 0/64 | 10/96 |
| conv-s23 | 16/32 | 0/64 | 16/96 |
| mlp_copy-s23 | 32/32 | 63/64 | 95/96 |
| conv_copy-s23 | 32/32 | 64/64 | 96/96 |
| mlp-s37 | 23/32 | 0/64 | 23/96 |
| conv-s37 | 28/32 | 0/64 | 28/96 |
| mlp_copy-s37 | 32/32 | 64/64 | 96/96 |
| conv_copy-s37 | 32/32 | 63/64 | 95/96 |

## Lesson

Paired no-copy controls establish a large gain from learned request-slot transfer. MLP/copy produced 286/288 exact audit files; conv/copy produced 287/288. Changing the backbone alone was less useful than preserving request information.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r002/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Paired width-16 versus width-32 comparison with learned request-slot copying; prefer fewer parameters only after validation exact and variable accuracy tie.
