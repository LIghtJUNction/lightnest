# r001 — complete ELF generation

Compare parallel complete-file decoders without target bytes as inputs.

Budget: 240 AdamW updates per run; seeds [11, 23, 37]; CPU/f32. Split sizes: {'train': 384, 'validation': 64, 'test': 96}. Matching updates and examples does not match FLOPs or parameter counts.

| Architecture | Parameters | Validation macro exact | Audit macro exact | Audit exact files | Train seconds (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp | 31017 | 0.000% | 0.000% | 0/288 | 1.73 |
| gru | 39497 | 0.000% | 0.000% | 0/288 | 54.15 |
| conv | 37129 | 0.000% | 0.000% | 0/288 | 2.95 |
| transformer | 43913 | 0.000% | 0.000% | 0/288 | 30.53 |

Validation-selected architecture: **mlp**; retained instance: `mlp-s11`.
Selection is frozen in `selection.json` before audit scoring. Macro exact gives the two task families equal weight; raw exact counts use every case. CPU times are indicative observations on this host, not isolated throughput benchmarks.

## Per-seed audit

| Run | Exit exact | Stdout exact | Entire-file exact |
| --- | ---: | ---: | ---: |
| mlp-s11 | 0/32 | 0/64 | 0/96 |
| gru-s11 | 0/32 | 0/64 | 0/96 |
| conv-s11 | 0/32 | 0/64 | 0/96 |
| transformer-s11 | 0/32 | 0/64 | 0/96 |
| mlp-s23 | 0/32 | 0/64 | 0/96 |
| gru-s23 | 0/32 | 0/64 | 0/96 |
| conv-s23 | 0/32 | 0/64 | 0/96 |
| transformer-s23 | 0/32 | 0/64 | 0/96 |
| mlp-s37 | 0/32 | 0/64 | 0/96 |
| gru-s37 | 0/32 | 0/64 | 0/96 |
| conv-s37 | 0/32 | 0/64 | 0/96 |
| transformer-s37 | 0/32 | 0/64 | 0/96 |

## Lesson

All primary scores are zero across four families and three seeds. Falling loss hid request-byte failure. The tie-break winner is diagnostic, not an accepted architecture. More expensive GRU/Transformer computation gave no success at this budget.

## Retained evidence

Source snapshots, source/toolchain hashes, protocol, request manifest, validation curves, selection and per-case audit remain here. Complete selected model: `weights/experiments/r001/model/`. Losing training checkpoints were removed.

The questions cover two fixed assembler layouts. Correctness is transferred by exact identity to independently assembled and natively checked references. Mismatching outputs are not executed. Exit values come from a finite 256-value population and recur across rounds.

## Next experiment

Paired copy-head ablation after increasing request-byte weight and budget; shared examples and schedules within round.
