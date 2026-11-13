# Architecture experiment results

Validation selects the architecture and checkpoint before audit access. Budgets and question sets vary across rounds; compare architectures within a round.

| Round | Updates | Runs | Validation-selected architecture | Parameters | Validation macro exact | Audit exact files, all seeds |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| [r001](archive/r001/REPORT.md) | 240 | 12 | mlp | 31017 | 0.00% | 0/288 |
| [r002](archive/r002/REPORT.md) | 1200 | 12 | conv_copy | 38538 | 100.00% | 287/288 |
| [r003](archive/r003/REPORT.md) | 1200 | 12 | conv_copy | 38538 | 98.44% | 288/288 |
| [r004](archive/r004/REPORT.md) | 1200 | 12 | conv_copy | 38538 | 98.96% | 288/288 |
| [r005](archive/r005/REPORT.md) | 2400 | 12 | conv_copy_small | 16458 | 100.00% | 282/288 |
| [r006](archive/r006/REPORT.md) | 2400 | 9 | conv_copy_small | 16458 | 100.00% | 474/480 |
| [r007](archive/r007/REPORT.md) | 2400 | 9 | mlp_copy_mode_small | 13786 | 100.00% | 480/480 |

Completed training runs: **78**.

Each audit case requires the entire emitted file and first stop position to match an independently assembled and natively checked reference. Both families are reported separately in the archives.

This covers exit-status and eight-byte stdout programs in two fixed ELF layouts. It does not establish arbitrary program synthesis, binary understanding, editing, or new-layout generalization.

[Current state](STATUS.json) · [Lessons](LESSONS.md) · [Next declared work](PLAN.json)
