"""Rebuild a compact index from completed immutable result records."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    lines=['# Architecture experiment results','',
           'Validation selects the architecture and checkpoint before audit access. Budgets and question sets vary across rounds; compare architectures within a round.', '',
           '| Round | Updates | Runs | Validation-selected architecture | Parameters | Validation macro exact | Audit exact files, all seeds |',
           '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    total=0
    for folder in sorted((ROOT/'experiments/archive').iterdir()):
        if not (folder/'results.json').exists(): continue
        results=json.loads((folder/'results.json').read_text())
        protocol=json.loads((folder/'protocol.json').read_text())
        winner=results['winner_by_validation'];runs=results['runs'];total+=len(runs)
        selected=[r for r in runs if r['variant']==winner]
        exact=sum(r['test']['exact'] for r in selected);count=sum(r['test']['count'] for r in selected)
        validation=statistics.mean(r['validation']['macro_exact'] for r in selected)
        lines.append(f"| [{folder.name}](archive/{folder.name}/REPORT.md) | {protocol['args']['steps']} | {len(runs)} | {winner} | {selected[0]['parameters']} | {validation:.2%} | {exact}/{count} |")
    lines+=['',f'Completed training runs: **{total}**.', '',
            'Each audit case requires the entire emitted file and first stop position to match an independently assembled and natively checked reference. Both families are reported separately in the archives.', '',
            'This covers exit-status and eight-byte stdout programs in two fixed ELF layouts. It does not establish arbitrary program synthesis, binary understanding, editing, or new-layout generalization.', '',
            '[Current state](STATUS.json) · [Lessons](LESSONS.md) · [Next declared work](PLAN.json)', '']
    (ROOT/'experiments/RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
