"""Verify transferred historical bytes and recompute aggregate experimental metrics."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'docs/evidence/token_lab_20261112'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def equal(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'Aggregate mismatch: {actual} != {expected}')


def verify():
    manifest = load(EVIDENCE / 'manifest.json')
    for item in manifest['files']:
        path = ROOT / item['destination']
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Evidence path escapes repository')
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Historical file changed: {path}')
    total = 0
    for name in ['dynamic-chunks', 'multihorizon', 'multihorizon-depth2']:
        folder = EVIDENCE / name
        config, records, summary = [load(folder / filename) for filename in ['config.json', 'results.json', 'summary.json']]
        expected = {(arm, seed) for arm in config['arms'] for seed in config['seeds']}
        observed = {(row['arm'], row['seed']) for row in records}
        if observed != expected or len(records) != len(expected):
            raise ValueError(f'Missing/duplicate runs: {name}')
        if any(row['steps'] != config['steps'] for row in records):
            raise ValueError(f'Unequal step budgets: {name}')
        for arm in config['arms']:
            runs = [row for row in records if row['arm'] == arm]
            if name == 'dynamic-chunks':
                metrics = [(split, metric, item) for split, metrics in summary['models'][arm].items()
                           for metric, item in metrics.items()]
            else:
                metrics = [(key.split('.')[0], key.split('.')[1], item) for key, item in summary[arm].items()]
            for split, metric, item in metrics:
                values = [row['metrics'][split][metric] for row in runs]
                equal(item['mean'], sum(values) / len(values))
                equal(item['min'], min(values))
                equal(item['max'], max(values))
                for row in runs:
                    equal(item['per_seed'][str(row['seed'])], row['metrics'][split][metric])
        total += len(records)
    validation = load(EVIDENCE / 'dynamic-chunks/validation.json')
    records = load(EVIDENCE / 'dynamic-chunks/results.json')
    if len(validation) != len(records):
        raise ValueError('Incomplete historical execution verification')
    for check in validation:
        row = next(row for row in records if (row['arm'], row['seed']) == (check['arm'], check['seed']))
        if not check['accuracy_reproduced'] or check['actual_slow_rows'] != row['metrics']['id']['slow_cell_rows_executed']:
            raise ValueError('Historical slow-cell execution count mismatch')
    print(f'Verified {len(manifest["files"])} SHA-256 files and aggregates for {total} historical runs.')


if __name__ == '__main__':
    verify()
