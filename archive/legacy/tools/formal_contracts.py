"""Small executable contracts mirrored in formal/LightNest.

This module checks finite evidence. It neither scores a neural model nor
replaces a checkpoint. The caller must establish test and evaluator provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def schedule_cost(depths: list[int] | tuple[int, ...], maximum: int = 64) -> int:
    """Validate absolute checkpoints before subtracting consecutive depths."""
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('maximum must be a positive integer')
    if not depths:
        raise ValueError('a nonempty schedule is required')
    previous = cost = 0
    for depth in depths:
        if type(depth) is not int or not previous < depth <= maximum:
            raise ValueError('depths must strictly increase without exceeding maximum')
        cost += depth - previous
        previous = depth
    if previous != maximum:
        raise ValueError('the final forced checkpoint is missing')
    return cost


def aligned_comparisons(parent: list[dict], candidate: list[dict]) -> list[tuple[bool, bool, bool]]:
    """Align by unique ID AND original-input digest, never by list order."""
    def index(rows: list[dict]) -> dict[str, dict]:
        if not isinstance(rows, list) or not rows:
            raise ValueError('nonempty record lists are required')
        indexed = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('each record must be an object')
            key, digest = row.get('id'), row.get('initial_sha256')
            if not isinstance(key, str) or not key or key in indexed:
                raise ValueError('record IDs must be nonempty and unique')
            if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
                raise ValueError('initial_sha256 must be a lowercase SHA-256 digest')
            if any(type(row.get(k)) is not bool for k in ('strict', 'false_edit')):
                raise ValueError('strict and false_edit must be booleans, not scores')
            if row['strict'] and row['false_edit']:
                raise ValueError('a false edit cannot also be a strict success')
            indexed[key] = row
        return indexed

    a, b = index(parent), index(candidate)
    if a.keys() != b.keys():
        raise ValueError('parent and candidate must cover exactly the same task IDs')
    out = []
    for key in sorted(a):
        if a[key]['initial_sha256'] != b[key]['initial_sha256']:
            raise ValueError('same task ID has a different input digest')
        out.append((a[key]['strict'], b[key]['strict'], b[key]['false_edit']))
    return out


def retention_gate(parent: list[dict], candidate: list[dict]) -> dict[str, Any]:
    rows = aligned_comparisons(parent, candidate)
    lost = sum(was and not now for was, now, _ in rows)
    false_edits = sum(bad for _, _, bad in rows)
    gained = sum(not was and now for was, now, _ in rows)
    return {
        'retention_passed': lost == 0 and false_edits == 0,
        'records': len(rows), 'lost_successes': lost,
        'candidate_false_edits': false_edits, 'added_successes': gained,
        'model_promoted': False,
        'scope': 'These supplied records only; no generalization or speed guarantee.',
    }


def read_evidence(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('evidence file exceeds 64 MiB')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or not isinstance(obj.get('records'), list):
        raise ValueError('expected an evidence object containing records')
    for key in ('dataset_sha256', 'evaluator_sha256', 'protocol_sha256'):
        if not isinstance(obj.get(key), str) or not DIGEST.fullmatch(obj[key]):
            raise ValueError(f'missing or invalid {key}; bind evidence before comparison')
    return obj


def compare_evidence(parent: dict, candidate: dict) -> dict:
    for key in ('dataset_sha256', 'evaluator_sha256', 'protocol_sha256'):
        if parent.get(key) != candidate.get(key) or not DIGEST.fullmatch(str(parent.get(key, ''))):
            raise ValueError(f'cannot compare different or missing {key}')
    return retention_gate(parent['records'], candidate['records'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    args = parser.parse_args()
    a, b = read_evidence(args.parent), read_evidence(args.candidate)
    report = compare_evidence(a, b)
    report['input_files_sha256'] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [('parent', args.parent), ('candidate', args.candidate)]
    }
    print(json.dumps(report, indent=2))
    if not report['retention_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
