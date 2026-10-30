"""Make genuine conflicting repair requests for the same damaged training ELF.

Only training programs are executed during construction. Each alternative is
a different one-bit repair of the same input, with independently checked
requested outputs. The original validation and test pools stay unchanged;
this follow-up permits public validation only, not another neural test query.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lightnest.artifact import digest
from lightnest.corpus import external
from archive.research.exact_x86 import ExactX86Runner
from lightnest.execute import ExecutableRunner, same_result
from tools.research_exact_repair import damaged, hashes, outputs, refreeze, schedule_identity, train, training_batches, verify, visible


def signature(results):
    return json.dumps([outputs(row) for row in results], sort_keys=True)


def prepare(out, parent, *, alternatives=2):
    if alternatives not in (1, 2, 3): raise ValueError('one to three alternatives per damaged input')
    parent, protocol, tasks = verify(parent); out = external(out)
    if any(json.loads(path.read_text()).get('test_queries', 0) for path in parent.parent.glob('repair*/review.json')):
        raise ValueError('neural tests in this pool were already queried; construct a new holdout first')
    sources = hashes(); own = 'tools/research_counterfactual_repair.py'
    construction_sources = {own: digest(Path(__file__).read_bytes())}
    out.mkdir(parents=True, exist_ok=False)
    targets = out / 'targets'; targets.mkdir()
    core = ExactX86Runner(max_steps=512, trace_limit=0)
    native = ExecutableRunner(out / 'construction-native', timeout=.5, cache=False)
    extra = []; counts = Counter(); paired = []
    for task in tasks:
        if task['split'] != 'train': continue
        image, _ = damaged(task); examples = visible(task); position = task['patch']['position']
        original = signature([row['expected'] for row in task['visible']])
        observed = signature(task['observed']); seen = {original, observed}
        original_bit = (task['patch']['before'] ^ task['patch']['after']).bit_length() - 1
        choices = sorted((bit for bit in range(8) if bit != original_bit),
                         key=lambda bit: digest(f'{task["id"]}/{bit}'.encode()))
        additions = []
        for bit in choices:
            if len(additions) >= alternatives: break
            target = bytearray(image); target[position] ^= 1 << bit; target = bytes(target)
            results = []
            for example in examples:
                result = core.run(target, example.case); counts['tensor_calls'] += 1
                if result['status'] != 'ok':
                    counts['unsupported_or_faulting_alternative'] += 1; break
                results.append(result)
            if len(results) != len(examples): continue
            fingerprint = signature(results)
            if fingerprint in seen:
                counts['no_new_requested_behavior'] += 1; continue
            independent = [native.run(target, example.case) for example in examples]
            if any(not same_result(a, b) for a, b in zip(results, independent, strict=True)):
                (out / 'construction-counterexample.json').write_text(json.dumps(
                    {'task': task['id'], 'bit': bit, 'tensor': [outputs(r) for r in results],
                     'native': [outputs(r) for r in independent]}, indent=2))
                raise RuntimeError('counterfactual target disagrees with native execution')
            identity = digest(target); path = targets / (identity + '.elf')
            if path.exists():
                if path.read_bytes() != target: raise ValueError('target digest collision')
            else: path.write_bytes(target)
            desired = [{'input': row['input'], 'expected': outputs(result)}
                       for row, result in zip(task['visible'], results, strict=True)]
            additions.append(task | {'id': task['id'] + f'/desired-flip-{bit}',
                                    'image_path': str(path), 'image_sha256': identity,
                                    'patch': {'position': position, 'bit': bit,
                                              'before': target[position], 'after': image[position]},
                                    'visible': desired, 'hidden': [],
                                    'counterfactual_parent': task['id']})
            seen.add(fingerprint)
        extra.extend(additions)
        paired.append({'id': task['id'], 'damaged_sha256': task['damaged_sha256'],
                       'alternate_requests': [row['id'] for row in additions]})
        print(json.dumps({'task': task['id'], 'alternatives': len(additions)}), flush=True)
    if not extra: raise ValueError('no independently checked counterfactual requests')
    if hashes() != sources or digest(Path(__file__).read_bytes()) != construction_sources[own]:
        raise ValueError('source changed during construction')
    all_tasks = tasks + extra
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in all_tasks))
    (out / 'counterexamples.json').write_text('[]')
    (out / 'paired-inputs.json').write_text(json.dumps(paired, indent=2))
    protocol |= {'sources': sources, 'construction_sources': construction_sources,
                 'tasks_sha256': digest((out / 'tasks.jsonl').read_bytes()),
                 'counterexamples_sha256': digest(b'[]'),
                 'paired_inputs_sha256': digest((out / 'paired-inputs.json').read_bytes()),
                 'counts': dict(Counter(task['split'] for task in all_tasks)),
                 'counterfactual_construction': dict(counts) | {'native_calls': native.native_calls},
                 'counterfactual_inputs': sum(bool(row['alternate_requests']) for row in paired),
                 'counterfactual_alternatives': len(extra),
                 'refrozen_from_protocol_sha256': digest((parent / 'protocol.json').read_bytes()),
                 'refreeze_reason': 'public-validation follow-up with paired semantic requests on training only',
                 'objective': 'categorical', 'edit_encoding': 'single_bit', 'attention_mode': 'dense',
                 'steps': 4096, 'neural_test_permitted': False,
                 'test_scope': 'public validation only; a fresh independent holdout is required for later method selection'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for name in sources | construction_sources:
        target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    verify(out)
    return {'counts': protocol['counts'], 'paired_inputs': protocol['counterfactual_inputs'],
            'extra_requests': len(extra), 'construction': protocol['counterfactual_construction'], 'test_queries': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'refreeze', 'train'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--alternatives', type=int, default=2)
    parser.add_argument('--batching', choices=('paired', 'shuffled'), default='paired')
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.phase == 'prepare':
        if not args.parent: parser.error('--parent is required')
        result = prepare(args.out, args.parent, alternatives=args.alternatives)
    elif args.phase == 'refreeze':
        if not args.parent: parser.error('--parent is required')
        parent_protocol = json.loads((external(args.parent) / 'protocol.json').read_text())
        if parent_protocol.get('neural_test_permitted') is not False or 'counterfactual_inputs' not in parent_protocol:
            raise ValueError('requires a frozen public-only counterfactual parent')
        refreeze(args.out, args.parent)
        out, protocol, tasks = verify(args.out)
        own = 'tools/research_counterfactual_repair.py'
        for name, identity in protocol['construction_sources'].items():
            origin = external(args.parent) / 'construction-source' / name
            if not origin.exists(): origin = external(args.parent) / 'source' / name
            raw = origin.read_bytes()
            if digest(raw) != identity: raise ValueError('original construction snapshot changed')
            preserved = out / 'construction-source' / name; preserved.parent.mkdir(parents=True, exist_ok=True)
            preserved.write_bytes(raw)
        protocol |= {'batch': 2, 'batching': args.batching,
                     'driver_sources': {own: digest(Path(__file__).read_bytes())},
                     'refreeze_reason': 'matched exposure control for conditional mini-batch organization'}
        protocol['training_schedule'] = schedule_identity(training_batches(tasks, protocol))
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        destination = out / 'source' / own; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(__file__).read_bytes())
        result = {'batching': args.batching, 'batch': 2, 'test_queries': 0,
                  'training_schedule': protocol['training_schedule']}
    else:
        _, protocol, _ = verify(args.out)
        own = 'tools/research_counterfactual_repair.py'
        driver = protocol.get('driver_sources', protocol['construction_sources'])
        if protocol.get('neural_test_permitted') is not False or driver.get(own) != digest(Path(__file__).read_bytes()):
            raise ValueError('requires the frozen public-only counterfactual protocol')
        result = train(args.out)
        (Path(args.out) / 'review.json').write_text(json.dumps(
            {'status': 'not_promoted', 'reason': 'public_validation_only', 'test_queries': 0,
             'major_breakthrough_claim': False}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
