"""Small, gated RGB semantic-repair experiment on disjoint program families.

The policy receives complete corrupted ELF bytes and visible input/output only.
Clean targets supervise training; hidden cases are used only after validation.
This pilot does not claim arbitrary-program repair or a research breakthrough.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elftools.elf.elffile import ELFFile
import torch
from torch.nn import functional as F

from lightnest_rgb.artifact import digest
from lightnest_rgb.corpus import external
from lightnest_rgb.exact_x86 import ExactX86Runner
from lightnest_rgb.delta_policy import DeltaRGBPolicy
from lightnest_rgb.execute import Case, ExecutableRunner, same_result
from lightnest_rgb.iterate import Example, RGBEditor, execution_frames
from lightnest_rgb.model import RGBConfig, RGBPolicy, greedy_action, patch_loss

SPLITS = {'arithmetic32': 'train', 'arithmetic64': 'train', 'logic32': 'train', 'loop': 'train',
          'unsigned_select': 'validation', 'memory': 'validation', 'signed_select': 'test', 'call': 'test'}
FIELDS = ('status', 'returncode', 'stdout', 'stderr')
COMMAND = 'Restore the executable to satisfy the visible program tests.'


def hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in
            ('lightnest_rgb/model.py', 'lightnest_rgb/delta_policy.py', 'lightnest_rgb/observe.py', 'lightnest_rgb/attention.py',
             'lightnest_rgb/context.py', 'lightnest_rgb/iterate.py', 'lightnest_rgb/feedback.py',
             'lightnest_rgb/artifact.py', 'lightnest_rgb/execute.py', 'lightnest_rgb/corpus.py',
             'lightnest_rgb/exact_core.py', 'lightnest_rgb/exact_x86.py', 'tools/research_exact_repair.py', 'uv.lock')}


def outputs(record):
    return {key: record[key] for key in FIELDS}


def operation_span(raw):
    elf = ELFFile(io.BytesIO(raw)); table = elf.get_section_by_name('.symtab')
    symbol = next(symbol for symbol in table.iter_symbols() if symbol.name == 'operation')
    section = elf.get_section(symbol.entry.st_shndx)
    start = section.header.sh_offset + symbol.entry.st_value - section.header.sh_addr
    return start, start + symbol.entry.st_size


def prepare(out, parent):
    out, parent = external(out), external(parent)
    protocol = json.loads((parent / 'protocol.json').read_text())
    summary = json.loads((parent / 'evaluation/summary.json').read_text())
    if not summary['passed'] or summary['protocol_sha256'] != digest((parent / 'protocol.json').read_bytes()):
        raise ValueError('requires a passed frozen differential parent')
    out.mkdir(parents=True, exist_ok=False)
    reference = {}
    for line in (parent / 'evaluation/results.jsonl').read_text().splitlines():
        row = json.loads(line); reference[row['program'], row['case']] = outputs(row['native'])
    core = ExactX86Runner(max_steps=512, trace_limit=0)
    native = ExecutableRunner(out / 'construction-native', timeout=.5, cache=False)
    tasks = []; counts = Counter(); exclusions = Counter(); selected = {}; counterexamples = []
    visible_indices = [1, 3, 5, 7]
    for program in protocol['programs']:
        if program['family'] not in SPLITS: continue
        raw = Path(program['path']).read_bytes()
        if digest(raw) != program['sha256']: raise ValueError('parent program changed')
        examples = [{'input': {'args': [], 'stdin': program['cases'][i]['stdin']},
                     'expected': reference[program['id'], i]} for i in visible_indices]
        hidden = [{'input': {'args': [], 'stdin': values['stdin']}, 'expected': reference[program['id'], i]}
                  for i, values in enumerate(program['cases']) if i not in visible_indices]
        start, stop = operation_span(raw)
        mutations = [(position, bit) for position in range(start, stop) for bit in range(8)]
        random.Random(program['sha256']).shuffle(mutations)
        group = []; seen = set()
        for position, bit in mutations[:96]:
            if len(group) >= 6: break
            damaged = bytearray(raw); damaged[position] ^= 1 << bit; damaged = bytes(damaged)
            if digest(damaged) in seen: continue
            seen.add(digest(damaged))
            actual = [core.run(damaged, Case.parse(example['input'])) for example in examples]
            if any(result['status'] != 'ok' for result in actual):
                exclusions['not_completed_by_tensor_core'] += 1; continue
            if all(same_result(result, example['expected']) for result, example in zip(actual, examples)):
                exclusions['no_visible_behavior_change'] += 1; continue
            independent = [native.run(damaged, Case.parse(example['input'])) for example in examples]
            if any(not same_result(a, b) for a, b in zip(actual, independent)):
                unstable = False
                for i, (a, b) in enumerate(zip(actual, independent)):
                    if same_result(a, b): continue
                    repeated = [outputs(b)] + [outputs(native.run(damaged, Case.parse(examples[i]['input']))) for _ in range(5)]
                    different = len({json.dumps(row, sort_keys=True) for row in repeated}) > 1
                    counterexamples.append({'program': program['id'], 'position': position, 'bit': bit,
                                            'visible_case': i, 'tensor': outputs(a), 'repeated_native': repeated,
                                            'native_output_unstable': different})
                    (out / 'counterexamples.json').write_text(json.dumps(counterexamples, indent=2))
                    if different:
                        unstable = True; break
                if unstable:
                    exclusions['unstable_native_output'] += 1; continue
                raise RuntimeError(f'stable tensor/native disagreement while constructing {program["id"]}, offset={position}, bit={bit}')
            record = {'id': f'{program["id"]}/{position}-{bit}', 'program': program['id'],
                      'family': program['family'], 'split': SPLITS[program['family']],
                      'image_path': program['path'], 'image_sha256': program['sha256'],
                      'patch': {'position': position, 'before': raw[position], 'after': damaged[position]},
                      'damaged_sha256': digest(damaged), 'visible': examples, 'hidden': hidden,
                      'observed': [outputs(row) for row in independent]}
            group.append(record); counts[record['split']] += 1
        if len(group) < 2: raise RuntimeError(f'not enough semantic mutants for {program["id"]}')
        group.sort(key=lambda row: digest(row['id'].encode()))
        selected[program['id']] = [row['id'] for row in group[:2]]
        tasks.extend(group)
        print(json.dumps({'program': program['id'], 'semantic_mutants': len(group), 'exclusions': dict(exclusions)}), flush=True)
    if len({task['image_sha256'] for task in tasks}) != len(selected):
        raise ValueError('duplicate programs across the frozen program split')
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in tasks))
    (out / 'counterexamples.json').write_text(json.dumps(counterexamples, indent=2))
    frozen = {'sources': hashes(), 'parent_protocol_sha256': digest((parent / 'protocol.json').read_bytes()),
              'tasks_sha256': digest((out / 'tasks.jsonl').read_bytes()), 'counts': dict(counts),
              'counterexamples_sha256': digest((out / 'counterexamples.json').read_bytes()),
              'parent_results_sha256': digest((parent / 'evaluation/results.jsonl').read_bytes()),
              'selection': selected, 'exclusions': dict(exclusions), 'family_splits': SPLITS,
              'validation_gate': 'mutant direct-action visible success >= 0.25 and strictly above initial weights',
              'test_queries': 0, 'width': 32, 'red_depths': [2], 'seed': 20261029,
              'steps': 512, 'batch': 1, 'learning_rate': .001, 'objective': 'balanced', 'edit_encoding': 'replacement',
              'scope': 'pilot on authored, single-bit semantic defects; program-family splits, no arbitrary repair claim'}
    (out / 'protocol.json').write_text(json.dumps(frozen, indent=2))
    snapshot = out / 'source'
    for name in hashes():
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return {'counts': dict(counts), 'selected_tasks': sum(map(len, selected.values())), 'native_calls': native.native_calls}


def verify(out):
    out = external(out); protocol = json.loads((out / 'protocol.json').read_text())
    if (protocol['sources'] != hashes() or protocol['tasks_sha256'] != digest((out / 'tasks.jsonl').read_bytes())
            or protocol['counterexamples_sha256'] != digest((out / 'counterexamples.json').read_bytes())):
        raise ValueError('frozen repair experiment changed')
    tasks = [json.loads(line) for line in (out / 'tasks.jsonl').read_text().splitlines()]
    for task in tasks:
        if digest(Path(task['image_path']).read_bytes()) != task['image_sha256']:
            raise ValueError('frozen clean program changed')
    return out, protocol, tasks


def refreeze(out, parent, *, steps=None, objective=None, edit_encoding=None):
    out, parent = external(out), external(parent)
    protocol = json.loads((parent / 'protocol.json').read_text())
    for name, value in protocol['sources'].items():
        if digest((parent / 'source' / name).read_bytes()) != value:
            raise ValueError('parent source snapshot changed')
    for name, key in (('tasks.jsonl', 'tasks_sha256'), ('counterexamples.json', 'counterexamples_sha256')):
        if digest((parent / name).read_bytes()) != protocol[key]:
            raise ValueError('parent input corpus changed')
    review = parent / 'review.json'
    if review.exists() and json.loads(review.read_text()).get('test_queries', 0):
        raise ValueError('previously queried tests cannot be refrozen as a new blind test')
    steps = protocol['steps'] if steps is None else steps
    objective = protocol.get('objective', 'balanced') if objective is None else objective
    edit_encoding = protocol.get('edit_encoding', 'replacement') if edit_encoding is None else edit_encoding
    if steps < 256 or steps > 16384 or steps % 256 or objective not in ('balanced', 'ranking') or edit_encoding not in ('replacement', 'xor'):
        raise ValueError('steps must be a multiple of 256 in 256..16384; objective balanced or ranking')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('tasks.jsonl', 'counterexamples.json'):
        (out / name).write_bytes((parent / name).read_bytes())
    protocol |= {'sources': hashes(), 'steps': steps, 'objective': objective, 'edit_encoding': edit_encoding,
                 'refrozen_from_protocol_sha256': digest((parent / 'protocol.json').read_bytes()),
                 'refreeze_reason': 'public-validation follow-up; unchanged task pool and unqueried neural test',
                 'validation_previously_exposed': True}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for name in hashes():
        target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    verify(out)
    return {'steps': steps, 'objective': objective, 'tasks_sha256': protocol['tasks_sha256'], 'test_queries': 0}


def ranking_loss(output, image, target):
    if len(image) != len(target): raise ValueError('equal complete images required')
    positions = [i for i, (a, b) in enumerate(zip(image, target)) if a != b]
    if len(positions) != 1: raise ValueError('ranking pilot requires exactly one changed byte')
    position = positions[0]; logits, bits = output['logits'], output['bits']
    expected = bits.new_tensor([(target[position] >> bit) & 1 for bit in range(8)])
    location = F.cross_entropy(logits[None], torch.tensor([position], device=logits.device))
    replacement = F.binary_cross_entropy_with_logits(bits[position], expected)
    positive = F.softplus(-logits[position])
    return location + 2 * replacement + .2 * positive


def make_policy(config, protocol):
    encoding = protocol.get('edit_encoding', 'replacement')
    if encoding not in ('replacement', 'xor'): raise ValueError('unknown edit encoding')
    return (DeltaRGBPolicy if encoding == 'xor' else RGBPolicy)(config)


def damaged(task):
    target = Path(task['image_path']).read_bytes(); image = bytearray(target)
    position = task['patch']['position']
    if image[position] != task['patch']['before']: raise ValueError('mutation precondition changed')
    image[position] = task['patch']['after']; image = bytes(image)
    if digest(image) != task['damaged_sha256']: raise ValueError('mutation changed')
    return image, target


def visible(task):
    return [Example(Case.parse(row['input']), row['expected']) for row in task['visible']]


@torch.inference_mode()
def direct_validation(model, tasks):
    model.eval(); core = ExactX86Runner(trace_limit=0, max_steps=512)
    records = []
    for task in tasks:
        image, _ = damaged(task); examples = visible(task)
        frames = execution_frames(examples, task['observed'])
        output, _, _ = model(image, COMMAND, frames, depth=2)
        action = greedy_action(output, image)
        candidate = bytearray(image)
        if not action['stop']: candidate[action['offset']] = action['value']
        actual = [core.run(bytes(candidate), example.case) for example in examples]
        passed = all(same_result(row, example.expected) for row, example in zip(actual, examples))
        records.append({'id': task['id'], 'success': passed, 'action': action})
    return {'tasks': len(records), 'successes': sum(row['success'] for row in records),
            'rate': sum(row['success'] for row in records) / len(records), 'records': records}


def train(out):
    out, protocol, tasks = verify(out)
    folder = out / 'training'; folder.mkdir(exist_ok=False)
    torch.manual_seed(protocol['seed']); torch.use_deterministic_algorithms(True)
    config = RGBConfig(width=32, heads=4, landmarks=2, detail_k=4, red_depths=(2,))
    model = make_policy(config, protocol); optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'])
    loss_function = ranking_loss if protocol.get('objective', 'balanced') == 'ranking' else patch_loss
    train_rows = [task for task in tasks if task['split'] == 'train']
    validation = [task for task in tasks if task['split'] == 'validation' and task['id'] in protocol['selection'][task['program']]]
    initial = direct_validation(model, validation)
    (folder / 'initial-validation.json').write_text(json.dumps(initial, indent=2))
    print(json.dumps({'phase': 'initial', 'validation_rate': initial['rate']}), flush=True)
    sampler = random.Random(protocol['seed']); started = time.perf_counter(); best = -1; loss_sum = 0
    stream = []; seen = set()
    with (folder / 'updates.jsonl').open('x') as log:
        for step in range(1, protocol['steps'] + 1):
            task = sampler.choice(train_rows); image, target = damaged(task)
            stream.append(task['id']); seen.add(task['id'])
            model.train(); optimizer.zero_grad(set_to_none=True)
            blue, red, _ = model(image, COMMAND, execution_frames(visible(task), task['observed']), depth=2)
            loss = loss_function(blue, image, target) + loss_function(red, image, target)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite repair loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            loss_sum += float(loss.detach())
            if step % 32 == 0:
                event = {'step': step, 'loss': loss_sum / 32, 'wall_seconds': time.perf_counter() - started}
                loss_sum = 0
                if step % 256 == 0:
                    result = direct_validation(model, validation); event['validation'] = result
                    if result['rate'] > best:
                        best = result['rate']
                        torch.save({'config': config.__dict__, 'state_dict': model.state_dict(), 'step': step,
                                    'validation': result, 'initial_rate': initial['rate'],
                                    'protocol_sha256': digest((out / 'protocol.json').read_bytes()),
                                    'sources': hashes(), 'seed': protocol['seed']}, folder / 'candidate.pt')
                log.write(json.dumps(event) + '\n'); log.flush()
                print(json.dumps({key: value if key != 'validation' else {k: v for k, v in value.items() if k != 'records'}
                                  for key, value in event.items()}), flush=True)
    torch.save({'config': config.__dict__, 'state_dict': model.state_dict(), 'step': protocol['steps'],
                'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'sampler_rng': sampler.getstate(), 'sources': hashes(),
                'protocol_sha256': digest((out / 'protocol.json').read_bytes())}, folder / 'final-state.pt')
    result = {'selected_validation_rate': best, 'initial_validation_rate': initial['rate'],
              'test_queries': 0, 'steps': protocol['steps'], 'unique_training_tasks': len(seen),
              'objective': protocol.get('objective', 'balanced'), 'edit_encoding': protocol.get('edit_encoding', 'replacement'),
              'stream_sha256': digest('\n'.join(stream).encode()),
              'validation_gate_passed': best >= .25 and best > initial['rate'],
              'major_breakthrough_claim': False}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def evaluate(out):
    out, protocol, tasks = verify(out)
    candidate = torch.load(out / 'training/candidate.pt', map_location='cpu', weights_only=True)
    if candidate['sources'] != hashes() or candidate['protocol_sha256'] != digest((out / 'protocol.json').read_bytes()):
        raise ValueError('candidate does not match frozen protocol')
    result = {'test_queries': 0, 'status': 'not_promoted', 'major_breakthrough_claim': False}
    if candidate['validation']['rate'] < .25 or candidate['validation']['rate'] <= candidate['initial_rate']:
        result['reason'] = 'validation_gate_failed'
        (out / 'review.json').write_text(json.dumps(result, indent=2)); return result
    config = candidate['config']; config['red_depths'] = tuple(config['red_depths'])
    model = make_policy(RGBConfig(**config), protocol); model.load_state_dict(candidate['state_dict']); model.eval()
    folder = out / 'evaluation'; folder.mkdir(exist_ok=False)
    native = ExecutableRunner(folder / 'native', cache=False)
    selected = [task for task in tasks if task['split'] == 'test' and task['id'] in protocol['selection'][task['program']]]
    records = []
    for task in selected:
        image, _ = damaged(task)
        arms = {}
        for name, runner in [('tensor', ExactX86Runner(trace_limit=0, max_steps=512)), ('native', native)]:
            editor = RGBEditor(model, runner, max_edits=1, max_candidates=9, wall_seconds=120, blue_risk=1., budgeted=False)
            proposal = editor.repair(image, COMMAND, visible(task))
            checked = [native.run(proposal['bytes'], Case.parse(row['input'])) for row in task['hidden']] if proposal['stop'] else []
            strict = proposal['stop'] and all(same_result(actual, row['expected']) for actual, row in zip(checked, task['hidden']))
            arms[name] = {key: value for key, value in proposal.items() if key != 'bytes'} | {'hidden_success': strict}
        records.append({'id': task['id'], 'arms': arms})
    (folder / 'results.json').write_text(json.dumps(records, indent=2))
    result |= {'reason': 'held_out_pilot_complete', 'test_queries': len(selected),
               'tasks': len(selected), 'checkpoint_step': candidate['step'],
               'arms': {name: {'hidden_successes': sum(row['arms'][name]['hidden_success'] for row in records),
                               'returned_candidates': sum(row['arms'][name]['stop'] for row in records)} for name in ('tensor', 'native')}}
    (out / 'review.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'refreeze', 'train', 'evaluate'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--objective', choices=('balanced', 'ranking'))
    parser.add_argument('--edit-encoding', choices=('replacement', 'xor'))
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.phase in ('prepare', 'refreeze'):
        if not args.parent: parser.error('--parent is required')
        result = (prepare(args.out, args.parent) if args.phase == 'prepare' else
                  refreeze(args.out, args.parent, steps=args.steps, objective=args.objective, edit_encoding=args.edit_encoding))
    else: result = train(args.out) if args.phase == 'train' else evaluate(args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
