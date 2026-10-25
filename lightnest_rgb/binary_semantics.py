"""Joint bytecode/source alignment and learned free-text explanations.

Compilers and docstrings create supervision only. Inference reads complete
artifact bytes through the same lossless tokenizer and neural RGB core.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
import hashlib
import importlib.util
import json
import marshal
import mmap
from pathlib import Path
import random
import re
import sys
import time

import torch
from torch.nn import functional as F

from .any2any import ArtifactCore, DEFAULT_MAX_BYTES, semantic_similarity
from .byte_tokenizer import ByteBPE
from .corpus import external
from .semantics import ROOT, lexical_scores, metrics, read, sha, write

OBJECTIVE = 'external-binary-language-v1'
VIEWS = ('source', 'bytecode')


def sources():
    return {name: sha(Path(__file__).parent / name)
            for name in ('any2any.py', 'context.py', 'byte_tokenizer.py', 'binary_semantics.py')}


def compile_bytecode(code):
    tree = ast.parse(code)
    if any(isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
           and ast.get_docstring(node) for node in ast.walk(tree)):
        raise ValueError('documentation must be removed before binary supervision')
    compiled = compile(tree, '<artifact>', 'exec', dont_inherit=True, optimize=0)
    return importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(compiled)


def behavior_probes():
    families = [
        ('greater_less', '{a} > {b}', '{a} < {b}', 'Return whether {a} is greater than {b}.', 'Return whether {a} is less than {b}.'),
        ('ge_le', '{a} >= {b}', '{a} <= {b}', 'Return whether {a} is at least {b}.', 'Return whether {a} is at most {b}.'),
        ('subtract', '{a} - {b}', '{b} - {a}', 'Return the result of subtracting {b} from {a}.', 'Return the result of subtracting {a} from {b}.'),
        ('divide', '{a} / {b}', '{b} / {a}', 'Return the result of dividing {a} by {b}.', 'Return the result of dividing {b} by {a}.'),
        ('floor_divide', '{a} // {b}', '{b} // {a}', 'Return the integer quotient of {a} divided by {b}.', 'Return the integer quotient of {b} divided by {a}.'),
        ('sum_product', '{a} + {b}', '{a} * {b}', 'Return the sum of the two arguments.', 'Return the product of the two arguments.'),
        ('max_min', '{a} if {a} > {b} else {b}', '{a} if {a} < {b} else {b}', 'Return the larger of the two arguments.', 'Return the smaller of the two arguments.'),
        ('equal_unequal', '{a} == {b}', '{a} != {b}', 'Return whether the two arguments are equal.', 'Return whether the two arguments are different.'),
    ]
    cases = [(3, 7), (8, 2), (-4, 5), (2, 2), (-6, -2), (5, 1)]
    probes = []
    for family, first, second, text_a, text_b in families:
        for a, b in [('left', 'right'), ('first', 'second'), ('a', 'b'), ('value', 'limit')]:
            programs = []
            for expression, description in [(first, text_a), (second, text_b)]:
                source = f'def operation({a}, {b}):\n    return {expression.format(a=a, b=b)}\n'
                binary = compile_bytecode(source)
                namespace = {}
                # Only these fixed, locally authored arithmetic programs execute.
                exec(marshal.loads(binary[16:]), {'__builtins__': {}}, namespace)
                observed = [namespace['operation'](*values) for values in cases]
                programs.append({'source': source, 'bytecode_hex': binary.hex(),
                                 'description': description.format(a=a, b=b), 'outputs': observed})
            if programs[0]['outputs'] == programs[1]['outputs']:
                raise ValueError('counterfactual probe has no behavioral difference')
            probes.append({'id': f'{family}/{a}-{b}', 'cases': cases, 'programs': programs})
    return probes


class Images:
    def __init__(self, out):
        self.file = (Path(out) / 'bytecode.bin').open('rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, row, view):
        if view == 'source':
            return row['code'].encode()
        if view != 'bytecode':
            raise ValueError('unknown artifact view')
        item = row['bytecode']
        return self.data[item['offset']:item['offset'] + item['size']]

    def close(self):
        self.data.close(); self.file.close()


def _words(text):
    return set(re.findall(r'[a-z]{3,}', text.lower().replace('_', ' ')))


def mine_negatives(rows):
    words = [_words(row['description']) for row in rows]
    index = defaultdict(list)
    for i, values in enumerate(words):
        for word in sorted(values):
            index[word].append(i)
    result = []
    for i, values in enumerate(words):
        rare = sorted((word for word in values if 1 < len(index[word]) <= 256), key=lambda word: (len(index[word]), word))[:3]
        candidates = sorted(set(j for word in rare for j in index[word]) - {i})
        rng = random.Random(rows[i]['code_sha256'])
        if len(candidates) > 64:
            candidates = rng.sample(candidates, 64)
        if not candidates:
            candidates = [j for j in rng.sample(range(len(rows)), min(32, len(rows))) if j != i]
        def similarity(j):
            common = values & words[j]
            return sum(1 / len(index[word]) for word in common)
        result.append(sorted(candidates, key=lambda j: (-similarity(j), j))[:8])
    return result


def prepare(out, parent, *, max_bytes=4096, vocab_size=2048):
    out, parent = external(out), external(parent)
    old = json.loads((parent / 'protocol.json').read_text())
    if any(sha(parent / f'{name}.jsonl') != digest for name, digest in old['splits'].items()):
        raise ValueError('parent input identities changed')
    out.mkdir(parents=True, exist_ok=False)
    counts = {}; exclusions = []; seen = set()
    with (out / 'bytecode.bin').open('xb') as archive:
        for split in ('self', 'test', 'validation', 'train'):
            kept = []
            for original in read(parent, split):
                row = dict(original)
                if split == 'train' and Path(row['path']).resolve().is_relative_to(ROOT):
                    raise ValueError('LightNest cannot enter the training corpus')
                try:
                    binary = compile_bytecode(row['code'])
                except SyntaxError as error:
                    exclusions.append({'id': row['id'], 'reason': 'standalone_compile', 'error': error.msg}); continue
                digest = hashlib.sha256(binary).hexdigest()
                if len(binary) > max_bytes or digest in seen:
                    exclusions.append({'id': row['id'], 'reason': 'byte_capacity_or_duplicate'}); continue
                seen.add(digest)
                row['bytecode'] = {'offset': archive.tell(), 'size': len(binary), 'sha256': digest}
                archive.write(binary); kept.append(row)
            with (out / f'{split}.jsonl').open('x') as file:
                for row in kept:
                    file.write(json.dumps(row) + '\n')
            counts[split] = {'examples': len(kept), 'projects': len({row['group'] for row in kept})}
    write(out / 'exclusions.json', exclusions)
    write(out / 'behavior_probes.json', behavior_probes())
    images = Images(out); train_rows = read(out, 'train')
    groups = defaultdict(list)
    for row in sorted(train_rows, key=lambda row: row['code_sha256']):
        groups[row['group']].append(row)
    tokenizer_rows = []
    while len(tokenizer_rows) < min(4096, len(train_rows)):
        for group in sorted(groups):
            if groups[group] and len(tokenizer_rows) < 4096:
                tokenizer_rows.append(groups[group].pop())
    write(out / 'tokenizer_training_ids.json', [row['id'] for row in tokenizer_rows])
    def training_values():
        for row in tokenizer_rows:
            yield images.get(row, 'source'); yield images.get(row, 'bytecode'); yield row['description'].encode()
    tokenizer = ByteBPE.train(training_values(), vocab_size=vocab_size)
    (out / 'tokenizer.json').write_text(tokenizer.serialized)
    neighbors = mine_negatives(train_rows)
    write(out / 'negatives.json', neighbors)
    capacity = {}
    lexical = {}
    for split in ('train', 'validation', 'test', 'self'):
        rows = read(out, split)
        capacity[split] = {view: max(len(tokenizer.encode(images.get(row, view))) for row in rows) for view in VIEWS}
        for row in rows:
            if hashlib.sha256(images.get(row, 'bytecode')).hexdigest() != row['bytecode']['sha256']:
                raise ValueError('compiled bytecode identity changed')
        if split == 'validation':
            lexical = {view: metrics(lexical_scores(view_rows(rows, images, view))) for view in VIEWS}
    images.close()
    snapshot = out / 'source'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        (snapshot / path.name).write_bytes(path.read_bytes())
    files = ['bytecode.bin', 'tokenizer.json', 'tokenizer_training_ids.json', 'negatives.json', 'exclusions.json', 'behavior_probes.json',
             *[f'{name}.jsonl' for name in counts]]
    write(out / 'protocol.json', {'objective': OBJECTIVE, 'counts': counts, 'capacity': capacity,
          'files': {name: sha(out / name) for name in files}, 'sources': sources(),
          'parent_protocol_sha256': sha(parent / 'protocol.json'), 'max_bytes': max_bytes,
          'inference_byte_capacity': DEFAULT_MAX_BYTES,
          'vocab_size': tokenizer.vocab_size, 'python_cache_tag': sys.implementation.cache_tag,
          'tokenizer_training_examples': len(tokenizer_rows),
          'pyc_magic': importlib.util.MAGIC_NUMBER.hex(), 'uv_lock_sha256': sha(ROOT / 'uv.lock'),
          'lexical_validation': lexical, 'test_queries': 0, 'self_queries': 0,
          'selection': 'maximize the weakest validation recall@1 margin over lexical across both views',
          'gate': 'both validation views beat lexical and initial neural weights, paired behavior >=0.75 on both views, bytecode language NLL gain >=0.1 before test/self',
          'scope': 'compiled Python bytecode, not Linux native machine-code understanding'})
    return {'counts': counts, 'capacity': capacity, 'lexical_validation': lexical}


def verify(out):
    out = external(out); protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['sources'] != sources() or protocol['uv_lock_sha256'] != sha(ROOT / 'uv.lock'):
        raise ValueError('frozen binary implementation or dependencies changed')
    if any(sha(out / name) != digest for name, digest in protocol['files'].items()):
        raise ValueError('frozen binary research data changed')
    return out, protocol


def view_rows(rows, images, view):
    return [row | {'code': images.get(row, view).decode('latin1')} for row in rows]


@torch.inference_mode()
def represent(model, values, batch_size=8):
    order = sorted(range(len(values)), key=lambda i: len(values[i]))
    chunks = []
    for start in range(0, len(order), batch_size):
        chunks.append(model.semantic_tokens([values[i] for i in order[start:start + batch_size]]))
    length = max(chunk['features'].shape[1] for chunk in chunks)
    result = {}
    for key in ('features', 'valid', 'weights', 'tokens'):
        padded = []
        for chunk in chunks:
            extra = length - chunk[key].shape[1]
            pad = (0, 0, 0, extra) if key == 'features' else (0, extra)
            padded.append(F.pad(chunk[key], pad))
        result[key] = torch.cat(padded)[torch.tensor(order).argsort()]
    return result


@torch.inference_mode()
def score_matrix(left, right, block=8):
    result = torch.empty(left['features'].shape[0], right['features'].shape[0])
    for i in range(0, result.shape[0], block):
        for j in range(0, result.shape[1], block):
            a = {key: value[i:i + block] for key, value in left.items()}
            b = {key: value[j:j + block] for key, value in right.items()}
            a_length = int(a['valid'].sum(-1).max())
            b_length = int(b['valid'].sum(-1).max())
            a = {key: value[:, :a_length] for key, value in a.items()}
            b = {key: value[:, :b_length] for key, value in b.items()}
            result[i:i + block, j:j + block] = semantic_similarity(a, b)
    return result


def language_targets(model, descriptions):
    sequences = [model.semantic_tokenizer.encode(text.encode()) for text in descriptions]
    device = model.byte.weight.device
    previous = torch.zeros(len(sequences), max(map(len, sequences)) + 1, dtype=torch.long, device=device)
    target = torch.full_like(previous, -100)
    for i, ids in enumerate(sequences):
        previous[i, :len(ids) + 1] = torch.tensor([model.semantic_tokenizer.bos, *ids], device=device)
        target[i, :len(ids) + 1] = torch.tensor([*ids, model.semantic_tokenizer.eos], device=device)
    return previous, target


@torch.inference_mode()
def validation(model, rows, images):
    docs = represent(model, [row['description'].encode() for row in rows])
    result = {}
    for view in VIEWS:
        memory = represent(model, [images.get(row, view) for row in rows])
        result[view] = metrics(score_matrix(memory, docs))
    return result


def sample_batch(rng, count, batch_size, neighbors):
    if count < batch_size:
        raise ValueError('not enough distinct training examples for the batch')
    chosen = rng.sample(range(count), batch_size // 2)
    for anchor in list(chosen):
        candidates = [index for index in neighbors[anchor] if index not in chosen]
        if candidates:
            chosen.append(rng.choice(candidates))
    while len(chosen) < batch_size:
        index = rng.randrange(count)
        if index not in chosen:
            chosen.append(index)
    return chosen


def train(out, *, steps=4096, seed=20261027, batch_size=8):
    out, protocol = verify(out)
    if steps < 1 or batch_size < 4 or batch_size % 2:
        raise ValueError('positive steps and an even batch of at least four required')
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    folder = out / f'training-{seed}'; folder.mkdir(exist_ok=False)
    tokenizer = ByteBPE((out / 'tokenizer.json').read_text())
    longest = max(value for views in protocol['capacity'].values() for value in views.values())
    config = {'width': 64, 'heads': 4, 'max_bytes': protocol['inference_byte_capacity'],
              'sequence_limit': max(256, min(512, longest)), 'note_slots': 16, 'semantic_bpe': tokenizer.serialized}
    model = ArtifactCore(**config); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rows = read(out, 'train'); val = read(out, 'validation'); images = Images(out)
    neighbors = json.loads((out / 'negatives.json').read_text())
    rng = random.Random(seed); started = time.perf_counter(); best = -float('inf'); losses = []; seen = set()
    stream = hashlib.sha256(); presentations = Counter()
    write(folder / 'plan.json', {'steps': steps, 'seed': seed, 'batch_size': batch_size,
          'parameters': sum(p.numel() for p in model.parameters()), 'config': config,
          'protocol_sha256': sha(out / 'protocol.json'), 'sources': sources(),
          'objective': 'symmetric contrastive + 0.25 * autoregressive copy/generate NLL',
          'sampler': 'half random anchors, half training-description hard neighbors'})
    model.eval()
    torch.save({'objective': OBJECTIVE, 'config': config, 'state_dict': model.state_dict(),
                'language_steps': 0, 'protocol_sha256': sha(out / 'protocol.json'),
                'sources': sources()}, folder / 'initial.pt')
    initial = validation(model, val, images)
    write(folder / 'initial-validation.json', initial)
    print(json.dumps({'phase': 'initial_neural_weights', 'validation': initial}), flush=True)
    with (folder / 'updates.jsonl').open('x') as log:
        for step in range(1, steps + 1):
            indices = sample_batch(rng, len(rows), batch_size, neighbors)
            chosen = [rows[index] for index in indices]
            view = VIEWS[step % 2]
            for row in chosen:
                seen.add(row['id']); presentations[view] += 1
                stream.update(f"{view}:{row['id']}\n".encode())
            model.train()
            memory = model.semantic_tokens([images.get(row, view) for row in chosen])
            docs = model.semantic_tokens([row['description'].encode() for row in chosen])
            scores = semantic_similarity(memory, docs) / .07
            expected = torch.arange(batch_size)
            contrastive = (F.cross_entropy(scores, expected) + F.cross_entropy(scores.T, expected)) / 2
            previous, target = language_targets(model, [row['description'] for row in chosen])
            output, _ = model.language_forward(memory, previous)
            language = F.nll_loss(output.flatten(0, 1), target.flatten())
            loss = contrastive + .25 * language
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            losses.append([float(contrastive.detach()), float(language.detach())])
            if step % 32 == 0 or step == steps:
                event = {'step': step, 'contrastive_loss': sum(row[0] for row in losses) / len(losses),
                         'language_nll': sum(row[1] for row in losses) / len(losses),
                         'wall_seconds': time.perf_counter() - started}
                if step % 512 == 0 or step == steps:
                    model.eval(); event['validation'] = validation(model, val, images)
                    margin = min(event['validation'][view]['recall_at_1'] - protocol['lexical_validation'][view]['recall_at_1'] for view in VIEWS)
                    event['selection_margin'] = margin
                    payload = {'objective': OBJECTIVE, 'config': config, 'state_dict': model.state_dict(),
                               'step': step, 'language_steps': step, 'sources': sources(), 'initial_validation': initial,
                               'initial_checkpoint_sha256': sha(folder / 'initial.pt'),
                               'protocol_sha256': sha(out / 'protocol.json'), 'validation': event['validation'],
                               'presentations': dict(presentations), 'unique_functions': len(seen),
                               'stream_sha256': stream.hexdigest()}
                    torch.save(payload | {'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                                          'sampler_rng': rng.getstate()}, folder / f'step-{step}.pt')
                    if margin > best:
                        best = margin; torch.save(payload, folder / f'candidate-{step}.pt')
                        (folder / 'selection.json').write_text(json.dumps({'checkpoint': f'candidate-{step}.pt', 'margin': best}))
                log.write(json.dumps(event) + '\n'); log.flush(); print(json.dumps(event), flush=True); losses = []
    images.close()
    return json.loads((folder / 'selection.json').read_text())


@torch.inference_mode()
def rank(model, raw, descriptions):
    left = represent(model, [raw]); right = represent(model, [text.encode() for text in descriptions])
    scores = score_matrix(left, right)[0]
    return [{'description': descriptions[i], 'score': float(scores[i])} for i in scores.argsort(descending=True).tolist()]


@torch.inference_mode()
def language_diagnostic(model, rows, images):
    rows = rows[:64]
    values = [images.get(row, 'bytecode') for row in rows]
    texts = [row['description'] for row in rows]
    result = {}
    for name, artifacts in [('matched', values), ('wrong_source', values[1:] + values[:1])]:
        total = count = 0
        for start in range(0, len(rows), 8):
            memory = model.semantic_tokens(artifacts[start:start + 8])
            previous, target = language_targets(model, texts[start:start + 8])
            output, _ = model.language_forward(memory, previous)
            total += float(F.nll_loss(output.flatten(0, 1), target.flatten(), reduction='sum'))
            count += int(target.ne(-100).sum())
        result[name + '_nll'] = total / count
    result['conditioning_gain'] = result['wrong_source_nll'] - result['matched_nll']
    result['examples'] = len(rows)
    result['free_generation'] = [{'id': row['id'], 'reference': row['description'],
                                  'generated': model.explain_bytes(raw)} for row, raw in zip(rows[:4], values[:4])]
    return result


@torch.inference_mode()
def probe_evaluation(model, probes):
    result = {}
    for view in VIEWS:
        records = []
        for probe in probes:
            programs = probe['programs']
            artifacts = [row['source'].encode() if view == 'source' else bytes.fromhex(row['bytecode_hex']) for row in programs]
            docs = [row['description'].encode() for row in programs]
            scores = score_matrix(represent(model, artifacts), represent(model, docs))
            lexical = lexical_scores([{'code': raw.decode('latin1'), 'description': row['description']}
                                      for raw, row in zip(artifacts, programs)])
            records.append({'id': probe['id'], 'neural_paired_correct': bool(torch.equal(scores.argmax(1), torch.arange(2))),
                            'lexical_paired_correct': bool(torch.equal(lexical.argmax(1), torch.arange(2)))})
        result[view] = {'pairs': len(records),
                        'neural_paired_accuracy': sum(row['neural_paired_correct'] for row in records) / len(records),
                        'lexical_paired_accuracy': sum(row['lexical_paired_correct'] for row in records) / len(records),
                        'rows': records}
    return result


def evaluate(out, checkpoint):
    out, protocol = verify(out)
    checkpoint = Path(checkpoint)
    destination = checkpoint.with_suffix('.review.json')
    checkpoint_sha = sha(checkpoint)
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('evaluated checkpoint changed')
        return {key: value for key, value in previous.items() if key not in {'language', 'behavior', 'test', 'self'}}
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload.get('objective') != OBJECTIVE or payload['protocol_sha256'] != sha(out / 'protocol.json')
            or payload['sources'] != sources()):
        raise ValueError('candidate identity does not match the frozen binary experiment')
    model = ArtifactCore(**payload['config']); model.load_state_dict(payload['state_dict']); model.eval()
    images = Images(out); rows = read(out, 'validation')
    result = {'checkpoint_sha256': checkpoint_sha, 'protocol_sha256': sha(out / 'protocol.json'),
              'validation': validation(model, rows, images), 'lexical_validation': protocol['lexical_validation'],
              'test_queries': 0, 'self_queries': 0, 'scope': protocol['scope']}
    result['language'] = language_diagnostic(model, rows, images)
    result['initial_validation'] = payload['initial_validation']
    validation_pass = all(result['validation'][view]['recall_at_1'] > max(protocol['lexical_validation'][view]['recall_at_1'], payload['initial_validation'][view]['recall_at_1']) for view in VIEWS)
    result['status'] = 'not_promoted'
    result['reason'] = 'validation_below_lexical'
    if validation_pass:
        result['behavior'] = probe_evaluation(model, json.loads((out / 'behavior_probes.json').read_text()))
        behavior_pass = all(result['behavior'][view]['neural_paired_accuracy'] >= .75 for view in VIEWS)
        language_pass = result['language']['conditioning_gain'] >= .1
        result['reason'] = 'behavior_or_language_gate_failed'
        if behavior_pass and language_pass:
            for split in ('test', 'self'):
                held_out = read(out, split)
                result[split] = validation(model, held_out, images)
                result[split + '_queries'] = len(held_out) * len(VIEWS)
                result[split + '_lexical'] = {view: metrics(lexical_scores(view_rows(held_out, images, view))) for view in VIEWS}
            passed = all(result[split][view]['recall_at_1'] > result[split + '_lexical'][view]['recall_at_1']
                         for split in ('test', 'self') for view in VIEWS)
            result['status'] = 'alignment_gate_passed' if passed else 'not_promoted'
            result['reason'] = 'held_out_gate_passed' if passed else 'held_out_gate_failed'
    write(destination, result); images.close()
    return {key: value for key, value in result.items() if key not in {'language', 'behavior', 'test', 'self'}}
