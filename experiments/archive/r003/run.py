"""Frozen train/validation/test tournament with compact automatic archival."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import torch
from torch.nn import functional as F

from .models import Candidate
from weightlang.modules import save_module, load_module
from .tasks import condition_tensor, compile_target

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('mlp', 'gru', 'conv', 'transformer')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def state(**values):
    write(ROOT / 'experiments/STATUS.json', dict(updated_utc=datetime.now(timezone.utc).isoformat(), **values))


def dataset(seed):
    rng = random.Random(seed)
    values = list(range(256)); rng.shuffle(values)
    rows = []
    for i, value in enumerate(values):
        split = 'train' if i < 192 else 'validation' if i < 224 else 'test'
        rows.append(dict(id=f'exit-{value}', family='exit', split=split, spec={'exit': value}))
    used = set()
    for split, count in [('train', 192), ('validation', 32), ('test', 64)]:
        for i in range(count):
            payload = rng.randbytes(8)
            while payload in used: payload = rng.randbytes(8)
            used.add(payload)
            rows.append(dict(id=f'stdout-{split}-{i}', family='stdout', split=split, spec={'stdout': payload.hex()}))
    for row in rows:
        row['raw'] = compile_target(row['spec'])
        row['sha256'] = digest(row['raw'])
    hashes = {s: {r['sha256'] for r in rows if r['split'] == s} for s in ('train', 'validation', 'test')}
    assert not any(hashes[a] & hashes[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    return rows


def tensors(rows, capacity):
    targets = torch.zeros(len(rows), capacity + 1, 9)
    valid = torch.zeros(len(rows), capacity + 1, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row['raw']); raw = torch.tensor(list(row['raw']), dtype=torch.long)
        targets[i, :n, :8] = ((raw[:, None] >> torch.arange(8)) & 1).float()
        targets[i, n, 8] = 1
        valid[i, :n + 1] = True
    return condition_tensor([r['spec'] for r in rows]), targets, valid


def variable_fields(rows):
    return {family: [j for j in range(len(raw[0])) if len({x[j] for x in raw}) > 1]
            for family in ('exit', 'stdout')
            for raw in [[r['raw'] for r in rows if r['split'] == 'train' and r['family'] == family]]}


def objective(output, targets, valid, weight):
    error = F.binary_cross_entropy_with_logits(output, targets, reduction='none')
    byte_mask = valid & ~targets[..., 8].bool()
    byte_loss = (error[..., :8].mean(-1) * weight)[byte_mask].mean()
    eos = targets[..., 8].bool()
    return byte_loss + .5 * (error[..., 8][eos].mean() + error[..., 8][byte_mask].mean())


@torch.inference_mode()
def measure(model, rows, packed, fields, weight):
    model.eval(); condition, targets, valid = packed
    total_loss = 0.; cases = []
    start = time.perf_counter()
    for offset in range(0, len(rows), 8):
        output = model(condition[offset:offset + 8])
        if not torch.isfinite(output).all(): raise FloatingPointError('nonfinite evaluation')
        total_loss += float(objective(output, targets[offset:offset+8], valid[offset:offset+8], weight)) * len(output)
        values = ((output[..., :8] >= 0).long() * (1 << torch.arange(8))).sum(-1)
        for i, row in enumerate(rows[offset:offset+8]):
            stops = (output[i, :, 8] >= 0).nonzero().flatten()
            stopped = bool(len(stops)); length = int(stops[0]) if stopped else model.capacity
            raw = bytes(values[i, :length].tolist()); positions = fields[row['family']]
            cases.append(dict(id=row['id'], family=row['family'], exact=stopped and raw == row['raw'],
                              stopped=stopped, target_size=len(row['raw']), generated_size=len(raw),
                              target_sha256=row['sha256'], generated_sha256=digest(raw),
                              variable_correct=sum(j < len(raw) and raw[j] == row['raw'][j] for j in positions),
                              variable_count=len(positions)))
    family_scores = {f: sum(c['exact'] for c in cases if c['family'] == f) / sum(c['family'] == f for c in cases)
                     for f in ('exit', 'stdout')}
    return dict(loss=total_loss/len(rows), exact=sum(c['exact'] for c in cases), count=len(cases),
                macro_exact=statistics.mean(family_scores.values()), families=family_scores,
                variable_accuracy=sum(c['variable_correct'] for c in cases)/sum(c['variable_count'] for c in cases),
                seconds=time.perf_counter()-start), cases


def rank(metrics):
    return (-metrics['macro_exact'], -metrics['variable_accuracy'], metrics.get('parameters', 0), metrics['loss'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--variable-weight', type=float, default=8.)
    parser.add_argument('--hypothesis', default='Compare parallel complete-file decoders without target bytes as inputs.')
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37])
    parser.add_argument('--data-seed', type=int, default=101)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS + tuple(v+'_copy' for v in VARIANTS) + ('mlp_copy_small', 'conv_copy_small'), default=list(VARIANTS))
    args = parser.parse_args()
    if not args.round.replace('-', '').isalnum(): parser.error('round must be alphanumeric')
    if min(args.steps, args.width, args.batch_size, args.threads) < 1 or args.width % 4: parser.error('invalid budget or width')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants): parser.error('duplicate seed or variant')
    archive = ROOT/'experiments/archive'/args.round
    cache = ROOT/'.cache/experiments'/args.round
    durable = ROOT/'weights/experiments'/args.round
    if any(p.exists() for p in (archive, cache, durable)): parser.error('round already exists; choose a new ID')
    archive.mkdir(parents=True); cache.mkdir(parents=True); durable.mkdir(parents=True)
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    protocol = dict(id=args.round, hypothesis=args.hypothesis,
                    args=vars(args), selection='best validation macro exact, variable-byte accuracy, parameter count, then loss; aggregate seeds',
                    budget='same minibatch indices and steps; not FLOP, parameter or wall-time matched',
                    primary='macro family full-file exact match including EOS',
                    limits='two fixed assembler layouts; no arbitrary program, inspect or edit claim; test never selects winner',
                    runtime=dict(torch=torch.__version__, device='cpu', threads=args.threads),
                    source_hashes={})
    for file in sorted(Path(__file__).parent.glob('*')):
        if file.is_file():
            shutil.copy2(file, archive/file.name); protocol['source_hashes'][str(file.relative_to(ROOT))] = digest(file.read_bytes())
    write(archive/'protocol.json', protocol)
    state(phase='preparing', round=args.round, incumbent=None, next_action='train all candidates before opening blind test')
    rows = dataset(args.data_seed); capacity = max(len(r['raw']) for r in rows)
    fields = variable_fields(rows)
    write(archive/'questions.json', [{k:v for k,v in r.items() if k != 'raw'} for r in rows])
    write(archive/'dataset.json', dict(capacity=capacity, train_variable_positions=fields,
          counts={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}, exact_overlap=0))
    groups = {s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    packed = {s:tensors(group, capacity) for s,group in groups.items()}
    weight = torch.ones(capacity+1)
    for indices in fields.values(): weight[indices] = args.variable_weight
    trained = []
    for seed in args.seeds:
        schedule = torch.randint(len(groups['train']), (args.steps,args.batch_size), generator=torch.Generator().manual_seed(seed+9000))
        for variant in args.variants:
            run = f'{variant}-s{seed}'
            state(phase='training', round=args.round, current=run, incumbent=None, next_action='finish validation selection')
            torch.manual_seed(seed); model=Candidate(variant,args.width,capacity,transfer=args.transfer)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            best=None; curve=[]; seconds=0.
            for step,indices in enumerate(schedule,1):
                model.train(); optimizer.zero_grad(set_to_none=True); start=time.perf_counter()
                condition,targets,valid=(v[indices] for v in packed['train'])
                loss=objective(model(condition),targets,valid,weight)
                if not torch.isfinite(loss): raise FloatingPointError(run)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                seconds+=time.perf_counter()-start
                if step % 80 == 0 or step == args.steps:
                    metrics,_=measure(model,groups['validation'],packed['validation'],fields,weight)
                    curve.append(dict(step=step,validation=metrics))
                    if best is None or rank(metrics)<rank(best):
                        best=metrics; best_step=step; torch.save(model.state_dict(),cache/(run+'.pt'))
                    print(json.dumps(dict(run=run,step=step,validation=metrics,train_seconds=seconds)),flush=True)
            trained.append(dict(run=run,variant=variant,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                                best_step=best_step,validation=best,curve=curve,train_seconds=seconds,
                                schedule_sha256=digest(schedule.numpy().tobytes())))
    aggregate={v:dict(parameters=next(r['parameters'] for r in trained if r['variant']==v), macro_exact=statistics.mean(r['validation']['macro_exact'] for r in trained if r['variant']==v),
                      variable_accuracy=statistics.mean(r['validation']['variable_accuracy'] for r in trained if r['variant']==v),
                      loss=statistics.mean(r['validation']['loss'] for r in trained if r['variant']==v)) for v in args.variants}
    winner=min(args.variants,key=lambda v:rank(aggregate[v]))
    selected=min((r for r in trained if r['variant']==winner),key=lambda r:rank(r['validation']))
    write(archive/'selection.json',dict(winner=winner,selected_run=selected['run'],validation=aggregate,training=trained))
    state(phase='blind_evaluation',round=args.round,incumbent=winner,next_action='evaluate frozen checkpoints once')
    # Selection is persisted before any access to blind-test metrics.
    for record in trained:
        model=Candidate(record['variant'],args.width,capacity,transfer=args.transfer)
        model.load_state_dict(torch.load(cache/(record['run']+'.pt'),weights_only=True))
        metrics,cases=measure(model,groups['test'],packed['test'],fields,weight)
        record['test']=metrics
        write(archive/(record['run']+'-cases.json'),cases)
        print(json.dumps(dict(run=record['run'],test=metrics)),flush=True)
    retained=Candidate(selected['variant'],args.width,capacity,transfer=args.transfer).eval()
    retained.load_state_dict(torch.load(cache/(selected['run']+'.pt'),weights_only=True))
    save_module(retained,durable/'model',kwargs=dict(variant=selected['variant'],width=args.width,
                capacity=capacity,transfer=args.transfer),origin='trained',
                training=dict(method='AdamW',steps=selected['best_step'],dataset=args.round),
                metadata=dict(round=args.round,selected_run=selected['run']))
    restored=load_module(durable/'model',trusted=True).module.eval()
    with torch.inference_mode():
        torch.testing.assert_close(retained(packed['validation'][0][:2]),restored(packed['validation'][0][:2]),atol=0,rtol=0)
    write(durable/'config.json',dict(variant=winner,width=args.width,capacity=capacity,transfer=args.transfer,seed=selected['seed'],source=str(archive/'models.wgt')))
    results=dict(winner_by_validation=winner,selected_run=selected['run'],runs=trained)
    write(archive/'results.json',results)
    lines=['# '+args.round+' — complete ELF generation','',
           '| Architecture | Validation macro exact | Test macro exact (mean) | Parameters | Training seconds (mean) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for v in args.variants:
        subset=[r for r in trained if r['variant']==v]
        lines.append(f"| {v} | {aggregate[v]['macro_exact']:.3%} | {statistics.mean(r['test']['macro_exact'] for r in subset):.3%} | {subset[0]['parameters']} | {statistics.mean(r['train_seconds'] for r in subset):.2f} |")
    lines += ['',f'Validation-selected candidate: **{winner}**. Selected run: `{selected["run"]}`.',
              '', '## Lessons', '',
              '- Full-file exact match and EOS determine success; fixed header bytes must not conceal request-byte failures.',
              '- Equal examples and optimizer steps do not imply equal compute. Parameter counts and measured CPU time are reported.',
              '- These test questions are now exposed. Future changes must select on validation; do not tune against this test.',
              '- Two assembler families cannot establish general binary understanding, editing or arbitrary generation.',
              '- Checkpoints of losing candidates were removed; code, questions, schedules hashes and per-case outcomes are retained.',
              '', '## Next experiment', '',
              'Investigate request-byte transfer and family balance; freeze any change and fresh questions before running it.', '']
    (archive/'REPORT.md').write_text('\n'.join(lines))
    shutil.rmtree(cache)
    state(phase='round_complete',round=args.round,incumbent=dict(variant=winner,run=selected['run'],
          weights=str(durable.relative_to(ROOT)),report=str((archive/'REPORT.md').relative_to(ROOT))),
          next_action='inspect family errors, freeze next hypothesis and new questions; application promotion pending broader tasks')
    print('ARCHIVED '+str(archive),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status_path = ROOT/'experiments/STATUS.json'
        if status_path.exists():
            current=json.loads(status_path.read_text())
            current.update(phase='failed', failure=str(error), next_action='fix failure before starting a new round')
            write(status_path,current)
        raise
