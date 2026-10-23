"""Train fixed native endpoints; freeze before test; preserve complete decisions."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .corpus import ROOT, outside, read, sha, write
from .model import Batch, Config, MachineNetwork, load, loss_for, save
from .sandbox import Sandbox


def projected(rows):
    return [{k:r[k] for k in ('architecture','command','binary','examples')} for r in rows]


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'archive'/'universal_native').glob('*')) if p.suffix in ('.py','.c')}


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['source']!=sources() or protocol['uv_lock_sha256']!=sha(ROOT/'uv.lock'):
        raise ValueError('source/dependencies changed after protocol freeze; prepare a new run')
    for name,digest in protocol['data'].items():
        if sha(out/'data'/name)!=digest:raise ValueError('frozen data changed')
    return protocol


def initialize_from_rv32(model,path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['architecture']!='lightnest-rpn3-v1':raise ValueError('unexpected pretraining architecture')
    old=payload['state_dict'];current=model.state_dict();copied=[]
    mapping={'command.':'command_embedding.','command_conv.':'command_conv.',
             'before.':'global_before.','after.':'global_after.','message.':'message.',
             'cell.':'update.','norm.':'norm.','readout.':'readout.','edit.':'edit.','bits.':'replacement.'}
    for name,value in current.items():
        for new,original in mapping.items():
            key=original+name[len(new):] if name.startswith(new) else ''
            if key in old and old[key].shape==value.shape:
                value.copy_(old[key]);copied.append(name);break
    if old['byte_embedding.weight'].shape==current['byte.weight'][1:].shape:
        current['byte.weight'][1:].copy_(old['byte_embedding.weight']);copied.append('byte.weight[1:]')
    if not copied:raise ValueError('no compatible pretrained parameters')
    model.load_state_dict(current,strict=True)
    return {'checkpoint_sha256':sha(path),'copied_tensors':copied,
            'note':'shared PyTorch feature/core initialization, then all native parameters trainable; not x86 pretraining evidence'}


@torch.inference_mode()
def diagnostic(model,rows):
    model.eval();total=edit=need=false=0
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8]
        for row,pred in zip(chunk,model.predict(projected(chunk)),strict=True):
            changed=row['binary']!=row['target'];correct=pred['stop'] and pred['final']==row['target']
            total+=int(correct);edit+=int(correct and changed);need+=int(changed)
            false+=int(not changed and bool(pred['patches']))
    return {'rows':len(rows),'exact':total,'edit_exact':edit,'edits':need,'false_edits':false}


def train(out,seed,initialization):
    out=outside(out);protocol=verify(out);settings=protocol['training']
    if seed not in settings['seeds']:raise ValueError('unfrozen seed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    model=MachineNetwork(Config(width=settings['width'],rounds=settings['rounds']))
    transfer=initialize_from_rv32(model,initialization)
    rows=read(out/'data/train.jsonl');validation=read(out/'data/validation.jsonl')
    pools=[[r for r in rows if (r['binary']!=r['target'])==changed] for changed in (False,True)]
    # Long historical translations are interleaved with native compiler tasks.
    native=[[r for r in pool if not r['family'].startswith('historical_')] for pool in pools]
    history=[[r for r in pool if r['family'].startswith('historical_')] for pool in pools]
    plan={'seed':seed,'config':asdict(model.config),'protocol_sha256':sha(out/'protocol.json'),'transfer':transfer,
          'parameters':sum(p.numel() for p in model.parameters()),'source':sources(),
          'environment':{'torch':str(torch.__version__),'threads':1,'device':'cpu'},'test_model_queries':0}
    write(out/'training'/f'{seed}-plan.json',plan)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=0.01)
    rg=random.Random(2026102417);stream=hashlib.sha256();started=time.perf_counter();losses=[];seen=set();validation_history=[]
    with (out/'training'/f'{seed}-updates.jsonl').open('x') as log:
        for step in range(settings['steps']):
            # Seven native minibatches followed by one historical minibatch.
            source=history if step%8==7 else native
            selected=[rg.choice(source[j%2]) for j in range(settings['batch'])];rg.shuffle(selected)
            for row in selected:stream.update((row['id']+'\n').encode());seen.add(row['id'])
            batch=Batch.encode(projected(selected),model.config)
            model.train();optimizer.zero_grad(set_to_none=True)
            z,b=model(batch);loss=loss_for(z,b,batch,selected)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%64==0:
                event={'step':step+1,'mean_loss':sum(losses)/len(losses),'gradient_norm':float(norm),'wall_seconds':time.perf_counter()-started}
                log.write(json.dumps(event)+'\n');log.flush();losses=[];print(seed,json.dumps(event),flush=True)
            if (step+1)%512==0:
                score=diagnostic(model,validation);validation_history.append({'step':step+1,**score});print('VALIDATION',seed,score,flush=True)
    path=out/'models'/f'universal-{seed}.pt'
    metadata={'seed':seed,'updates':settings['steps'],'presentations':settings['steps']*settings['batch'],
              'unique_records':len(seen),'stream_sha256':stream.hexdigest(),'wall_seconds_including_validation':time.perf_counter()-started,
              'protocol_sha256':sha(out/'protocol.json'),'transfer':transfer,'validation_history':validation_history,
              'status':'research_candidate','trained_architectures':['x86_64'],'source':sources()}
    save(path,model.eval(),metadata)
    torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'sampler_rng':rg.getstate(),'updates':settings['steps']},out/'training'/f'{seed}-optimizer.pt')
    write(out/'training'/f'{seed}-complete.json',metadata|{'checkpoint_sha256':sha(path)})
    print('TRAINED',seed,sha(path),flush=True)


def freeze(out):
    out=outside(out);protocol=verify(out);models={}
    streams=set()
    for seed in protocol['training']['seeds']:
        result=json.loads((out/'training'/f'{seed}-complete.json').read_text())
        path=out/'models'/f'universal-{seed}.pt'
        if sha(path)!=result['checkpoint_sha256'] or result['source']!=sources():raise ValueError('endpoint changed')
        models[path.name]=sha(path);streams.add(result['stream_sha256'])
    if len(streams)!=1:raise ValueError('two initialization seeds saw different training streams')
    write(out/'final-freeze.json',{'models':models,'data':protocol['data'],'source':sources(),
          'protocol_sha256':sha(out/'protocol.json'),'test_model_queries':0})


def evaluate(out,seed):
    out=outside(out);verify(out);frozen=json.loads((out/'final-freeze.json').read_text())
    path=out/'models'/f'universal-{seed}.pt'
    if frozen['source']!=sources() or frozen['models'][path.name]!=sha(path):raise ValueError('frozen endpoint changed')
    torch.set_num_threads(1);model=load(path);sandbox=Sandbox(out/f'sandbox-evaluation-{seed}')
    report={'seed':seed,'checkpoint_sha256':sha(path),'splits':{},'trained_architectures':['x86_64']}
    for split in ('validation','test'):
        rows=read(out/'data'/f'{split}.jsonl');predictions=[];started=time.perf_counter()
        for start in range(0,len(rows),8):predictions.extend(model.predict(projected(rows[start:start+8])))
        inference=time.perf_counter()-started;records=[]
        for row,pred in zip(rows,predictions,strict=True):
            case_inputs=[e['inputs'] for e in row['hidden']];expected=[e['output'] for e in row['hidden']]
            result=sandbox.run(bytes.fromhex(pred['final']),case_inputs)
            behavior=result['status']=='ok' and result['outputs']==expected
            need=row['binary']!=row['target']
            records.append({'id':row['id'],'group_id':row['group_id'],'family':row['family'],'need_edit':need,
                 'strict':pred['stop'] and pred['final']==row['target'] and behavior and (need or not pred['patches']),
                 'false_edit':not need and bool(pred['patches']),'hidden_behavior':behavior,'sandbox_status':result['status'],
                 'initial_sha256':hashlib.sha256(bytes.fromhex(row['binary'])).hexdigest(),'prediction':pred})
        summary={'rows':len(rows),'edits':sum(r['need_edit'] for r in records),
                 'strict_edits':sum(r['strict'] and r['need_edit'] for r in records),'strict_total':sum(r['strict'] for r in records),
                 'false_edits':sum(r['false_edit'] for r in records),'hidden_behavior_pass':sum(r['hidden_behavior'] for r in records),
                 'inference_wall_seconds':inference,'families':{}}
        for family in sorted({r['family'] for r in records}):
            part=[r for r in records if r['family']==family]
            summary['families'][family]={'rows':len(part),'strict':sum(r['strict'] for r in part),'false_edits':sum(r['false_edit'] for r in part)}
        write(out/'evaluation'/f'{seed}-{split}.json',{'summary':summary,'records':records,'checkpoint_sha256':sha(path)})
        report['splits'][split]=summary;print('EVALUATED',seed,split,json.dumps(summary),flush=True)
    report['promoted']=False
    report['promotion_note']='No universal strongest claim: other CPU architectures are untrained; synthetic family generalization only.'
    write(out/'evaluation'/f'{seed}-report.json',report)
    return report
