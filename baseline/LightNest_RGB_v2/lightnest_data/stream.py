from __future__ import annotations
import gzip,json,random
from pathlib import Path
from typing import Iterator
from .core import canonical,digest,file_sha,SPLITS

INPUT_KEYS={'schema_version','record_id','architecture','abi','request','program_hex'}

def model_input(row: dict) -> dict:
    """The only object to pass to the inherited neural encoder."""
    if set(row) != INPUT_KEYS: raise ValueError('Model input contains unexpected fields; labels/metadata must be separate')
    return {'command':row['request'],'binary':row['program_hex']}

def legacy_training_row(inp: dict, label: dict) -> dict:
    if inp['record_id'] != label['record_id']: raise ValueError('Input/label mismatch')
    return model_input(inp) | {'edits':[{'offset':e['offset'],'value':e['after']} for e in label['edits']]}

class DatasetStream:
    """Bounded sequential loader with deterministic shard order and resumable cursor.
    Whole groups stay in one shard and on one rank/worker. The cursor denotes
    the NEXT record. Save it only after the consuming optimizer update commits.
    A different world/worker layout or shuffle seed is rejected on resume.
    """
    def __init__(self, root: str|Path, split='train', *, training=True, rank=0, world_size=1,
                 worker_id=0, num_workers=1, epoch=0, seed=0, verify_hashes=True):
        self.root=Path(root); self.manifest=json.loads((self.root/'manifest.json').read_text())
        if split not in SPLITS: raise ValueError('Unknown split')
        if training and split!='train': raise ValueError('Training may read only train; evaluation requires explicit training=False')
        if split=='quarantine': raise ValueError('Quarantine is never loadable for training/evaluation')
        if not (0<=rank<world_size and 0<=worker_id<num_workers): raise ValueError('Invalid rank/worker')
        self.params=dict(split=split,training=training,rank=rank,world_size=world_size,worker_id=worker_id,num_workers=num_workers,epoch=epoch,seed=seed)
        self.identity=digest({'content_id':self.manifest['content_id'],'params':self.params})
        shards=[s for s in self.manifest['shards'] if s['split']==split]
        count=world_size*num_workers; index=rank*num_workers+worker_id
        self.shards=shards[index::count]
        if training: random.Random(seed+epoch).shuffle(self.shards)
        self.shard_pos=0;self.row_pos=0
        self.verify_hashes=verify_hashes;self._running=False
    def state_dict(self):
        return {'version':1,'identity':self.identity,'shard_pos':self.shard_pos,'row_pos':self.row_pos}
    def load_state_dict(self,state):
        if self._running: raise RuntimeError('Close current iterator before restoring')
        if state.get('version')!=1 or state.get('identity')!=self.identity: raise ValueError('Dataset/seed/topology changed')
        s,r=state['shard_pos'],state['row_pos']
        if type(s)is not int or type(r)is not int or not 0<=s<=len(self.shards) or r<0: raise ValueError('Invalid cursor')
        if (s==len(self.shards) and r!=0) or (s<len(self.shards) and r>self.shards[s]['rows']): raise ValueError('Invalid row cursor')
        self.shard_pos,self.row_pos=s,r
    def __iter__(self) -> Iterator:
        if self._running: raise RuntimeError('Only one active iterator per stream')
        self._running=True
        try:
            while self.shard_pos<len(self.shards):
                s=self.shards[self.shard_pos]
                keys=('inputs','labels') if self.params['training'] else ('inputs',)
                for k in keys:
                    if not (self.root/s[k]['path']).resolve().is_relative_to(self.root.resolve()):raise ValueError('Shard path escapes dataset')
                if self.verify_hashes:
                    for k in keys:
                        if file_sha(self.root/s[k]['path'])!=s[k]['sha256']: raise ValueError('Shard checksum failed')
                with gzip.open(self.root/s['inputs']['path'],'rt',encoding='utf-8') as a:
                    b=gzip.open(self.root/s['labels']['path'],'rt',encoding='utf-8') if self.params['training'] else None
                    try:
                        for i,line in enumerate(a):
                            other=b.readline() if b else None
                            if b and not other: raise ValueError('Truncated labels')
                            if i<self.row_pos:continue
                            inp=json.loads(line); model_input(inp)
                            if b:
                                lab=json.loads(other)
                                if inp['record_id']!=lab['record_id']:raise ValueError('Mismatched label order')
                            self.row_pos=i+1
                            yield (inp,lab) if b else inp
                        if b and b.readline():raise ValueError('Extra labels')
                    finally:
                        if b:b.close()
                self.shard_pos+=1;self.row_pos=0
        finally:self._running=False
    def batches(self, max_records=16, max_bytes=4096):
        """Checkpoint after each yielded batch; one-item lookahead is not consumed.
        This is a data cursor only. Save model, optimizer and RNG in the same
        checkpoint. DataLoader prefetch needs per-worker acknowledgement too.
        """
        if max_records<1 or max_bytes<4:raise ValueError('Invalid batch budget')
        it=iter(self);batch=[];total=0
        try:
            while True:
                before=self.state_dict()
                try:row=next(it)
                except StopIteration:break
                inp=row[0] if self.params['training'] else row
                size=len(inp['program_hex'])//2
                if size>max_bytes:raise ValueError('A single sample exceeds byte budget')
                if batch and (len(batch)>=max_records or total+size>max_bytes):
                    after=self.state_dict()
                    self.shard_pos,self.row_pos=before['shard_pos'],before['row_pos']
                    yield batch
                    self.shard_pos,self.row_pos=after['shard_pos'],after['row_pos']
                    batch=[];total=0
                batch.append(row);total+=size
            if batch:yield batch
        finally:
            it.close()
