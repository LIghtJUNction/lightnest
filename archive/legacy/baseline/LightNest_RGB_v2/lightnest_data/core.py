from __future__ import annotations
import gzip, hashlib, json, os, sqlite3
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
from typing import Any, Iterator

SCHEMA_VERSION = '1.0.0'
ARCH = 'rv32im_restricted'
ABI = 'lightnest-rv32-scratch-v1'
SPLITS = ('train', 'validation_public', 'regression_public', 'test_reserved', 'quarantine')

def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

@contextmanager
def gz_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as g:
            yield g

def read_rows(path: Path, layout: str = 'jsonl') -> Iterator[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        if layout == 'json':
            # Legacy arrays are a bounded migration exception; new imports use JSONL.
            if path.stat().st_size > 64 * 1024 * 1024: raise ValueError('Convert a large legacy array to JSONL first')
            rows = json.load(f)
            if not isinstance(rows, list): raise ValueError('Expected a legacy JSON array')
            yield from rows
        else:
            for n, line in enumerate(f, 1):
                if len(line) > 1024 * 1024: raise ValueError(f'oversized row {path}:{n}')
                if not line.strip(): raise ValueError(f'blank row {path}:{n}')
                item = json.loads(line)
                if not isinstance(item, dict): raise ValueError(f'Expected object {path}:{n}')
                yield item

def decode(hex_text: str) -> bytes:
    if not isinstance(hex_text, str) or not hex_text or len(hex_text) % 2: raise ValueError('Invalid hex string')
    raw = bytes.fromhex(hex_text)
    if raw.hex() != hex_text.lower(): raise ValueError('Hex must not contain whitespace')
    if not 4 <= len(raw) <= 1024 or len(raw) % 4: raise ValueError('Code must be aligned and 4..1024 bytes for this ABI')
    return raw

def opcode_name(w: int) -> str:
    op, f, h = w & 127, (w >> 12) & 7, w >> 25
    if w == 0x00100073: return 'EBREAK'
    if op == 0x33:
        if h == 0: return ('ADD','SLL','SLT','SLTU','XOR','SRL','OR','AND')[f]
        if h == 32 and f in (0,5): return 'SUB' if f == 0 else 'SRA'
        if h == 1 and f == 0: return 'MUL'
    if op == 0x13:
        if f == 0: return 'ADDI'
        if f == 1 and h == 0: return 'SLLI'
        if f == 5 and h in (0,32): return 'SRAI' if h else 'SRLI'
    if op == 3 and f == 2: return 'LW'
    if op == 0x23 and f == 2: return 'SW'
    if op == 0x63 and f in (0,1,4): return {0:'BEQ',1:'BNE',4:'BLT'}[f]
    if op == 0x6f and ((w >> 7) & 31) == 0: return 'JAL'
    raise ValueError(f'Unsupported instruction 0x{w:08x}')

def validate_program(raw: bytes) -> None:
    """Validate every instruction, including unexecuted paths, against the ABI."""
    n=len(raw)//4
    for i in range(n):
        w=int.from_bytes(raw[4*i:4*i+4],'little');name=opcode_name(w);a=(w>>15)&31
        if name=='EBREAK' and i!=n-1:raise ValueError('Early EBREAK is not supported')
        if name=='LW':
            addr=(w>>20);addr=addr-4096 if addr>=2048 else addr
            if a!=0 or not 0<=addr<256 or addr%4:raise ValueError('Load outside absolute data words')
        if name=='SW':
            addr=((w>>25)<<5)|((w>>7)&31);addr=addr-4096 if addr>=2048 else addr
            if a!=0 or not 64<=addr<256 or addr%4:raise ValueError('Store outside absolute scratch words')
        if name in ('BEQ','BNE','BLT','JAL'):
            if name=='JAL':
                off=((w>>31)<<20)|(((w>>12)&255)<<12)|(((w>>20)&1)<<11)|(((w>>21)&1023)<<1);bits=21
            else:
                off=((w>>31)<<12)|(((w>>7)&1)<<11)|(((w>>25)&63)<<5)|(((w>>8)&15)<<1);bits=13
            if off>=1<<(bits-1):off-=1<<bits
            if off%4 or not 0<=i+off//4<n:raise ValueError('Branch or jump leaves instruction bounds')
    if int.from_bytes(raw[-4:],'little')!=0x00100073:raise ValueError('Missing final EBREAK')

def neutral_fingerprint(raw: bytes) -> str:
    """A conservative leakage guard, not a proof of semantic equivalence.
    Neutralize ADD/SUB choice and last ADDI return selector. Keep registers,
    immediates, padding and every other instruction unchanged.
    """
    b = bytearray(raw)
    for i in range(0, len(b), 4):
        w = int.from_bytes(b[i:i+4], 'little')
        if opcode_name(w) in ('ADD','SUB'): b[i+3] &= ~64
    if len(b) >= 8:
        w = int.from_bytes(b[-8:-4], 'little')
        if w & 0x707f == 0x13 and (w >> 7) & 31 == 10 and w >> 20 == 0:
            b[-8:-4] = (w & ~(31 << 15)).to_bytes(4, 'little')
    return hashlib.sha256(b).hexdigest()

class DiskUnion:
    """Disk-backed component grouping. No full corpus in Python memory."""
    def __init__(self, db: sqlite3.Connection): self.db = db
    def find(self, key: str) -> str:
        self.db.execute('INSERT OR IGNORE INTO uf VALUES (?,?)', (key,key))
        path = []; cur = key
        while True:
            p = self.db.execute('SELECT parent FROM uf WHERE key=?', (cur,)).fetchone()[0]
            if p == cur: break
            path.append(cur); cur = p
        self.db.executemany('UPDATE uf SET parent=? WHERE key=?', [(cur,k) for k in path])
        return cur
    def union(self, a: str, b: str) -> None:
        a,b = self.find(a),self.find(b)
        if a != b:
            lo,hi = sorted((a,b)); self.db.execute('UPDATE uf SET parent=? WHERE key=?', (lo,hi))

DIFFICULTY = {'mixed':1,'overwrite':1,'spill':2,'branch':3,'loop':3,'branch_spill':4,'loop_spill':4,'novel_mixed':2}

def normalize(row: dict, source: dict, row_number: int, default_inputs: list) -> tuple[dict,dict,dict,dict]:
    binary, target = decode(row['binary']), decode(row['target'])
    validate_program(binary);validate_program(target)
    if len(binary) != len(target): raise ValueError('Only fixed-length byte substitutions are supported in v1')
    command = row['command']
    if not isinstance(command,str) or not 1 <= len(command.encode('utf-8')) <= 256: raise ValueError('Bad request length')
    edits = [{'offset':i, 'before':a, 'after':b} for i,(a,b) in enumerate(zip(binary,target)) if a!=b]
    given = sorted((e['offset'],e.get('value',e.get('after'))) for e in row['edits'])
    if any(type(p) is not int or type(v) is not int for p,v in given): raise ValueError('Edit must use integer byte indices and values')
    if given != [(e['offset'],e['after']) for e in edits]: raise ValueError('Edits do not exactly reconstruct target')
    if len(edits) > 8: raise ValueError('More than 8 edits exceeds current task contract')
    for e in edits:
        off=e['offset']; i=off//4*4
        names=[opcode_name(int.from_bytes(b[i:i+4],'little')) for b in (binary,target)]
        if set(names) != {'ADD','SUB'} or off%4 != 3 or e['before']^e['after'] != 64: raise ValueError('Unexpected mutation in dependency-transform task')
    for b in (binary,target):
        ops=[opcode_name(int.from_bytes(b[i:i+4],'little')) for i in range(0,len(b),4)]
        if ops[-1] != 'EBREAK': raise ValueError('Missing final EBREAK')
    payload={'architecture':ARCH,'abi':ABI,'request':command,'program_hex':binary.hex()}
    input_id=digest(payload)
    record_id=digest({'input_id':input_id,'target_hex':target.hex(),'task':'dependency_transform'})
    inp={'schema_version':SCHEMA_VERSION,'record_id':record_id, **payload}
    inpvecs=[x for x,_ in row['cases']] if row.get('cases') else default_inputs
    if not inpvecs or any(not 1<=len(x)<=16 or any(type(v) is not int or not -2**31<=v<2**31 for v in x) for x in inpvecs): raise ValueError('Invalid execution input')
    suite_id=digest({'inputs':inpvecs,'abi':ABI,'fuel':1800})
    label={'schema_version':SCHEMA_VERSION,'record_id':record_id,'target_hex':target.hex(),'action':'EDIT' if edits else 'STOP','edits':edits,'edit_order':'any_remaining','must_stop':True}
    ns=source['namespace']
    raw_group=row.get('construction_block_id') or row.get('origin_id') or row.get('origin_pair_id') or row.get('pair_id') or row.get('episode_id') or row.get('parent_id') or row['id']
    group=digest([ns,raw_group])
    family=row.get('family',row.get('category','unspecified'))
    member={'member_id':digest([source['name'],row_number]),'source_name':source['name'],'source_row':row_number,'source_id':row['id'], 'record_id':record_id,'group_token':group,'origin_key':str(raw_group),'namespace':ns,'original_split':source['split'],'role':source['kind'],'family':family,
            'episode_id':row.get('parent_id') or row.get('episode_id') or row['id'],
            'pair_id':(ns+':'+str(row['pair_id'])) if row.get('pair_id') else None,
            'pair_state_key':(ns+':'+str(row['pair_state_key'])) if row.get('pair_state_key') else None,
            'invariance_pair_id':row.get('invariance_state_key') or row.get('invariance_pair_id'),
            'return_variant':row.get('lane',row.get('variant')),'nuisance_variant':row.get('nuisance_variant'),
            'state_mask':row.get('state_mask'), 'case_suite_id':suite_id,
            'label_origin':'source_generator_or_oracle_not_model_prediction'}
    supplied=[y for _,y in row['cases']] if row.get('cases') else None
    aux={'input_id':input_id,'binary_sha256':hashlib.sha256(binary).hexdigest(),'target_sha256':hashlib.sha256(target).hexdigest(),'neutral_sha256':neutral_fingerprint(binary),'group':group,'inputs':inpvecs,'suite_id':suite_id,'supplied_expected':supplied,'opcodes':sorted(set(opcode_name(int.from_bytes(binary[i:i+4],'little')) for i in range(0,len(binary),4))),'byte_length':len(binary)}
    return inp,label,member,aux
