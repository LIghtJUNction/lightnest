"""Native x86-64 compiler corpus plus exact translations of historical TRAIN tasks."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile

from elftools.elf.elffile import ELFFile

from .sandbox import Sandbox
from .targets import decode, host_architecture

ROOT=Path(__file__).resolve().parents[1]
MASK=2**64-1
FAMILIES=('add_sub','xor_or','and_or','left_right','rotate','minimum_maximum','constant','compare')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2,allow_nan=False);file.write('\n')


def jsonl(path,rows):
    with Path(path).open('x') as file:
        for row in rows:file.write(json.dumps(row,separators=(',',':'))+'\n')


def read(path):
    with Path(path).open() as file:return [json.loads(line) for line in file]


def outside(path):
    path=Path(path).resolve()
    if path==ROOT or ROOT in path.parents:raise ValueError("generated data and models belong outside Git")
    return path


def extract_object(raw):
    elf=ELFFile(io.BytesIO(raw));symbols=elf.get_section_by_name('.symtab')
    f=next(s for s in symbols.iter_symbols() if s.name=='f')
    section=elf.get_section(f.entry.st_shndx)
    for sec in elf.iter_sections():
        if sec.header.sh_type in ('SHT_RELA','SHT_REL') and sec.header.sh_info==f.entry.st_shndx and sec.num_relocations():
            raise ValueError("standalone function contains unresolved relocations")
    offset=f.entry.st_value-section.header.sh_addr
    code=section.data()[offset:offset+f.entry.st_size]
    decode('x86_64',code)
    return code


def compile_c(source):
    with tempfile.TemporaryDirectory(prefix='lightnest-cc-') as temp:
        path=Path(temp)/'function.o'
        result=subprocess.run(['clang','-x','c','-O2','-march=x86-64-v3','-fno-stack-protector','-fno-asynchronous-unwind-tables',
                               '-fno-pic','-fno-pie','-c','-o',str(path),'-'],input=source.encode(),capture_output=True,timeout=20)
        if result.returncode:raise ValueError(result.stderr.decode()[-2000:])
        return extract_object(path.read_bytes())


def compile_asm(source):
    with tempfile.TemporaryDirectory(prefix='lightnest-mc-') as temp:
        path=Path(temp)/'function.o'
        result=subprocess.run(['llvm-mc','--triple=x86_64','--filetype=obj','-o',str(path),'-'],input=source.encode(),capture_output=True,timeout=20)
        if result.returncode:raise ValueError(result.stderr.decode()[-2000:])
        return extract_object(path.read_bytes())


def sx(value,bits):return (value+2**(bits-1))%2**bits-2**(bits-1)


def translate_rv32(code):
    """Lower the documented historical RV32 ABI into x86-64 stack/register operations.

    Source and target are translated independently; matching lengths are checked
    afterward. This is a corpus construction tool, never the neural predictor.
    """
    words=[int.from_bytes(code[i:i+4],'little') for i in range(0,len(code),4)]
    lines=['.intel_syntax noprefix','.text','.globl f','.type f,@function','f:',
           'mov r8,rdi','mov r9,rsi','mov r10,rdx','sub rsp,512','mov rdi,rsp','xor eax,eax','mov ecx,128','cld','rep stosd',
           'mov DWORD PTR [rsp+128],r8d','mov DWORD PTR [rsp+132],r9d','mov DWORD PTR [rsp+136],r10d',
           'mov DWORD PTR [rsp+48],3']
    for i,w in enumerate(words):
        lines.append(f'i{i}:');op=w&127;d=w>>7&31;f=w>>12&7;a=w>>15&31;b=w>>20&31;hi=w>>25
        if w==0x00100073:
            lines+=['mov eax,DWORD PTR [rsp+40]','add rsp,512','ret'];continue
        if op==0x33:
            lines += [f'mov eax,DWORD PTR [rsp+{4*a}]',f'mov ecx,DWORD PTR [rsp+{4*b}]']
            if hi==1 and f==0:lines.append('imul eax,ecx')
            elif hi==32 and f==0:lines.append('sub eax,ecx')
            elif hi in (0,32):
                if f in (0,4,6,7):lines.append({0:'add',4:'xor',6:'or',7:'and'}[f]+' eax,ecx')
                elif f in (1,5):lines.append(('shl' if f==1 else 'sar' if hi else 'shr')+' eax,cl')
                elif f in (2,3):lines+=['cmp eax,ecx',('setl' if f==2 else 'setb')+' al','movzx eax,al']
                else:raise ValueError('unsupported historical R instruction')
            else:raise ValueError('unsupported historical R instruction')
            if d:lines.append(f'mov DWORD PTR [rsp+{4*d}],eax')
        elif op==0x13:
            lines.append(f'mov eax,DWORD PTR [rsp+{4*a}]')
            if f==0:lines.append(f'add eax,{sx(w>>20,12)}')
            elif f in (1,5):lines.append(('shl' if f==1 else 'sar' if hi else 'shr')+f' eax,{b}')
            else:raise ValueError('unsupported historical immediate')
            if d:lines.append(f'mov DWORD PTR [rsp+{4*d}],eax')
        elif op==3 and f==2 and a==0:
            addr=sx(w>>20,12)
            if not 0<=addr<256 or addr%4:raise ValueError('historical load bounds')
            if d:lines += [f'mov eax,DWORD PTR [rsp+{128+addr}]',f'mov DWORD PTR [rsp+{4*d}],eax']
        elif op==0x23 and f==2 and a==0:
            addr=sx((w>>25)<<5 | (w>>7&31),12)
            if not 64<=addr<256 or addr%4:raise ValueError('historical store bounds')
            lines += [f'mov eax,DWORD PTR [rsp+{4*b}]',f'mov DWORD PTR [rsp+{128+addr}],eax']
        elif op==0x63 and f in (0,1,4):
            off=sx((w>>31)<<12|(w>>7&1)<<11|(w>>25&63)<<5|(w>>8&15)<<1,13)
            lines += [f'mov eax,DWORD PTR [rsp+{4*a}]',f'cmp eax,DWORD PTR [rsp+{4*b}]',
                      {0:'je',1:'jne',4:'jl'}[f]+f' i{i+off//4}']
        elif op==0x6f and d==0:
            off=sx((w>>31)<<20|(w>>12&255)<<12|(w>>20&1)<<11|(w>>21&1023)<<1,21)
            lines.append(f'jmp i{i+off//4}')
        else:raise ValueError('unsupported historical opcode')
    lines+=['.size f,.-f']
    return compile_asm('\n'.join(lines)+'\n')


def specification(family,seed,changed):
    rg=random.Random(seed);k=rg.randint(129,100000);shift=rg.randint(1,31)
    if family=='add_sub':expr=f'((a {"-" if changed else "+"} b) ^ {k}ULL) + c';command='replace addition of the first two arguments with subtraction'
    elif family=='xor_or':expr=f'((a {"|" if changed else "^"} b) + {k}ULL) ^ c';command='replace XOR of the first two arguments with OR'
    elif family=='and_or':expr=f'((a {"|" if changed else "&"} b) ^ {k}ULL) + c';command='replace AND of the first two arguments with OR'
    elif family=='left_right':expr=f'(a {">>" if changed else "<<"} {shift}) ^ (b + {k}ULL) ^ c';command='replace the left shift of the first argument with a right shift'
    elif family=='rotate':
        expr=(f'((a >> {shift}) | (a << {64-shift}))' if changed else f'((a << {shift}) | (a >> {64-shift}))')+f' ^ b ^ c ^ {k}ULL';command='replace the left rotation of the first argument with a right rotation'
    elif family=='minimum_maximum':expr=f'((a {">" if changed else "<"} b) ? a : b) ^ c ^ {k}ULL';command='replace unsigned minimum of the first two arguments with maximum'
    elif family=='constant':expr=f'(a+b+c) ^ {k+17 if changed else k}ULL';command='increase the final XOR constant by seventeen'
    elif family=='compare':expr=f'(a {">" if changed else "<"} b) + c + {k}ULL';command='replace unsigned less-than of the first two arguments with greater-than'
    else:raise ValueError(family)
    return '#include <stdint.h>\nuint64_t f(uint64_t a,uint64_t b,uint64_t c){return '+expr+';}\n',command,k,shift


def reference(family,seed,inputs):
    _,_,k,s=specification(family,seed,True);a,b,c=inputs
    if family=='add_sub':v=(((a-b)&MASK)^k)+c
    elif family=='xor_or':v=(((a|b)+k)&MASK)^c
    elif family=='and_or':v=((a|b)^k)+c
    elif family=='left_right':v=(a>>s)^((b+k)&MASK)^c
    elif family=='rotate':v=((a>>s)|(a<<(64-s)))^b^c^k
    elif family=='minimum_maximum':v=max(a,b)^c^k
    elif family=='constant':v=((a+b+c)&MASK)^(k+17)
    elif family=='compare':v=int(a>b)+c+k
    else:raise ValueError(family)
    return v&MASK


def group(family,seed,split,sandbox):
    original,command,_,_=specification(family,seed,False)
    target,_,_,_=specification(family,seed,True)
    before,after=compile_c(original),compile_c(target)
    capacity=max(len(before),len(after))
    before=before.ljust(capacity,b'\x90');after=after.ljust(capacity,b'\x90')
    if not 1<=sum(a!=b for a,b in zip(before,after))<=64:
        return None
    rg=random.Random(seed+654321)
    cases=[[0,0,0],[1,2,3],[MASK,1,0],[2**63,2**63-1,7],[7,7,1],[1,7,1],[7,1,1]]
    while len(cases)<40:
        case=[rg.randrange(2**64) for _ in range(3)]
        if case not in cases:cases.append(case)
    expected=[reference(family,seed,x) for x in cases]
    result=sandbox.run(after,cases)
    if result['status']!='ok' or result['outputs']!=expected:
        raise ValueError(f'compiler/Python reference disagree: {family}')
    if sandbox.run(before,cases)['status']!='ok':raise ValueError('original corpus function failed')
    visible=[{'inputs':x,'output':y} for x,y in zip(cases[:8],expected[:8],strict=True)]
    hidden=[{'inputs':x,'output':y} for x,y in zip(cases[8:],expected[8:],strict=True)]
    gid=f'{split}/{family}/{seed}'
    base={'architecture':'x86_64','command':command,'examples':visible,'hidden':hidden,'family':family,
          'group_id':gid,'origin':'clang-generated synthetic uint64 function','reference_seed':seed}
    return [base|{'id':gid+'/edit','binary':before.hex(),'target':after.hex()},
            base|{'id':gid+'/noop','binary':after.hex(),'target':after.hex()}]


def prepare(out,historical_run,groups=32):
    if host_architecture()!='x86_64':raise ValueError('this training corpus currently targets x86-64')
    out=outside(out);out.mkdir(parents=True,exist_ok=False)
    for name in ('data','models','training','evaluation'):(out/name).mkdir()
    sandbox=Sandbox(out/'sandbox-prepare');datasets={};known=set();counts=Counter()
    for split,size,offset in (('train',groups,100000),('validation',4,200000),('test',8,300000)):
        records=[]
        for fi,family in enumerate(FAMILIES):
            got=attempt=0
            while got<size:
                seed=offset+fi*10000+attempt;attempt+=1
                if attempt>size*100:raise RuntimeError('native corpus construction exhausted')
                result=group(family,seed,split,sandbox)
                if result is None:counts['incompatible_length_groups']+=1;continue
                digests={hashlib.sha256(bytes.fromhex(r[k])).hexdigest() for r in result for k in ('binary','target')}
                if digests&known:counts['duplicate_groups']+=1;continue
                known.update(digests);records.extend(result);got+=1
            print('BUILT',split,family,got,'groups',flush=True)
        datasets[split]=records
    # Explicitly reuse only already-designated historical TRAIN records.
    previous=json.loads((Path(historical_run)/'protocol.json').read_text())
    train_path=Path(historical_run)/'data/train.jsonl'
    if sha(train_path)!=previous['data']['train.jsonl']:raise ValueError('historical TRAIN receipt changed')
    historical=read(train_path);rg=random.Random(2026102408);rg.shuffle(historical)
    from archive.legacy.hybrid import BASE
    from rgb.runtime import common
    seen=set();accepted=0
    for row in historical:
        if row['group_id'] in seen or row['binary']==row['target']:continue
        seen.add(row['group_id'])
        raw,target=bytes.fromhex(row['binary']),bytes.fromhex(row['target'])
        try:
            before,after=translate_rv32(raw),translate_rv32(target)
            if len(before)!=len(after) or len(before)>4096 or len(decode('x86_64',before))>768:continue
        except ValueError:
            counts['historical_conversion_rejected']+=1;continue
        if not 1<=sum(a!=b for a,b in zip(before,after))<=16:continue
        inputs=[[0,0,0],[1,2,3],[2**31-1,1,0],[2**31,2**32-1,1]]
        inputs += [[rg.randrange(2**32) for _ in range(3)] for _ in range(12)]
        old_inputs=[[sx(x,32) for x in xs] for xs in inputs]
        for original,converted in ((raw,before),(target,after)):
            expected=[common.execute_a(original,x)&0xffffffff for x in old_inputs]
            independent=[common.execute_b(original,x)&0xffffffff for x in old_inputs]
            native=sandbox.run(converted,inputs)
            if expected!=independent or native['status']!='ok' or native['outputs']!=expected:
                raise ValueError('historical RV32/two-VM/x86 translation mismatch')
        expected=[common.execute_b(target,x)&0xffffffff for x in old_inputs]
        examples=[{'inputs':x,'output':y} for x,y in zip(inputs[:8],expected[:8],strict=True)]
        hidden=[{'inputs':x,'output':y} for x,y in zip(inputs[8:],expected[8:],strict=True)]
        base={'architecture':'x86_64','command':row['command'],'examples':examples,'hidden':hidden,
              'family':'historical_'+row['family'],'group_id':row['group_id'],'origin':'verified historical TRAIN translation',
              'source_record_id':row['id']}
        datasets['train'].extend([base|{'id':'translated/'+row['id']+'/edit','binary':before.hex(),'target':after.hex()},
                                  base|{'id':'translated/'+row['id']+'/noop','binary':after.hex(),'target':after.hex()}])
        accepted+=1
        if accepted==64:break
    if not accepted:raise ValueError('no historical TRAIN data converted')
    for split,rows in datasets.items():jsonl(out/'data'/f'{split}.jsonl',rows)
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'lightnest_native').glob('*')) if p.suffix in ('.py','.c')}
    protocol={'architecture':'lightnest-universal-machine-v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'host_architecture':host_architecture(),'compiler':subprocess.check_output(['clang','--version']).decode().splitlines()[0],
              'compiler_target':'x86-64-v3','historical_train_sha256':sha(train_path),'historical_groups_converted':accepted,
              'counts':{k:len(v) for k,v in datasets.items()},'construction_counts':dict(counts),
              'data':{p.name:sha(p) for p in (out/'data').glob('*.jsonl')},'source':source,'uv_lock_sha256':sha(ROOT/'uv.lock'),
              'training':{'seeds':[104729,130363],'steps':1536,'batch':8,'lr':0.0003,'threshold':0.5,'width':96,'rounds':8},
              'test_model_queries':0,'scope':'synthetic compiled x86-64 bounded code-block rewriting with NOP padding; other ISAs decode only',
              'selection':'fixed endpoints; no final-test checkpoint or threshold selection',
              'acceptance':'exact bytes + STOP + hidden native cases; report false edits separately',
              'boundary':'not a complete OS/CPU emulator, arbitrary ELF repair or correctness proof'}
    write(out/'protocol.json',protocol);print(json.dumps(protocol['counts']),flush=True)
