"""Native bounded interpreter and separately written Python RV32IM checker.

Only the interpreter and trusted reference C kernels are compiled for the host.
Mutated RV32 code is never loaded as a native executable/shared library.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import shutil
import struct
import subprocess
from pathlib import Path

CODE, DATA, OUT, STACK, STACK_BYTES, RETURN_PC = 0x1000, 0x20000, 0x21000, 0x30000, 16384, 0x800
MASK, FUEL, MAX_CODE = 0xffffffff, 30000, 16384

class Input(ct.Structure):
    _fields_ = [('a',ct.c_uint32*32),('n',ct.c_uint32),('key',ct.c_uint32),('alias',ct.c_uint32)]

    @classmethod
    def make(cls, a, n, key, alias=0):
        if len(a)!=32 or not 0<=n<=32 or not 0<=alias<32:
            raise ValueError('Input contract: 32 words, 0<=n<=32, 0<=alias<32')
        return cls((ct.c_uint32*32)(*(int(x)&MASK for x in a)), n, key&MASK, alias)

    def record(self):
        return {'a':list(self.a),'n':int(self.n),'key':int(self.key),'alias':int(self.alias)}

class Output(ct.Structure):
    _fields_ = [('value',ct.c_uint32),('a',ct.c_uint32*32),('out',ct.c_uint32*32),
                ('status',ct.c_uint32),('steps',ct.c_uint32)]

    def observable(self):
        return (int(self.status),int(self.value),tuple(self.a),tuple(self.out))

    def record(self):
        return {'status':int(self.status),'value':int(self.value),'a':list(self.a),
                'out':list(self.out),'steps':int(self.steps)}


def run_command(args: list[str], timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr[-4000:]}')
    return result.stdout


def compile_kernel(source: str, directory: Path, optimization: str) -> tuple[bytes,int,Path]:
    """Compile identical trusted source to RV32IM ELF and a native reference."""
    if optimization not in ('O0','O2','Os'):
        raise ValueError('Unsupported optimization')
    directory.mkdir(parents=True, exist_ok=False)
    c = directory/'reference.c'; c.write_text(source)
    link = directory/'layout.ld'
    link.write_text('ENTRY(entry)\nSECTIONS { . = 0x1000; .text : { *(.text .text.*) } '
                    '.rodata : { *(.rodata .rodata.*) } .data : { *(.data .data.*) } '
                    '.bss : { *(.bss .bss.*) } /DISCARD/ : { *(.comment) *(.eh_frame*) } }\n')
    elf, native = directory/'reference.elf', directory/'reference.so'
    args = ['clang','--target=riscv32-unknown-elf','-march=rv32im','-mabi=ilp32',
            '-'+optimization,'-ffreestanding','-fno-builtin','-fno-stack-protector',
            '-fno-pic','-mno-relax','-nostdlib','-fuse-ld=lld','-Wl,--no-relax',
            '-Wl,-T,'+str(link),str(c),'-o',str(elf)]
    run_command(args)
    run_command(['gcc','-std=c11','-O2','-fPIC','-shared','-fno-builtin',str(c),'-o',str(native)])
    image,entry = load_elf(elf.read_bytes())
    (directory/'build-command.txt').write_text(' '.join(args)+'\n')
    (directory/'reference.rv32').write_bytes(image)
    return image,entry,native


def load_elf(raw: bytes) -> tuple[bytes,int]:
    """Strict ELF32 little-endian RISC-V allocated-section loader."""
    if len(raw)<52 or raw[:7]!=b'\x7fELF\x01\x01\x01':
        raise ValueError('Expected little-endian ELF32')
    header = struct.unpack_from('<HHIIIIIHHHHHH',raw,16)
    typ,machine,version,entry,_,shoff,flags,_,_,_,shsize,shnum,_ = header
    if typ!=2 or machine!=243 or version!=1 or flags!=0 or shsize!=40 or shnum>1024:
        raise ValueError('Unsupported ELF type, ISA flags, or section layout')
    sections=[];top=CODE
    for i in range(shnum):
        if shoff+(i+1)*40>len(raw):raise ValueError('Truncated section table')
        _,kind,sflags,addr,off,size,*_ = struct.unpack_from('<10I',raw,shoff+i*40)
        if not sflags&2 or size==0:continue
        if addr<CODE or addr+size>CODE+MAX_CODE or kind not in (1,8):
            raise ValueError('Allocated section outside declared image')
        if kind!=8 and off+size>len(raw):raise ValueError('Truncated section payload')
        if any(addr<b+s and b<addr+size for b,s,_ in sections):raise ValueError('Overlapping section')
        sections.append((addr,size,bytes(size) if kind==8 else raw[off:off+size]));top=max(top,addr+size)
    if not CODE<=entry<top or entry%4:raise ValueError('Bad entry point')
    image=bytearray((top-CODE+3)//4*4)
    for addr,size,payload in sections:image[addr-CODE:addr-CODE+size]=payload
    return bytes(image),entry


class NativeVM:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        src=Path(__file__).with_name('rv32.c');libpath=directory/'rv32-interpreter.so'
        run_command(['gcc','-std=c11','-O2','-Wall','-Wextra','-fPIC','-shared',str(src),'-o',str(libpath)])
        self.lib=ct.CDLL(str(libpath));self.work=(ct.c_uint64*2)(0,0)
        p=ct.POINTER
        self.lib.vm_run.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,p(Input),p(Output),ct.c_void_p]
        self.lib.vm_run.restype=ct.c_int
        self.lib.vm_check.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,p(Input),p(Output),ct.c_uint32,p(ct.c_uint64)]
        self.lib.vm_check.restype=ct.c_int
        self.lib.vm_batch.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,p(Input),p(Output),ct.c_uint32]
        self.lib.vm_batch.restype=None

    def run(self, code: bytes, entry: int, inp: Input, trace=False):
        out=Output();coverage=(ct.c_uint8*(len(code)//4))() if trace else None
        self.lib.vm_run(code,len(code),entry,ct.byref(inp),ct.byref(out),coverage)
        self.work[0]+=1;self.work[1]+=out.steps
        return out, {i for i,v in enumerate(coverage or ()) if v}

    def batch(self, code: bytes, entry: int, cases):
        inputs=(Input*len(cases))(*cases);outputs=(Output*len(cases))()
        self.lib.vm_batch(code,len(code),entry,inputs,outputs,len(cases))
        self.work[0]+=len(cases);self.work[1]+=sum(o.steps for o in outputs)
        return list(outputs)

    def check(self, code: bytes, entry: int, packed):
        inputs,outputs=packed
        return int(self.lib.vm_check(code,len(code),entry,inputs,outputs,len(inputs),self.work))


def pack(cases, expected):
    if len(cases)!=len(expected):raise ValueError('Mismatched test lengths')
    return (Input*len(cases))(*cases),(Output*len(expected))(*expected)


def reference(path: Path, cases):
    lib=ct.CDLL(str(path));fn=lib.entry;fn.argtypes=[ct.POINTER(ct.c_uint32),ct.c_uint32,
        ct.c_uint32,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint32)];fn.restype=ct.c_uint32
    outputs=[]
    for inp in cases:
        a=(ct.c_uint32*32)(*inp.a);out=(ct.c_uint32*32)()
        b=ct.cast(ct.byref(a,inp.alias*4),ct.POINTER(ct.c_uint32))
        value=fn(a,inp.n,inp.key,b,out)
        outputs.append(Output(value,a,out,0,0))
    return outputs


def signed(x,bits=32):
    return (x+(1<<(bits-1)))%(1<<bits)-(1<<(bits-1))


def python_run(code: bytes, entry: int, inp: Input) -> Output:
    """Independent decoder/execution loop; does not invoke native VM helpers."""
    memory={DATA+i:b for i,b in enumerate(struct.pack('<32I',*inp.a))}
    regs=[0]*32;regs[1]=RETURN_PC;regs[2]=STACK+STACK_BYTES-16
    regs[10:15]=[DATA,int(inp.n),int(inp.key),DATA+4*int(inp.alias),OUT]
    pc=entry;status=0;steps=0
    def access(addr,width,store=None):
        if width not in (1,2,4) or addr%width:raise MemoryError
        legal=any(base<=addr and addr+width<=base+size for base,size in
                  ((DATA,128),(OUT,128),(STACK,STACK_BYTES)))
        if not legal:
            if store is None and CODE<=addr and addr+width<=CODE+len(code):
                return int.from_bytes(code[addr-CODE:addr-CODE+width],'little')
            raise MemoryError
        if store is None:return sum(memory.get(addr+i,0)<<(8*i) for i in range(width))
        for i in range(width):memory[addr+i]=(store>>(8*i))&255
    if not code or len(code)>MAX_CODE or len(code)%4 or not CODE<=entry<CODE+len(code) or inp.n>32 or inp.alias>=32:
        return Output(0,(ct.c_uint32*32)(),(ct.c_uint32*32)(),5,0)
    for steps in range(FUEL):
        if pc==RETURN_PC:break
        if pc%4 or not CODE<=pc<CODE+len(code):status=1;break
        word=int.from_bytes(code[pc-CODE:pc-CODE+4],'little')
        opcode=word&127;rd=(word//128)%32;f=(word//4096)%8
        rs=(word//32768)%32;rt=(word//1048576)%32;hi=word>>25
        x,y=regs[rs],regs[rt];next_pc=(pc+4)&MASK;value=None;im=signed(word>>20,12)
        try:
            if opcode==55:value=word&0xfffff000
            elif opcode==23:value=pc+(word&0xfffff000)
            elif opcode==111:
                off=((word>>31)<<20)|((word>>12&255)<<12)|((word>>20&1)<<11)|((word>>21&1023)<<1)
                value=next_pc;next_pc=(pc+signed(off,21))&MASK
            elif opcode==103 and f==0:value=next_pc;next_pc=((x+im)&MASK)&~1
            elif opcode==99:
                conditions={0:x==y,1:x!=y,4:signed(x)<signed(y),5:signed(x)>=signed(y),6:x<y,7:x>=y}
                if f not in conditions:raise ValueError
                off=((word>>31)<<12)|((word>>7&1)<<11)|((word>>25&63)<<5)|((word>>8&15)<<1)
                if conditions[f]:next_pc=(pc+signed(off,13))&MASK
            elif opcode==3:
                width={0:1,1:2,2:4,4:1,5:2}.get(f,0);value=access((x+im)&MASK,width)
                if f in (0,1):value=signed(value,width*8)
            elif opcode==35:
                off=signed(((word>>25)<<5)|((word>>7)&31),12)
                access((x+off)&MASK,{0:1,1:2,2:4}.get(f,0),y)
            elif opcode==19:
                if f in (0,2,3,4,6,7):
                    value={0:lambda:x+im,2:lambda:int(signed(x)<im),3:lambda:int(x<(im&MASK)),
                           4:lambda:x^(im&MASK),6:lambda:x|(im&MASK),7:lambda:x&(im&MASK)}[f]()
                elif f==1 and hi==0:value=x<<rt
                elif f==5 and hi in (0,32):value=(signed(x) if hi else x)>>rt
                else:raise ValueError
            elif opcode==51:
                if hi==0:
                    value=[lambda:x+y,lambda:x<<(y%32),lambda:int(signed(x)<signed(y)),lambda:int(x<y),
                           lambda:x^y,lambda:x>>(y%32),lambda:x|y,lambda:x&y][f]()
                elif hi==32 and f in (0,5):value=x-y if f==0 else signed(x)>>(y%32)
                elif hi==1:
                    if f<=3:
                        value=[lambda:x*y,lambda:(signed(x)*signed(y))>>32,
                               lambda:(signed(x)*y)>>32,lambda:(x*y)>>32][f]()
                    elif f in (4,6):
                        xx,yy=signed(x),signed(y)
                        q=-1 if yy==0 else (abs(xx)//abs(yy))*(-1 if (xx<0)!=(yy<0) else 1)
                        value=q if f==4 else (xx if yy==0 else xx-q*yy)
                    elif f==5:value=MASK if y==0 else x//y
                    else:value=x if y==0 else x%y
                else:raise ValueError
            elif opcode==15 and f==0:pass
            else:raise ValueError
        except MemoryError:status=2;break
        except ValueError:status=3;break
        if value is not None and rd:regs[rd]=value&MASK
        regs[0]=0;pc=next_pc
    else:status=4;steps=FUEL
    def words(base):return (ct.c_uint32*32)(*(sum(memory.get(base+4*i+j,0)<<(8*j) for j in range(4)) for i in range(32)))
    return Output(regs[10],words(DATA),words(OUT),status,steps)
