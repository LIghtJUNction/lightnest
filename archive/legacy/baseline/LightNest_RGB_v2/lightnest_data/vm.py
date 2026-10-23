"""Bounded validators derived from the supplied harder-task experiment.
Two decoding implementations; no native execution or network calls.
The VM contract is a restricted subset, not a general RV32 emulator.
"""
from .vendor import rv32

def s32(x): return (int(x)+2**31)%2**32-2**31

def sx(x,n): return (x+2**(n-1))%2**n-2**(n-1)

def assemble(source):
    """Existing assembler plus SW; labels retain the original raw-RV32 ABI."""
    labels={};placed=[];pc=256
    for line in source.splitlines():
        line=line.split('#',1)[0].strip()
        if not line: continue
        if line.endswith(':'):
            if line[:-1] in labels: raise ValueError('duplicate label')
            labels[line[:-1]]=pc
        else: placed.append((pc,line));pc+=4
    words=[]
    for pc,line in placed:
        parts=line.replace(',',' ').split()
        if parts[0]=='sw':
            imm,rs1=rv32._mem_operand(parts[2]);rs2=rv32._reg(parts[1]);rs1=rv32._reg(rs1)
            if not -2048<=imm<2048: raise ValueError('invalid store offset')
            v=imm&4095;word=((v>>5)<<25)|(rs2<<20)|(rs1<<15)|(2<<12)|((v&31)<<7)|0x23
        else: word=rv32._encode_line(line,pc,labels)
        words.append(word)
    return b''.join(w.to_bytes(4,'little') for w in words)

def execute_a(binary,inputs,fuel=1800):
    """Original LightNest stepper with bounded data-only SW support."""
    import struct
    if not binary or len(binary)%4 or len(binary)>1024 or len(inputs)>16:raise ValueError('bad VM input')
    mem=bytearray(4096);mem[256:256+len(binary)]=binary
    for i,x in enumerate(inputs):struct.pack_into('<i',mem,4*i,s32(x))
    regs=[0]*32;regs[12]=len(inputs);pc=256
    for step in range(fuel):
        if pc<256 or pc>=256+len(binary) or pc%4:raise ValueError('PC outside code')
        w=int.from_bytes(mem[pc:pc+4],'little')
        if w==rv32.EBREAK:return regs[10]
        if w&127==0x23 and (w>>12)&7==2:
            a=(w>>15)&31;b=(w>>20)&31;address=regs[a]+sx(((w>>25)<<5)|((w>>7)&31),12)
            if not 64<=address<256 or address%4:raise ValueError('store outside scratch')
            struct.pack_into('<i',mem,address,s32(regs[b]));pc+=4
        else:
            pc=rv32._step(w,pc,regs,mem)
            if pc is None:raise ValueError('invalid instruction')
        regs[0]=0
    raise ValueError('fuel exhausted')

def execute_b(binary,inputs,fuel=1800):
    """Second decoder/VM, unsigned register storage; does not call rv32._step."""
    if not binary or len(binary)%4 or len(binary)>1024:raise ValueError('length')
    words=[int.from_bytes(binary[k:k+4],'little') for k in range(0,len(binary),4)]
    reg=[0]*32;reg[12]=len(inputs);memory={i*4:int(v)&0xffffffff for i,v in enumerate(inputs)};pc=0
    for _ in range(fuel):
        if not 0<=pc<len(words):raise ValueError('pc')
        w=words[pc];op=w&127;d=w>>7&31;f=w>>12&7;a=w>>15&31;b=w>>20&31;hi=w>>25;x=reg[a];y=reg[b];nxt=pc+1;v=None
        if w==0x00100073:return s32(reg[10])
        if op==0x33:
            if hi==0:
                if f==0:v=x+y
                elif f==1:v=x<<(y&31)
                elif f==2:v=int(s32(x)<s32(y))
                elif f==3:v=int(x<y)
                elif f==4:v=x^y
                elif f==5:v=x>>(y&31)
                elif f==6:v=x|y
                elif f==7:v=x&y
            elif hi==32 and f in (0,5):v=x-y if f==0 else s32(x)>>(y&31)
            elif hi==1 and f==0:v=x*y
            else:raise ValueError('R encoding')
        elif op==0x13:
            if f==0:v=x+sx(w>>20,12)
            elif f==1 and hi==0:v=x<<b
            elif f==5 and hi in (0,32):v=(s32(x) if hi else x)>>b
            else:raise ValueError('I encoding')
        elif op==3 and f==2:
            addr=s32(x)+sx(w>>20,12)
            if addr<0 or addr>=256 or addr%4:raise ValueError('load')
            v=memory.get(addr,0)
        elif op==0x23 and f==2:
            addr=s32(x)+sx(((w>>25)<<5)|(w>>7&31),12)
            if not 64<=addr<256 or addr%4:raise ValueError('store')
            memory[addr]=y
        elif op==0x63 and f in (0,1,4):
            imm=((w>>31)<<12)|((w>>7&1)<<11)|((w>>25&63)<<5)|((w>>8&15)<<1)
            take=(x==y if f==0 else x!=y if f==1 else s32(x)<s32(y))
            if take:
                off=sx(imm,13)
                if off%4:raise ValueError('branch')
                nxt=pc+off//4
        elif op==0x6f and d==0:
            imm=((w>>31)<<20)|((w>>12&255)<<12)|((w>>20&1)<<11)|((w>>21&1023)<<1)
            off=sx(imm,21)
            if off%4:raise ValueError('jump')
            nxt=pc+off//4
        else:raise ValueError('opcode')
        if v is not None and d:reg[d]=v&0xffffffff
        reg[0]=0;pc=nxt
    raise ValueError('fuel')
