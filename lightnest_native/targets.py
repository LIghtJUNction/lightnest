"""Instruction decoding through iced-x86 and the locally installed LLVM targets.

Decoding, execution and trained-model coverage are deliberately distinct.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import io
import platform
import re
import subprocess
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile
import iced_x86 as iced


def canonical_arch(name: str) -> str:
    name = name.lower().replace("-", "_")
    return {"amd64":"x86_64", "x86_64":"x86_64", "i386":"x86", "i686":"x86",
            "arm64":"aarch64", "armv7":"arm", "rv32":"riscv32", "rv64":"riscv64",
            "powerpc64le":"ppc64le", "s390x":"systemz"}.get(name,name)


def host_architecture() -> str:
    return canonical_arch(platform.machine())


def token(text: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (size-1) + 1


def run(args, *, input=None, timeout=20):
    result = subprocess.run(args, input=input, capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-2000:])
    return result


@lru_cache(maxsize=1)
def llvm_targets() -> dict[str, str]:
    text = run(["llvm-mc", "--version"]).stdout.decode()
    return {canonical_arch(m[0]):m[1].strip() for m in re.findall(r"^\s+([\w-]+)\s+- (.+)$",text,re.M)}


def llvm_triple(architecture):
    arch=canonical_arch(architecture)
    return {"x86_64":"x86_64", "x86":"i386", "arm":"armv7", "ppc32":"powerpc", "ppc64":"powerpc64",
            "ppc32le":"powerpcle", "ppc64le":"powerpc64le", "systemz":"s390x"}.get(arch,arch)


@dataclass(frozen=True)
class Instruction:
    offset: int
    raw: bytes
    text: str
    mnemonic: int
    flow: int
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    branch: int | None = None
    memory_read: bool = False
    memory_write: bool = False

    def json(self):
        return asdict(self) | {"raw":self.raw.hex()}


@lru_cache(maxsize=8192)
def decode(architecture: str, code: bytes) -> tuple[Instruction, ...]:
    arch = canonical_arch(architecture)
    if not code or len(code) > 65536:
        raise ValueError("decoder accepts 1..65536 code bytes per block")
    if arch in ("x86_64","x86"):
        factory=iced.InstructionInfoFactory();result=[]
        read_access={iced.OpAccess.READ,iced.OpAccess.COND_READ,iced.OpAccess.READ_WRITE,iced.OpAccess.READ_COND_WRITE}
        write_access={iced.OpAccess.WRITE,iced.OpAccess.COND_WRITE,iced.OpAccess.READ_WRITE,iced.OpAccess.READ_COND_WRITE}
        decoder=iced.Decoder(64 if arch=="x86_64" else 32,code,ip=0)
        for instruction in decoder:
            if instruction.is_invalid:
                return decode_llvm(arch,code)
            info=factory.info(instruction);reads=set();writes=set()
            for reg in info.used_registers():
                key=f"r{iced.RegisterExt.full_register(reg.register)}"
                if reg.access in read_access:reads.add(key)
                if reg.access in write_access:writes.add(key)
            if instruction.rflags_read:reads.add("flags")
            if instruction.rflags_written or instruction.rflags_cleared or instruction.rflags_set or instruction.rflags_undefined:
                writes.add("flags")
            memory=list(info.used_memory())
            mr=any(m.access in read_access for m in memory);mw=any(m.access in write_access for m in memory)
            if mr:reads.add("memory")
            if mw:writes.add("memory")
            branch=instruction.near_branch_target if instruction.flow_control in (iced.FlowControl.CONDITIONAL_BRANCH,iced.FlowControl.UNCONDITIONAL_BRANCH,iced.FlowControl.CALL) else None
            result.append(Instruction(instruction.ip,code[instruction.ip:instruction.next_ip],str(instruction),
                                      instruction.mnemonic,instruction.flow_control,tuple(sorted(reads)),tuple(sorted(writes)),branch,mr,mw))
    else:
        return decode_llvm(arch,code)
    if not result or b"".join(i.raw for i in result)!=code:
        raise ValueError("decoder did not cover every input byte")
    return tuple(result)


def decode_llvm(architecture,code):
    arch=canonical_arch(architecture)
    if arch not in llvm_targets():raise ValueError(f"LLVM target is unavailable: {architecture}")
    text=run(["llvm-mc",f"--triple={llvm_triple(arch)}","--disassemble","--show-encoding"],
             input=(" ".join(f"0x{x:02x}" for x in code)+"\n").encode()).stdout.decode()
    result=[];offset=0
    for line in text.splitlines():
        match=re.search(r"^(.*?)\s*(?:#|//|;|@)\s*encoding:\s*\[([^]]+)\]",line)
        if not match:continue
        raw=bytes(int(x.strip(),16) for x in match[2].split(","));assembly=match[1].strip()
        if code[offset:offset+len(raw)]!=raw:raise ValueError("LLVM disassembly changed instruction bytes")
        result.append(Instruction(offset,raw,assembly,token(assembly.split()[0],2048),0));offset+=len(raw)
    if not result or offset!=len(code):raise ValueError("decoder did not cover every input byte")
    return tuple(result)


def assemble(architecture,source):
    arch=canonical_arch(architecture)
    if arch not in llvm_targets():raise ValueError(f"LLVM target is unavailable: {architecture}")
    if not isinstance(source,str) or not 1<=len(source.encode())<=65536:raise ValueError("assembly block size invalid")
    with tempfile.TemporaryDirectory(prefix='lightnest-assemble-') as temp:
        path=Path(temp)/'block.o'
        run(['llvm-mc',f'--triple={llvm_triple(arch)}','--filetype=obj','-o',str(path),'-'],input=source.encode())
        elf=ELFFile(io.BytesIO(path.read_bytes()));section=elf.get_section_by_name('.text')
        if section is None:raise ValueError("backend did not emit an ELF .text block")
        for sec in elf.iter_sections():
            if sec.header.sh_type in ('SHT_RELA','SHT_REL') and sec.num_relocations():
                raise ValueError("unresolved relocations require a linker; cannot return standalone bytes")
        code=section.data();decode(arch,code);return code


def graph(instructions: tuple[Instruction, ...]):
    """Conservative linear register/memory features plus branch edges, not semantic proof."""
    edges=set();last={};offsets={ins.offset:i for i,ins in enumerate(instructions)}
    for i,ins in enumerate(instructions):
        for key in ins.reads:
            if key in last:edges.add((i,last[key]))
        for key in ins.writes:last[key]=i
        if ins.branch in offsets:edges.add((offsets[ins.branch],i))
    return tuple(sorted(edges))


def elf_functions(raw: bytes):
    elf=ELFFile(io.BytesIO(raw))
    arch={"EM_X86_64":"x86_64","EM_386":"x86","EM_AARCH64":"aarch64","EM_ARM":"arm",
          "EM_RISCV":"riscv64" if elf.elfclass==64 else "riscv32","EM_PPC64":"ppc64le" if elf.little_endian else "ppc64",
          "EM_MIPS":"mipsel" if elf.little_endian else "mips","EM_S390":"systemz","EM_LOONGARCH":"loongarch64"}.get(elf.header.e_machine)
    if arch is None:raise ValueError(f"unsupported ELF machine metadata: {elf.header.e_machine}")
    table=elf.get_section_by_name(".symtab")
    if table is None:raise ValueError("ELF has no function symbol table")
    functions=[]
    for symbol in table.iter_symbols():
        if symbol.entry.st_info.type!="STT_FUNC" or not symbol.entry.st_size or not isinstance(symbol.entry.st_shndx,int):continue
        section=elf.get_section(symbol.entry.st_shndx)
        offset=symbol.entry.st_value-section.header.sh_addr
        code=section.data()[offset:offset+symbol.entry.st_size]
        functions.append({"name":symbol.name,"architecture":arch,"binary":code.hex(),"section":section.name,
                          "instructions":[i.json() for i in decode(arch,code)]})
    return functions
