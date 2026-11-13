"""Reference-only assembler and structured request encoding; never used by model inference."""
import subprocess
import tempfile
from pathlib import Path
import torch

def compile_target(spec):
    """Label construction only; same assembly/flags as corpus.py.

    Self-contained to avoid importing the corpus's isolation dependencies.
    The generation path never calls a compiler or inserts any target bytes.
    """
    if set(spec) == {'exit'}:
        assembly = f".global _start\n.text\n_start:\n mov ${spec['exit']},%edi\n mov $60,%eax\n syscall\n"
    else:
        payload = bytes.fromhex(spec['stdout'])
        assembly = ('.global _start\n.text\n_start:\n mov $1,%edi\n lea message(%rip),%rsi\n mov $8,%edx\n'
                    ' mov $1,%eax\n syscall\n xor %edi,%edi\n mov $60,%eax\n syscall\n.section .rodata\nmessage:\n.byte '
                    + ','.join(map(str, payload)) + '\n')
    assembly += '.section .note.GNU-stack,"",@progbits\n'
    with tempfile.TemporaryDirectory(prefix='hnet-label-') as temp:
        root = Path(temp); (root / 'target.s').write_text(assembly)
        subprocess.run(['as', '--64', '-o', str(root / 'target.o'), str(root / 'target.s')],
                       check=True, capture_output=True, timeout=10)
        subprocess.run(['ld', '--nmagic', '--build-id=none', '--strip-all', '-o', str(root / 'target'),
                        str(root / 'target.o')], check=True, capture_output=True, timeout=10)
        result = subprocess.run([str(root / 'target')], capture_output=True, timeout=2)
        expected_code = spec.get('exit', 0)
        expected_stdout = bytes.fromhex(spec.get('stdout', ''))
        if (result.returncode, result.stdout, result.stderr) != (expected_code, expected_stdout, b''):
            raise RuntimeError('reference program failed requested behavior')
        return (root / 'target').read_bytes()

def condition_tensor(specs, device="cpu"):
    result = []
    for spec in specs:
        if set(spec) == {"exit"} and type(spec["exit"]) is int and 0 <= spec["exit"] <= 255:
            mode, values = 0, [spec["exit"], *([0] * 8)]
        elif set(spec) == {"stdout"} and isinstance(spec["stdout"], str):
            payload = bytes.fromhex(spec["stdout"])
            if len(payload) != 8:
                raise ValueError("current learned stdout curriculum uses exactly eight requested bytes")
            mode, values = 1, [0, *payload]
        else:
            raise ValueError("generation request must specify exit status or eight-byte stdout")
        result.append([mode, *values])
    return torch.tensor(result, dtype=torch.long, device=device)
