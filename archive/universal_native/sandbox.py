"""Isolated, time- and syscall-limited Linux x86-64 user-mode function runner."""
from pathlib import Path
import os
import platform
import struct
import subprocess
import tempfile


class Sandbox:
    def __init__(self, directory: Path):
        if platform.system()!="Linux" or platform.machine()!="x86_64":
            raise RuntimeError("native sandbox currently requires Linux x86-64")
        directory.mkdir(parents=True,exist_ok=True)
        self.runner=directory/"runner"
        if self.runner.exists():
            raise FileExistsError(self.runner)
        subprocess.run(["cc","-O2","-Wall","-Wextra","-Werror","-o",str(self.runner),
                        str(Path(__file__).with_suffix(".c"))],check=True,capture_output=True)

    def run(self, code: bytes, inputs: list[list[int]], timeout=2.0):
        if not code or len(code)>16384 or not 1<=len(inputs)<=64:
            raise ValueError("sandbox code/case limit exceeded")
        if any(len(row)!=3 or any(type(x)is not int or not 0<=x<2**64 for x in row) for row in inputs):
            raise ValueError("sandbox ABI requires three uint64 inputs")
        payload=struct.pack("<II",len(code),len(inputs))+code
        payload+=b"".join(struct.pack("<3Q",*row) for row in inputs)
        with tempfile.TemporaryFile() as output:
            try:
                result=subprocess.run([str(self.runner)],input=payload,stdout=output,stderr=subprocess.DEVNULL,
                                      timeout=timeout,env={"PATH":"/usr/bin"},cwd="/",close_fds=True)
            except subprocess.TimeoutExpired:
                return {"status":"timeout","outputs":None}
            output.seek(0); raw=output.read(513)
        if result.returncode or len(raw)!=len(inputs)*8:
            return {"status":"rejected","outputs":None,"returncode":result.returncode}
        return {"status":"ok","outputs":list(struct.unpack("<"+"Q"*len(inputs),raw))}
