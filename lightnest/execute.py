"""Run complete executables in isolated namespaces with bounded resources."""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

from .artifact import MAX_IMAGE, digest, inspect_elf


@dataclass(frozen=True)
class Case:
    args: tuple[str, ...] = ()
    stdin: bytes = b""

    def __post_init__(self):
        if len(self.args) > 64 or any(not isinstance(a, str) or "\0" in a or len(a) > 4096 for a in self.args):
            raise ValueError("invalid executable argument list")
        if not isinstance(self.stdin, bytes) or len(self.stdin) > 1024 * 1024:
            raise ValueError("stdin exceeds case capacity")

    def json(self):
        return {"args": list(self.args), "stdin": self.stdin.hex()}

    @classmethod
    def parse(cls, row):
        if set(row) != {"args", "stdin"}:
            raise ValueError("execution receives input only, not expected outputs")
        return cls(tuple(row["args"]), bytes.fromhex(row["stdin"]))


def _filter(file):
    library = ctypes.CDLL(ctypes.util.find_library("seccomp"), use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]; library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # ALLOW by default inside the filesystem/namespace jail.
    if not context:
        raise RuntimeError("could not create executable syscall filter")
    denied = ("clone", "clone3", "fork", "vfork", "socket", "connect", "bind", "listen", "accept", "accept4",
              "ptrace", "bpf", "unshare", "setns", "mount", "umount2", "open_by_handle_at",
              "process_vm_readv", "process_vm_writev", "keyctl", "perf_event_open", "userfaultfd")
    try:
        for name in denied:
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and library.seccomp_rule_add(context, 0x00050001, syscall, 0):
                raise RuntimeError(f"could not deny {name}")
        if library.seccomp_export_bpf(context, file.fileno()):
            raise RuntimeError("could not export executable syscall filter")
        file.flush(); file.seek(0)
    finally:
        library.seccomp_release(context)


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


class ExecutableRunner:
    def __init__(self, directory: Path, *, timeout=3.0, cache=True):
        self.directory = Path(directory).resolve(); self.directory.mkdir(parents=True, exist_ok=True)
        if not 0 < timeout <= 30:
            raise ValueError("invalid executable timeout")
        self.timeout, self.cache_enabled = timeout, cache
        self.cache = {}; self.native_calls = 0; self.cache_hits = 0; self.preflight_rejections = 0
        self.version = subprocess.check_output(["bwrap", "--version"]).decode().strip()
        self.policy = digest(Path(__file__).read_bytes() + Path(__file__).with_name("artifact.py").read_bytes()
                             + str((self.timeout, self.version)).encode())

    def run(self, raw: bytes, case: Case) -> dict:
        if not 1 <= len(raw) <= MAX_IMAGE:
            raise ValueError("executable size exceeds capacity")
        image_sha = digest(raw)
        key = (self.policy, image_sha, tuple(case.args), digest(case.stdin))
        if self.cache_enabled and key in self.cache:
            self.cache_hits += 1
            return dict(self.cache[key], cached=True)
        started = time.perf_counter()
        try:
            info = inspect_elf(raw)
            if info["machine"] != "EM_X86_64" or info["type"] not in ("ET_EXEC", "ET_DYN"):
                raise ValueError("native runner requires Linux x86-64 executable ELF")
            if not info["entry_in_executable_segment"]:
                raise ValueError("entry point lies outside executable mappings")
            if not any(s["type"] == "PT_LOAD" and s["flags"] & 1 and
                       s["address"] <= info["entry"] < s["address"] + s["file_size"] for s in info["segments"]):
                raise ValueError("entry point has no file-backed executable code")
        except ValueError as error:
            self.preflight_rejections += 1
            result = {"status": "invalid_artifact", "returncode": None, "stdout": "", "stderr": "",
                      "stdout_sha256": digest(b""), "stderr_sha256": digest(b""), "image_sha256": image_sha,
                      "wall_seconds": time.perf_counter() - started, "cached": False,
                      "preflight_rejected": True, "rejection": str(error)}
            if self.cache_enabled: self.cache[key] = result
            return result
        path = self.directory / (image_sha + ".elf")
        if not path.exists():
            try:
                with path.open("xb") as file:
                    file.write(raw)
                path.chmod(0o500)
            except FileExistsError:
                pass
        if digest(path.read_bytes()) != image_sha:
            raise ValueError("immutable execution image changed")
        started = time.perf_counter()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr, tempfile.TemporaryFile() as seccomp:
            _filter(seccomp)
            args = ["bwrap", "--unshare-all", "--new-session", "--die-with-parent", "--cap-drop", "ALL",
                    "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
                    "--proc", "/proc", "--dev", "/dev", "--size", str(16 * 1024 * 1024), "--tmpfs", "/tmp",
                    "--dir", "/app", "--ro-bind", str(path), "/app/program", "--dir", "/work", "--chdir", "/work",
                    "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "LC_ALL", "C",
                    "--setenv", "LANG", "C", "--setenv", "TZ", "UTC", "--seccomp", str(seccomp.fileno()),
                    "/app/program", *case.args]
            status = "ok"
            try:
                process = subprocess.run(args, input=case.stdin, stdout=stdout, stderr=stderr,
                                         timeout=self.timeout, close_fds=True, pass_fds=(seccomp.fileno(),),
                                         preexec_fn=_limits, env={"PATH": "/usr/bin"}, start_new_session=True)
                code = process.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
            stdout.seek(0); out = stdout.read(1024 * 1024 + 1)
            stderr.seek(0); err = stderr.read(1024 * 1024 + 1)
        if err.startswith(b"bwrap:"):
            status = "isolation_error"
        result = {"status": status, "returncode": code, "stdout": out.hex(), "stderr": err.hex(),
                  "stdout_sha256": digest(out), "stderr_sha256": digest(err), "image_sha256": image_sha,
                  "wall_seconds": time.perf_counter() - started, "cached": False}
        self.native_calls += 1
        if self.cache_enabled:
            self.cache[key] = result
        return result


def same_result(actual: dict, expected: dict) -> bool:
    return actual["status"] == expected["status"] == "ok" and all(
        actual[key] == expected[key] for key in ("returncode", "stdout", "stderr"))
