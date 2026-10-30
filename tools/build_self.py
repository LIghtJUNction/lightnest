"""Build a single ELF containing LightNest's model weights and runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


WRAPPER_MAGIC = b"LIGHTNEST_WRAPPER_V1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--any2any-checkpoint", type=Path, required=True)
    parser.add_argument("--language-model", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, default=Path("/usr/lib/ollama/llama-server"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.any2any_checkpoint, args.language_model, args.llama_server):
        if not path.is_file(): raise SystemExit(f"missing build input: {path}")
    cpu_backends = sorted(args.llama_server.parent.glob("libggml-cpu-*.so"))
    if not cpu_backends:
        raise SystemExit(f"no llama.cpp CPU backends beside {args.llama_server}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lightnest-self-build-") as work:
        # PyInstaller's ``--add-data`` destination is a directory, so stage
        # the checkpoint under the stable name the runtime uses.
        staged_checkpoint = Path(work) / "self-any2any.pt"
        shutil.copy2(args.any2any_checkpoint, staged_checkpoint)
        backend_args = [item for backend in cpu_backends for item in ("--add-binary", f"{backend}:.")]
        command = ["uv", "run", "--locked", "pyinstaller", "--clean", "--onefile",
                   "--name", "lightnest-self", "--distpath", str(Path(work) / "dist"),
                   "--workpath", str(Path(work) / "build"), "--specpath", str(Path(work)),
                   "--collect-all", "torch", "--add-data", f"{staged_checkpoint}:.",
                   "--add-binary", f"{args.llama_server}:.", *backend_args,
                   "--paths", ".", "lightnest/self_hosting/self_runtime.py"]
        subprocess.run(command, check=True)
        built = Path(work) / "dist" / "lightnest-self"
        launcher = Path(work) / "lightnest-self-launcher"
        subprocess.run(["cc", "-O2", "-s", "-std=c11",
                        str(Path(__file__).with_name("self_launcher.c")), "-o", str(launcher)], check=True)
        base_size = built.stat().st_size
        model_size = args.language_model.stat().st_size
        shutil.copy2(launcher, args.output)
        with args.output.open("ab") as output, built.open("rb") as base, args.language_model.open("rb") as language:
            shutil.copyfileobj(base, output, length=16 * 1024 * 1024)
            shutil.copyfileobj(language, output, length=16 * 1024 * 1024)
            output.write(WRAPPER_MAGIC)
            output.write(struct.pack("<QQ", base_size, model_size))
        args.output.chmod(0o755)
        print(args.output)


if __name__ == "__main__": main()
