"""Run the source checks shared by local development and CI."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("lightnest", "archive", "tools", "tests")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", action="store_true", help="also run full Lean proof checks")
    args = parser.parse_args()

    missing = [tool for tool in ("gcc", "clang", "ld.lld", "llvm-mc", "bwrap") if not shutil.which(tool)]
    if missing:
        parser.exit(1, f"Missing compiler tools: {', '.join(missing)}. Install GCC, Clang, LLD, LLVM tools and bubblewrap.\n")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = [
        ["tools/check_source_tree.py"],
        ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        ["-m", "unittest", "discover", "-s", "archive/rpn3_rv32/tests", "-t", "archive/rpn3_rv32", "-v"],
        ["-m", "unittest", "discover", "-s", "archive/legacy/tests", "-t", "archive/legacy", "-v"],
        ["-m", "unittest", "discover", "-s", "archive/legacy/tools/template_io", "-p", "test_*.py", "-v"],
    ]
    if args.lean:
        commands.append(["archive/legacy/tools/check_lean.py"])

    # Compile in memory so syntax checks leave no bytecode in the source tree.
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if ".venv" not in path.parts and "__pycache__" not in path.parts:
                compile(path.read_bytes(), str(path.relative_to(ROOT)), "exec")
    print("Python syntax OK.", flush=True)

    for command in commands:
        print(f"Running: {sys.executable} {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
