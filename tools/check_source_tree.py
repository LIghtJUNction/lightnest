"""Reject generated artifacts in the published source tree, including Lean output."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py', '.c', '.md', '.json', '.txt', '.yml', '.yaml', '.toml', '.lean'}
SPECIAL = {'.gitignore', 'lean-toolchain'}
IGNORE = {'.git', '__pycache__', '.venv', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.lake'}
FORBIDDEN = {'data', 'baseline_data', 'models', 'results', 'runs', 'caches', 'checkpoints', '.lake'}


def check(root: Path = ROOT) -> tuple[int, list[str]]:
    root = root.resolve()
    if (root / '.git').exists():
        result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                                capture_output=True, check=True)
        paths = [root / p.decode('utf-8') for p in result.stdout.split(b'\0') if p]
    else:
        paths = [p for p in root.rglob('*') if p.is_file()
                 and not IGNORE.intersection(p.relative_to(root).parts)]
    errors = []
    for path in paths:
        rel = path.relative_to(root)
        if path.is_symlink():
            errors.append(f'{rel}: source symlinks are not allowed')
            continue
        if not path.is_file():
            errors.append(f'{rel}: missing tracked file')
            continue
        if path.name not in SPECIAL and path.suffix not in ALLOWED:
            errors.append(f'{rel}: forbidden extension')
        if FORBIDDEN.intersection(rel.parts[:-1]):
            errors.append(f'{rel}: generated-data directory')
        if path.stat().st_size > 1024 * 1024:
            errors.append(f'{rel}: source file exceeds 1 MiB')
            continue
        try:
            text = path.read_text(encoding='utf-8')
            if '\x00' in text:
                errors.append(f'{rel}: binary payload')
        except UnicodeError:
            errors.append(f'{rel}: not UTF-8 source')
    return len(paths), errors


if __name__ == '__main__':
    count, errors = check()
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Source-only tree OK: {count} source files scanned.')
