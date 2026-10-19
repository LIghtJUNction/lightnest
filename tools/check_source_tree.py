"""Validate that the published tree contains code, docs, and configuration only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".c", ".md", ".json", ".txt", ".yml", ".yaml", ".toml"}
IGNORE = {".git", "__pycache__", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN = {"data", "baseline_data", "models", "results", "runs", "caches", "checkpoints"}
errors = []
count = 0
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    if any(part in IGNORE for part in rel.parts):
        continue
    count += 1
    if path.name != ".gitignore" and path.suffix not in ALLOWED:
        errors.append(str(rel) + ": forbidden extension")
    if any(part in FORBIDDEN for part in rel.parts[:-1]):
        errors.append(str(rel) + ": generated-data directory")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Source-only tree OK: {count} tracked-style files scanned.")
