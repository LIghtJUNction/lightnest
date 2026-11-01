#!/bin/sh
# Run the project from any working directory while keeping user paths relative to it.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --locked --project "$PROJECT_ROOT" python -m lightnest "$@"
