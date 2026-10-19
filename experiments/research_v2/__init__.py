"""Bounded multi-path study; external research artifacts are explicitly located."""
from pathlib import Path
import os
import sys

ARTIFACT_ROOT = Path(os.environ.get('LIGHTNEST_ARTIFACT_ROOT', Path(__file__).resolve().parents[2]))
BASE = ARTIFACT_ROOT / 'baseline' / 'LightNest_RGB_v2'
if not (BASE / 'rgb' / 'runtime.py').is_file():
    raise RuntimeError('Set LIGHTNEST_ARTIFACT_ROOT to an extracted Hybrid v1 archive or a complete checkout.')
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
