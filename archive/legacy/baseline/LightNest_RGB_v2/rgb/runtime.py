"""Use a pinned local C64 runtime, without modifying the source archive."""
from pathlib import Path
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime' / 'experiment'))
import common
common.graph = lru_cache(maxsize=512)(common.graph.__wrapped__)


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    torch.use_deterministic_algorithms(True)
