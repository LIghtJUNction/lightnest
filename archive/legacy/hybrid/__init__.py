"""Bounded LightNest hybrid state/action research. No automatic deployment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'baseline'/'LightNest_RGB_v2'
if str(BASE) not in sys.path: sys.path.insert(0,str(BASE))
