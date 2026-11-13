"""Host imports only; candidate computation is defined in model.wgt."""
from pathlib import Path
import torch
from torch import nn
from weightlang.modules import bind_models

globals().update(bind_models(Path(__file__).with_name('model.wgt'), globals(), backend='torch'))
