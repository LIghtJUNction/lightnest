"""Conditional computation inside ONE nn.Module, not three external agents.

Green forms request-conditioned evidence slots; it does not delete raw nodes.
Blue learns a shallow readout; red continues the unchanged shared C64 state.
The first phase freezes the parent core. Thus optional new paths cannot corrupt
red's fallback. New green/blue/routing weights are initially UNTRAINED.

Inference performs a real subset dispatch: a blue acceptance skips the last
(red_rounds-blue_rounds) recurrent steps. A multiply-by-zero gate that still
executes all red steps would not be conditional computation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import common

ARCHITECTURE = 'lightnest-rgb-evidence-shared-core-v1'
LEGACY_ARCHITECTURE = 'lightnest-rgb-evidence-shared-core-v0'


@dataclass(frozen=True)
class RGBConfig:
    blue_rounds: int = 4
    red_rounds: int = 64
    evidence_slots: int = 8
    freeze_parent: bool = True
    green_enabled: bool = True

    def __post_init__(self):
        if not (type(self.blue_rounds) is int and type(self.red_rounds) is int
                and 1 <= self.blue_rounds < self.red_rounds <= 64):
            raise ValueError('require 1 <= blue_rounds < red_rounds <= 64')
        if type(self.evidence_slots) is not int or not 1 <= self.evidence_slots <= 64:
            raise ValueError('evidence_slots must be 1..64')
        if type(self.freeze_parent) is not bool or type(self.green_enabled) is not bool:
            raise ValueError('freeze_parent must be bool')


@dataclass(frozen=True)
class RoutePolicy:
    """Research threshold, disabled by default; not a correctness certificate.

    Enable only with separately evaluated green/blue/router weights. The CLI
    does not enable this policy: it defaults to red while routing is unvalidated.
    Training uses forward_train, never this hard-dispatch path.
    """
    enabled: bool = False
    max_predicted_error: float = 0.01

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be bool')
        if not math.isfinite(self.max_predicted_error) or not 0 <= self.max_predicted_error <= 1:
            raise ValueError('invalid routing threshold')


@dataclass
class GreenEvidence:
    slots: torch.Tensor
    source_weights: torch.Tensor  # [batch, slot, instruction]; not semantic proof


class GreenObserver(nn.Module):
    """Bounded latent summary with a pointer back to EVERY source instruction.

    source_weights[...,i] refers to byte interval [4*i,4*i+4). Padding receives
    zero mass. Repeated instruction occurrences are NOT merged. Exact whole
    observations may be deduplicated in evidence.py before this stage.
    """
    def __init__(self, width: int, slots: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.request_query = nn.Linear(width, width, bias=False)
        self.keys = nn.Linear(width, width, bias=False)
        self.values = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, request, valid):
        q = self.queries[None] + self.request_query(request)[:, None]
        scores = torch.bmm(q, self.keys(h).transpose(1, 2)) / math.sqrt(h.shape[-1])
        weights = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        slots = self.norm(torch.bmm(weights, self.values(h)))
        return GreenEvidence(slots, weights)


class BlueActor(nn.Module):
    """Small independent readout initialized from the parent's byte/STOP heads."""
    def __init__(self, parent):
        super().__init__()
        d = parent.config.width
        for name in ('byte_fusion', 'pointer', 'replacement', 'stop_fusion', 'stop'):
            setattr(self, name, deepcopy(getattr(parent, name)))
        self.memory_query = nn.Linear(d, d, bias=False)
        self.memory_out = nn.Linear(d, d, bias=False)
        # Small nonzero residual lets green receive gradients in the first step.
        self.memory_scale = nn.Parameter(torch.tensor(0.01))
        self.none_index = parent.none_index

    def forward(self, h, context, evidence):
        q = self.memory_query(h)
        weights = (torch.bmm(q, evidence.slots.transpose(1, 2)) / math.sqrt(h.shape[-1])).softmax(-1)
        h = h + self.memory_scale.tanh() * self.memory_out(torch.bmm(weights, evidence.slots))
        h = h.masked_fill(~context.valid[..., None], 0.)
        binary, emb, slots = context.binary, context.byte_embeddings, context.slots
        batch, length = binary.shape
        decoded = torch.tanh(self.byte_fusion(torch.cat((h.repeat_interleave(4, 1), emb, slots), -1)))
        pointer = self.pointer(decoded).squeeze(-1).masked_fill(binary.eq(0), -1e9)
        maximum = h.masked_fill(~context.valid[..., None], -1e9).max(1).values
        mean = h.sum(1) / context.valid.sum(1, keepdim=True)
        stop = self.stop(torch.tanh(self.stop_fusion(torch.cat((maximum, mean, context.command), -1))))
        scores = torch.cat((F.pad(pointer, (0, self.none_index-length), value=-1e9), stop), -1)
        bits = self.replacement(decoded).masked_fill(binary.eq(0)[..., None], 0.)
        return scores, F.pad(bits, (0, 0, 0, self.none_index-length))


class RGBNetwork(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, parent, config=RGBConfig()):
        super().__init__()
        if parent.local_stop_enabled:
            raise ValueError('v0 requires the pinned C64 without local STOP branch')
        self.config = config
        self.red = deepcopy(parent)
        self.green = GreenObserver(parent.config.width, config.evidence_slots)
        self.blue = BlueActor(parent)
        d = parent.config.width
        self.router = nn.Sequential(nn.Linear(2*d+3, 32), nn.Tanh(), nn.Linear(32, 1))
        if config.freeze_parent:
            for p in self.red.parameters(): p.requires_grad_(False)
        self.policy = RoutePolicy()

    @property
    def none_index(self):
        return self.red.none_index

    def encode_shared(self, command, binary, relations):
        """Frozen reusable input states; no label or policy accesses."""
        inputs, relation, context = self.red.encode_inputs(command, binary, relations)
        h = inputs
        for _ in range(self.config.blue_rounds):
            h = self.red.shared_step(h, inputs, relation, context.valid)
        return inputs, relation, context, h

    def fast_readout(self, context, h):
        evidence = self.green(h, context.command, context.valid)
        if not self.config.green_enabled:
            # Ablation: same allocated weights and forward operations, but no
            # learned summary reaches the actor or router. Not equal active DOF.
            evidence = GreenEvidence(torch.zeros_like(evidence.slots),
                                     torch.zeros_like(evidence.source_weights))
        scores, bits = self.blue(h, context, evidence)
        p = scores.softmax(-1)
        top = p.topk(2, dim=-1).values
        routing = torch.cat((context.command, evidence.slots.mean(1),
                             top[:, :1], top[:, :1]-top[:, 1:], p[:, -1:]), -1)
        # Error prediction must not teach the actor to manipulate its own gate.
        risk_logits = self.router(routing.detach()).squeeze(-1)
        return evidence, (scores, bits), risk_logits

    def _begin(self, command, binary, relations):
        inputs, relation, context, h = self.encode_shared(command, binary, relations)
        evidence, blue, risk = self.fast_readout(context, h)
        return inputs, relation, context, h, evidence, blue, risk

    def _continue_red(self, h, inputs, relations, context):
        for _ in range(self.config.red_rounds-self.config.blue_rounds):
            h = self.red.shared_step(h, inputs, relations, context.valid)
        return self.red.decode_state(h, context)

    def forward_train(self, command, binary, relations):
        """Both paths for supervised training; no inference-cost claim here."""
        inputs, rel, ctx, h, green, blue, risk = self._begin(command, binary, relations)
        red = self._continue_red(h, inputs, rel, ctx)
        return {'blue': blue, 'red': red, 'risk_logits': risk, 'evidence': green}

    @staticmethod
    def _slice_context(context, indices):
        return type(context)(*(getattr(context, name).index_select(0, indices)
                               for name in ('binary','byte_embeddings','slots','command','valid')))

    @torch.inference_mode()
    def forward(self, command, binary, relations, *, mode='auto', evidence=False):
        if self.training:
            raise RuntimeError('call eval() for inference or forward_train() for training')
        if mode not in ('auto', 'red_only', 'blue_only'):
            raise ValueError('mode must be auto/red_only/blue_only')
        inputs, rel, ctx, h = self.encode_shared(command, binary, relations)
        batch = binary.shape[0]
        # Red-only is independent of untrained or corrupted auxiliary modules.
        # Policy disabled is genuinely zero fast-path overhead (except when the
        # caller explicitly requests the diagnostic green evidence).
        if mode == 'red_only' or (mode == 'auto' and not self.policy.enabled):
            scores, bits = self._continue_red(h, inputs, rel, ctx)
            use_blue = torch.zeros(batch, dtype=torch.bool, device=binary.device)
            risk = torch.ones(batch, device=binary.device)
            invalid = use_blue.clone()
            green = self.green(h, ctx.command, ctx.valid) if evidence else None
            fast_evaluated = False
        else:
            green, blue, risk_logits = self.fast_readout(ctx, h)
            valid_fast = torch.isfinite(risk_logits)
            for t in (*blue, green.slots, green.source_weights):
                valid_fast &= torch.isfinite(t).flatten(1).all(1)
            invalid = ~valid_fast
            if mode == 'blue_only' and bool(invalid.any()):
                raise FloatingPointError('nonfinite forced blue prediction')
            risk = risk_logits.sigmoid()
            risk = torch.where(valid_fast, risk, torch.ones_like(risk))
            if mode == 'blue_only':
                use_blue = valid_fast
            else:
                use_blue = valid_fast & (risk <= self.policy.max_predicted_error)
            red_ids = (~use_blue).nonzero().flatten()
            scores, bits = (t.clone() for t in blue)
            if red_ids.numel():
                sc, bi = self._continue_red(h.index_select(0, red_ids),
                    inputs.index_select(0, red_ids), rel.index_select(0, red_ids),
                    self._slice_context(ctx, red_ids))
                scores.index_copy_(0, red_ids, sc)
                bits.index_copy_(0, red_ids, bi)
            fast_evaluated = True
        if not (bool(torch.isfinite(scores).all()) and bool(torch.isfinite(bits).all())):
            raise FloatingPointError('nonfinite selected output; red is also invalid')
        result = {'scores': scores, 'bits': bits, 'used_blue': use_blue,
                  'predicted_blue_error': risk,
                  'fast_nonfinite_fallback': invalid,
                  'fast_evaluated': fast_evaluated,
                  'shared_step_examples': batch*self.config.blue_rounds,
                  'red_step_examples': int((~use_blue).sum())*(self.config.red_rounds-self.config.blue_rounds),
                  'diagnostic_mode': mode != 'auto', 'deployment_approved': False}
        if evidence and green is not None:
            result['evidence_slots'] = green.slots
            result['source_weights'] = green.source_weights
        return result

    def parameter_counts(self):
        return {k:sum(p.numel() for p in getattr(self,k).parameters())
                for k in ('red','green','blue','router')}

    def save(self, path, *, note='untrained architecture initialization'):
        """One checkpoint; never persist an enabled policy as an implicit default."""
        path=Path(path)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:
            torch.save({'architecture':ARCHITECTURE,'rgb_config':asdict(self.config),
                        'red_config':asdict(self.red.config),'state_dict':self.state_dict(),
                        'fast_path_validated':False,'note':note},f)
        path.chmod(0o600)


def load_rgb(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>32*1024**2:
        raise ValueError('checkpoint must be a regular file <=32 MiB')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('architecture') not in (ARCHITECTURE, LEGACY_ARCHITECTURE): raise ValueError('architecture mismatch')
    parent=common.arch.BreakthroughNetwork(common.GrowthConfig(**saved['red_config']),local_stop=False)
    model=RGBNetwork(parent,RGBConfig(**saved['rgb_config']))
    expected=model.state_dict();state=saved['state_dict']
    if set(expected)!=set(state): raise ValueError('tensor inventory mismatch')
    for k,v in state.items():
        if not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not bool(torch.isfinite(v).all()):
            raise ValueError('invalid tensor '+k)
    model.load_state_dict(state,strict=True)
    return model.eval()
