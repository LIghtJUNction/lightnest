"""Function-preserving growth of the complete LightNest relation network.

Every variant keeps the original command/byte encoder, graph message matrices,
GRU, and all byte/STOP heads.  The original configured message rounds execute
unchanged.  New work is either extra applications of that SAME graph/GRU core
or a gated residual branch after the old rounds.  No extra normalization is
inserted on the inherited path.  Inputs remain ordinary byte/command tokens
and direct definition/use edges; labels and execution results are not inputs.

``zero_extra`` starts each extra residual coefficient at zero. ``fixed_extra``
executes the complete extra graph updates, and is deliberately not advertised
as function preserving. ``zero_attention`` and ``zero_mlp`` start one outer
residual coefficient at zero; their parameter budgets are closely matched.
The zero coefficients are trainable tanh gates, not an adaptive halting rule.
ALL configured extra work is executed, including when a gate is zero.

Zero-gated branches are related to ReZero (Bachlechner et al., UAI 2021);
preserving the old function when growing a model is related to Net2Net
(Chen et al., ICLR 2016).  This is a controlled adaptation, not a novelty claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import stat

import torch
from torch import nn
from torch.nn import functional as F

from .relation_patch import (
    RelationConfig, RelationPatchNetwork, RelationPredictor,
    encode_batch, load_relation, supervised_loss,
)

ARCHITECTURE = 'rv32-full-parent-growth-v1'
VARIANTS = ('zero_extra', 'fixed_extra', 'zero_attention', 'zero_mlp')
MAX_CHECKPOINT_BYTES = 32 * 1024**2
PARAMETER_ROLES = ('hidden_matrix', 'embedding', 'output', 'norm_bias', 'auxiliary')


@dataclass(frozen=True)
class GrowthConfig(RelationConfig):
    message_steps: int = 8
    variant: str = 'zero_extra'
    extra_steps: int = 8
    heads: int = 4
    ff_width: int | None = None
    mlp_width: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in VARIANTS:
            raise ValueError('unknown growth variant')
        if (type(self.extra_steps) is not int or not 0 <= self.extra_steps <= 56
                or self.message_steps + self.extra_steps > 64):
            raise ValueError('extra_steps must be in 0..56 with at most 64 total rounds')
        if (type(self.heads) is not int or not 1 <= self.heads <= 16
                or self.width % self.heads):
            raise ValueError('heads must divide width and be in 1..16')
        if self.ff_width is None:
            object.__setattr__(self, 'ff_width', 2 * self.width)
        if type(self.ff_width) is not int or not 16 <= self.ff_width <= 1024:
            raise ValueError('ff_width must be in 16..1024')
        if self.mlp_width is None:
            # Attention: 4*d*d + 2*d*ff + 9*d + ff parameters.
            # MLP: (2*d + 1)*hidden + 3*d parameters.  Every parameter is used.
            target = 4 * self.width**2 + 2 * self.width * self.ff_width + 9 * self.width + self.ff_width
            matched = round((target - 3 * self.width) / (2 * self.width + 1))
            object.__setattr__(self, 'mlp_width', matched)
        if type(self.mlp_width) is not int or not 16 <= self.mlp_width <= 2048:
            raise ValueError('mlp_width must be in 16..2048')


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        descriptor = json.dumps([name, list(value.shape), str(value.dtype)], separators=(',', ':'))
        digest.update(descriptor.encode() + b'\0')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _require_finite(value, label):
    # In particular, 0 * NaN is NOT a function-preserving residual.  Do not
    # silently discard invalid branch results even while its gate is zero.
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite growth ' + label)


class _AttentionResidual(nn.Module):
    """One global bidirectional attention + FF branch, with internal pre-LNs.

    There is no positional code, answer-derived mask, graph closure, or CPU
    register table.  The matching MLP sees the same final inherited states.
    """
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, config.ff_width)
        self.ff_out = nn.Linear(config.ff_width, d)

    def forward(self, h, valid):
        batch, n, d = h.shape
        qkv_input = self.norm1(h)
        dim = d // self.heads
        q = self.q_proj(qkv_input).reshape(batch, n, self.heads, dim).transpose(1, 2)
        k = self.k_proj(qkv_input).reshape(batch, n, self.heads, dim).transpose(1, 2)
        v = self.v_proj(qkv_input).reshape(batch, n, self.heads, dim).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(dim))
        weights = weights.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
        weights = weights.masked_fill(~valid[:, None, :, None], 0.)
        attended = (weights @ v).transpose(1, 2).reshape(batch, n, d)
        attended = self.out_proj(attended).masked_fill(~valid[..., None], 0.)
        ff = self.ff_out(F.gelu(self.ff_in(self.norm2(h + attended))))
        return (attended + ff).masked_fill(~valid[..., None], 0.)


class _MLPResidual(nn.Module):
    """Matched-budget branch without communication between instructions."""
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.ff_in = nn.Linear(config.width, config.mlp_width)
        self.ff_out = nn.Linear(config.mlp_width, config.width)

    def forward(self, h, valid):
        delta = self.ff_out(F.gelu(self.ff_in(self.norm(h))))
        return delta.masked_fill(~valid[..., None], 0.)


class GrowthPatchNetwork(RelationPatchNetwork):
    architecture = ARCHITECTURE
    native_packer_compatible = False

    def __init__(self, config=GrowthConfig()):
        if not isinstance(config, GrowthConfig):
            raise ValueError('a GrowthConfig is required')
        # No inherited module is deleted or replaced.  Their tensor names are
        # unchanged, permitting an exact full-parent copy and a direct audit.
        super().__init__(config)
        self.initialization_report = None
        if config.variant == 'zero_extra':
            self.extra_gates = nn.Parameter(torch.zeros(config.extra_steps)) if config.extra_steps else None
        elif config.variant == 'zero_attention':
            self.growth_gate = nn.Parameter(torch.zeros(()))
            self.growth = _AttentionResidual(config)
        elif config.variant == 'zero_mlp':
            self.growth_gate = nn.Parameter(torch.zeros(()))
            self.growth = _MLPResidual(config)

    @property
    def parameter_count(self):
        return sum(value.numel() for value in self.parameters())

    def parameter_roles(self):
        """Same inherited tensor roles as the prior relation training driver."""
        embeddings = {name + '.weight' for name, module in self.named_modules()
                      if isinstance(module, nn.Embedding)}
        outputs = ('pointer.', 'replacement.', 'stop.', 'stop_fusion.')
        roles = {}
        for name, value in self.named_parameters():
            if name in embeddings:
                role = 'embedding'
            elif name.startswith(outputs):
                role = 'output'
            elif value.ndim < 2:
                role = 'norm_bias'
            elif value.ndim == 2:
                role = 'hidden_matrix'
            else:
                role = 'auxiliary'
            roles[name] = role
        return roles

    def parameter_counts_by_role(self):
        result = {role: 0 for role in PARAMETER_ROLES}
        roles = self.parameter_roles()
        for name, value in self.named_parameters():
            result[roles[name]] += value.numel()
        return result

    @classmethod
    def from_relation(cls, parent, *, config=None, seed=0):
        """Copy ALL parent tensors; initialize only the new growth tensors.

        Both the caller's RNG and parent's tensors/configuration are unchanged.
        The finite zero branches should preserve all logits and decisions at
        birth; the experiment must still independently verify that property.
        Optimizer state is deliberately not inherited.
        """
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in 0..2**63-1')
        with torch.random.fork_rng(devices=[]):
            if isinstance(parent, (str, Path)):
                parent = load_relation(parent)
            if type(parent) is not RelationPatchNetwork:
                raise ValueError('a plain RelationPatchNetwork or its checkpoint is required')
            if config is None:
                config = GrowthConfig(**asdict(parent.config))
            for attribute in ('mode', 'width', 'command_capacity', 'binary_capacity', 'message_steps'):
                if getattr(parent.config, attribute) != getattr(config, attribute):
                    raise ValueError('parent and child must preserve ' + attribute)
            child = cls(config)
            parent_named = dict(parent.named_parameters())
            modules = dict(child.named_modules())
            mapping, evidence = {}, {}
            copied, initialized = 0, 0
            with torch.no_grad():
                for name, value in child.named_parameters():
                    source = parent_named.get(name)
                    if source is not None:
                        if source.shape != value.shape or source.dtype != torch.float32:
                            raise ValueError('incompatible parent parameter: ' + name)
                        if not bool(torch.isfinite(source).all()):
                            raise ValueError('nonfinite parent parameter: ' + name)
                        value.copy_(source.detach().cpu())
                        copied += value.numel()
                    else:
                        derived = int.from_bytes(hashlib.sha256(f'{seed}\0{name}'.encode()).digest()[:8],
                                                 'little') % 2**63
                        generator = torch.Generator(device='cpu').manual_seed(derived)
                        module_name, _, leaf = name.rpartition('.')
                        module = modules[module_name]
                        if name in ('extra_gates', 'growth_gate'):
                            value.zero_()
                        elif isinstance(module, nn.LayerNorm):
                            value.fill_(0. if leaf == 'bias' else 1.)
                        elif isinstance(module, nn.Linear):
                            bound = 1 / math.sqrt(module.in_features)
                            value.uniform_(-bound, bound, generator=generator)
                        else:
                            raise ValueError('missing new-parameter initialization: ' + name)
                        initialized += value.numel()
                    mapping[name] = name if source is not None else None
                    evidence[name] = dict(parent_name=mapping[name], shape=list(value.shape),
                                          elements=value.numel(), child_sha256=_tensor_hash(value),
                                          parent_sha256=_tensor_hash(source) if source is not None else None,
                                          exact_copy=bool(torch.equal(value, source.detach().cpu()))
                                          if source is not None else None)
            omitted = sorted(set(parent_named) - {value for value in mapping.values() if value is not None})
            if omitted:
                raise ValueError('growth omitted parent tensors: ' + ', '.join(omitted))
            expected = config.variant != 'fixed_extra' or config.extra_steps == 0
            report = dict(
                architecture=ARCHITECTURE, operation='preserve_full_parent_and_append_growth', seed=seed,
                function_preserving_expected=expected, independent_birth_audit_required=True,
                optimizer_state_inherited=False, parameter_name_map=mapping, tensors=evidence,
                parameters=child.parameter_count, parameters_by_role=child.parameter_counts_by_role(),
                copied_parameter_values=copied, newly_initialized_parameter_values=initialized,
                all_copied_tensors_exact=all(e['exact_copy'] for e in evidence.values()
                                            if e['parent_name'] is not None),
                omitted_parent_parameters=omitted,
                parent_state_sha256=state_fingerprint(parent), child_initial_state_sha256=state_fingerprint(child),
                parent_config=asdict(parent.config), child_config=asdict(child.config),
                unchanged_parent_rounds=parent.config.message_steps,
                new_initialization='per-name SHA256(seed + NUL + name), first 8 bytes little endian mod 2**63',
                all_growth_work_executed_even_at_zero_gate=True,
                no_normalization_added_to_inherited_path=True,
                gate_semantics='tanh of trainable zero-initialized coefficient; fixed_extra applies F directly',
                branch_first_step='at zero gate, branch-only weights have zero task gradient; gate can learn',
            )
            child.initialization_report = report
            child.train(parent.training)
        return child, report

    def _extra_steps(self, extra_steps):
        if self.config.variant in ('zero_attention', 'zero_mlp'):
            if extra_steps not in (None, 0):
                raise ValueError('attention/MLP branches do not have extra graph rounds')
            return 0
        result = self.config.extra_steps if extra_steps is None else extra_steps
        if (type(result) is not int or not 0 <= result <= 56
                or self.config.message_steps + result > 64):
            raise ValueError('extra steps must be in 0..56 with at most 64 total rounds')
        if self.config.variant == 'zero_extra' and result > self.config.extra_steps:
            raise ValueError('zero_extra cannot invent untrained gates beyond its configured depth')
        return result

    def gate_values(self):
        if self.config.variant == 'zero_extra':
            return [] if self.extra_gates is None else self.extra_gates.detach().tanh().cpu().tolist()
        if self.config.variant == 'fixed_extra':
            return [1.] * self.config.extra_steps
        return [float(self.growth_gate.detach().tanh())]

    def compute_accounting(self, batch_size, instruction_count, *, extra_steps=None):
        """Executed dense core MACs, excluding encoder/heads/nonlinear work.

        Relation matrices are multiplied densely even when their edges are
        sparse.  Counts include new work at a zero gate.  This is not a wall
        time or a complete FLOP estimate; real CPU time must also be measured.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch size must be a positive integer')
        if (type(instruction_count) is not int
                or not 1 <= instruction_count <= self.config.binary_capacity // 4):
            raise ValueError('instruction count exceeds capacity')
        extra = self._extra_steps(extra_steps)
        b, n, d = batch_size, instruction_count, self.config.width
        per_graph = b * (8 * n * d * d + 2 * n * n * d)
        attention_macs = mlp_macs = scores = 0
        if self.config.variant == 'zero_attention':
            attention_macs = b * (4 * n * d * d + 2 * n * n * d + 2 * n * d * self.config.ff_width)
            scores = b * self.config.heads * n * n
        elif self.config.variant == 'zero_mlp':
            mlp_macs = b * 2 * n * d * self.config.mlp_width
        return dict(
            variant=self.config.variant, parameters=self.parameter_count,
            inherited_graph_rounds=self.config.message_steps, executed_extra_graph_rounds=extra,
            executed_graph_rounds=self.config.message_steps + extra,
            executed_attention_blocks=int(self.config.variant == 'zero_attention'),
            executed_mlp_branches=int(self.config.variant == 'zero_mlp'),
            core_linear_macs=per_graph * (self.config.message_steps + extra) + attention_macs + mlp_macs,
            dense_attention_score_elements=scores, all_new_work_executed_at_zero_gate=True,
            mac_exclusions=['frontend', 'output_heads', 'bias', 'normalization', 'nonlinearities',
                            'softmax', 'masking', 'gates', 'finite_checks'])

    def forward(self, command, binary, relations, message_steps=None, return_states=False,
                *, extra_steps=None):
        if message_steps is not None and message_steps != self.config.message_steps:
            raise ValueError('the inherited message depth is fixed; use extra_steps for a depth diagnostic')
        steps = self.config.message_steps
        self._validate(command, binary, relations, steps)
        extra = self._extra_steps(extra_steps)
        batch, length = binary.shape
        n, d = length // 4, self.config.width
        # This encoder, original loop, and output decoder preserve the exact
        # operation order in RelationPatchNetwork.forward.  Birth tests bind
        # every logit and the complete original hidden-state prefix to it.
        byte_embeddings = self.byte_embedding(binary)
        raw = (binary - 1).clamp_min(0).reshape(batch, n, 4)
        fields = torch.cat((byte_embeddings.reshape(batch, n, 64),
                            self.opcode_embedding(raw[:, :, 0] & 127),
                            self.funct3_embedding((raw[:, :, 1] >> 4) & 7),
                            self.funct7_embedding(raw[:, :, 3] >> 1)), -1)
        c = self._command(command)
        valid = binary[:, ::4].ne(0)
        inputs = torch.tanh(self.instruction_fusion(torch.cat((fields, c[:, None].expand(-1, n, -1)), -1)))
        inputs = inputs.masked_fill(~valid[..., None], 0.)
        h = inputs
        if self.config.mode == 'local':
            identity = torch.eye(n, device=binary.device)[None] * valid[:, :, None]
            relation = torch.stack((identity, identity), 1)
        else:
            relation = relations
        states = [h] if return_states else None
        for _ in range(steps):
            from_definition = torch.bmm(relation[:, 0], self.definition_message(h))
            from_use = torch.bmm(relation[:, 1], self.use_message(h))
            messages = torch.tanh(self.message_norm(inputs + from_definition + from_use))
            h = self.update(messages.reshape(-1, d), h.reshape(-1, d)).reshape(batch, n, d)
            h = h.masked_fill(~valid[..., None], 0.)
            if return_states:
                states.append(h)
        for index in range(extra):
            from_definition = torch.bmm(relation[:, 0], self.definition_message(h))
            from_use = torch.bmm(relation[:, 1], self.use_message(h))
            messages = torch.tanh(self.message_norm(inputs + from_definition + from_use))
            proposed = self.update(messages.reshape(-1, d), h.reshape(-1, d)).reshape(batch, n, d)
            proposed = proposed.masked_fill(~valid[..., None], 0.)
            _require_finite(proposed, 'graph update')
            if self.config.variant == 'zero_extra':
                h = h + self.extra_gates[index].tanh() * (proposed - h)
            else:
                # Applying F directly also makes fixed_extra exactly equal to
                # the plain parent's deeper execution (up to its depth cap).
                h = proposed
            h = h.masked_fill(~valid[..., None], 0.)
            if return_states:
                states.append(h)
        if self.config.variant in ('zero_attention', 'zero_mlp'):
            delta = self.growth(h, valid)
            _require_finite(delta, 'residual branch')
            h = (h + self.growth_gate.tanh() * delta).masked_fill(~valid[..., None], 0.)
            if return_states:
                states.append(h)
        slots = self.byte_slots(torch.arange(length, device=binary.device) % 4)[None].expand(batch, -1, -1)
        decoded = torch.tanh(self.byte_fusion(torch.cat((h.repeat_interleave(4, 1), byte_embeddings, slots), -1)))
        pointer = self.pointer(decoded).squeeze(-1).masked_fill(binary.eq(0), -1e9)
        max_h = h.masked_fill(~valid[..., None], -1e9).max(1).values
        mean_h = h.sum(1) / valid.sum(1, keepdim=True)
        stop = self.stop(torch.tanh(self.stop_fusion(torch.cat((max_h, mean_h, c), -1))))
        pointer = F.pad(pointer, (0, self.none_index - length), value=-1e9)
        scores = torch.cat((pointer, stop), -1)
        byte_logits = self.replacement(decoded).masked_fill(binary.eq(0)[..., None], 0.)
        bits = F.pad(byte_logits, (0, 0, 0, self.none_index - length))
        return (scores, bits, states) if return_states else (scores, bits)

    def save(self, path, **metadata):
        reserved = {'architecture', 'config', 'state_dict', 'initialization_report'}
        if reserved.intersection(metadata):
            raise ValueError('metadata cannot replace architecture, config, tensors, or initialization report')
        payload = dict(architecture=ARCHITECTURE, config=asdict(self.config), state_dict=self.state_dict(),
                       initialization_report=self.initialization_report, **metadata)
        if str(path).endswith('.gz'):
            with Path(path).open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                    torch.save(payload, stream)
        else:
            torch.save(payload, path)


def load_growth(path):
    """Bounded, finite-tensor loader; optimizer state is not reconstructed."""
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError('checkpoint must be a regular file within 32 MiB')
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            raw = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if len(raw) > MAX_CHECKPOINT_BYTES:
            raise ValueError('decompressed checkpoint exceeds 32 MiB')
        saved = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    else:
        saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('incompatible growth architecture')
    with torch.random.fork_rng(devices=[]):
        model = GrowthPatchNetwork(GrowthConfig(**saved['config']))
    state, expected = saved['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('checkpoint tensor inventory mismatch')
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid checkpoint tensor: ' + name)
    model.load_state_dict(state)
    report = saved.get('initialization_report')
    if report is not None and not isinstance(report, dict):
        raise ValueError('initialization_report must be a dict or None')
    model.initialization_report = report
    return model


class GrowthPredictor(RelationPredictor):
    """The inherited expected-byte edit/STOP contract, with no hidden history."""

