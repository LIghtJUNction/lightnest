"""Complete-32-parent adapter with shared multi-round readouts.

The original byte/command encoder, definition/use message matrices, GRU, and
all position, replacement-bit, and STOP heads keep their tensor names and
operations.  There is one shared update F and one shared decoder for every
round.  The default remains the complete 32-round parent, not its first eight
rounds.  Every update already receives the original encoded inputs.

Optional local STOP reads the pointer-softmax-weighted old decoded byte
features through a single zero-initialized linear map (64 new parameters at
width 64).  The pointer weights stay differentiable.  This is an ordinary
softmax pooling residual, not a newly invented attention algorithm.  It uses
no labels, required offsets, register tables, direction flags, execution
outcomes, or target-ancestry masks.  Zero initialization preserves every old
logit while allowing the new linear weights to learn on the first step.

Checkpoints are owned by the experiment driver: record the parent reference,
this ARCHITECTURE/API_VERSION, local_stop_enabled, and complete state_dict;
rebuild with make_model(parent, local_stop=...) then strict-load that state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from lightnest.growth_patch import GrowthConfig, GrowthPatchNetwork, state_fingerprint

ARCHITECTURE = 'rv32-shared-depth-local-stop-v1'
API_VERSION = 1
PARENT_ROUNDS = 32
MAX_ROUNDS = 64


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite shared-depth ' + label)


def _round(value):
    if type(value) is not int or not 1 <= value <= MAX_ROUNDS:
        raise ValueError('rounds must be an integer in 1..64')
    return value


@dataclass(frozen=True)
class ReadoutContext:
    binary: torch.Tensor
    byte_embeddings: torch.Tensor
    slots: torch.Tensor
    command: torch.Tensor
    valid: torch.Tensor


class BreakthroughNetwork(GrowthPatchNetwork):
    """A complete frozen-shape parent plus an optional candidate-local STOP."""
    architecture = ARCHITECTURE
    api_version = API_VERSION

    def __init__(self, config: GrowthConfig, *, local_stop=False):
        if (not isinstance(config, GrowthConfig) or config.variant != 'fixed_extra'
                or config.message_steps != 8 or config.extra_steps != 24):
            raise ValueError('this experiment requires the complete fixed32 parent config')
        if type(local_stop) is not bool:
            raise ValueError('local_stop must be a bool')
        super().__init__(config)
        self.local_stop_enabled = local_stop
        if local_stop:
            self.local_stop_readout = nn.Linear(config.width, 1, bias=False)
            nn.init.zeros_(self.local_stop_readout.weight)

    def parameter_roles(self):
        roles = super().parameter_roles()
        if self.local_stop_enabled:
            roles['local_stop_readout.weight'] = 'output'
        return roles

    def encode_inputs(self, command, binary, relations):
        """Return original inputs, direct relation tensor, and cached decoder inputs."""
        self._validate(command, binary, relations, self.config.message_steps)
        batch, length = binary.shape
        n = length // 4
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
        if self.config.mode == 'local':
            identity = torch.eye(n, device=binary.device)[None] * valid[:, :, None]
            relation = torch.stack((identity, identity), 1)
        else:
            relation = relations
        slots = self.byte_slots(torch.arange(length, device=binary.device) % 4)[None].expand(batch, -1, -1)
        return inputs, relation, ReadoutContext(binary, byte_embeddings, slots, c, valid)

    def shared_step(self, h, inputs, relation, valid):
        """The original F, including input injection on every iteration."""
        batch, n, d = h.shape
        from_definition = torch.bmm(relation[:, 0], self.definition_message(h))
        from_use = torch.bmm(relation[:, 1], self.use_message(h))
        messages = torch.tanh(self.message_norm(inputs + from_definition + from_use))
        proposed = self.update(messages.reshape(-1, d), h.reshape(-1, d)).reshape(batch, n, d)
        proposed = proposed.masked_fill(~valid[..., None], 0.)
        _finite(proposed, 'graph update')
        return proposed.masked_fill(~valid[..., None], 0.)

    def decode_state(self, h, context, *, local_stop_scale=1., return_aux=False):
        """Reuse one old decoder; optional aux tensors are label-free diagnostics."""
        if type(local_stop_scale) not in (int, float) or not math.isfinite(local_stop_scale):
            raise ValueError('local_stop_scale must be a finite scalar')
        binary, byte_embeddings = context.binary, context.byte_embeddings
        valid, c, slots = context.valid, context.command, context.slots
        length = binary.shape[1]
        decoded = torch.tanh(self.byte_fusion(torch.cat((h.repeat_interleave(4, 1), byte_embeddings, slots), -1)))
        pointer = self.pointer(decoded).squeeze(-1).masked_fill(binary.eq(0), -1e9)
        max_h = h.masked_fill(~valid[..., None], -1e9).max(1).values
        mean_h = h.sum(1) / valid.sum(1, keepdim=True)
        old_stop = self.stop(torch.tanh(self.stop_fusion(torch.cat((max_h, mean_h, c), -1))))
        stop = old_stop
        local_delta = None
        if self.local_stop_enabled:
            weights = pointer.masked_fill(binary.eq(0), -torch.inf).softmax(-1)
            candidate = torch.sum(weights[..., None] * decoded, dim=1)
            _finite(candidate, 'local STOP candidate')
            local_delta = self.local_stop_readout(candidate)
            _finite(local_delta, 'local STOP residual')
            stop = old_stop + local_stop_scale * local_delta
        padded_pointer = F.pad(pointer, (0, self.none_index - length), value=-1e9)
        scores = torch.cat((padded_pointer, stop), -1)
        byte_logits = self.replacement(decoded).masked_fill(binary.eq(0)[..., None], 0.)
        bits = F.pad(byte_logits, (0, 0, 0, self.none_index - length))
        if return_aux:
            valid_pointer = pointer.masked_fill(binary.eq(0), -torch.inf)
            aux = dict(edit_logsumexp=valid_pointer.logsumexp(-1),
                       max_edit=valid_pointer.max(-1).values, stop=stop.squeeze(-1),
                       old_stop=old_stop.squeeze(-1), local_delta=local_delta)
            return scores, bits, aux
        return scores, bits

    def forward(self, command, binary, relations, message_steps=None, return_states=False,
                *, rounds=None, readout_rounds=None, local_stop_scale=1.):
        """Single tuple or depth->tuple dict, from one connected shared unroll.

        `rounds=None` defaults to32.  With `return_states=True`, single output
        is (scores,bits,states), preserving the prior API; multiple readouts
        return (readout_dict,states).  States include h0 through h_max_round.
        `local_stop_scale=0` is a frozen-weights residual-off diagnostic.
        """
        if message_steps is not None:
            raise ValueError('use rounds for the complete-parent inference depth')
        multiple = readout_rounds is not None
        if multiple:
            if rounds is not None:
                raise ValueError('choose rounds or readout_rounds, not both')
            if not isinstance(readout_rounds, (tuple, list)) or not readout_rounds:
                raise ValueError('readout_rounds must be a nonempty tuple or list')
            requested = tuple(sorted(_round(value) for value in readout_rounds))
            if len(set(requested)) != len(requested):
                raise ValueError('readout rounds must be unique')
        else:
            requested = (_round(PARENT_ROUNDS if rounds is None else rounds),)
        inputs, relation, context = self.encode_inputs(command, binary, relations)
        h = inputs
        states = [h] if return_states else None
        outputs = {}
        requested_set = set(requested)
        for index in range(1, requested[-1] + 1):
            h = self.shared_step(h, inputs, relation, context.valid)
            if return_states:
                states.append(h)
            if index in requested_set:
                outputs[index] = self.decode_state(h, context, local_stop_scale=local_stop_scale)
        if multiple:
            return (outputs, states) if return_states else outputs
        scores, bits = outputs[requested[0]]
        return (scores, bits, states) if return_states else (scores, bits)

    def save(self, *args, **kwargs):
        # Inherited GrowthPatchNetwork.save hard-codes its old architecture;
        # using that format here would mislabel the additional STOP tensor.
        raise NotImplementedError('use the experiment checkpoint writer with this architecture and complete state_dict')


def make_model(parent, local_stop=False):
    """Copy every complete32 parent tensor without modifying RNG or parent."""
    if type(parent) is not GrowthPatchNetwork:
        raise ValueError('parent must be the frozen-round GrowthPatchNetwork loaded by growth_train')
    if any(value.device.type != 'cpu' for value in parent.parameters()):
        raise ValueError('this bounded experiment factory expects CPU parent tensors')
    parent_state = parent.state_dict()
    before = state_fingerprint(parent)
    with torch.random.fork_rng(devices=[]):
        child = BreakthroughNetwork(parent.config, local_stop=local_stop)
    incompatible = child.load_state_dict(parent_state, strict=False)
    expected_new = ['local_stop_readout.weight'] if local_stop else []
    if sorted(incompatible.missing_keys) != expected_new or incompatible.unexpected_keys:
        raise ValueError('complete parent tensor inventory was not preserved')
    child_state = child.state_dict()
    for name, value in parent_state.items():
        if not torch.equal(value, child_state[name]) or value.data_ptr() == child_state[name].data_ptr():
            raise ValueError('parent tensor copy failed: ' + name)
    if state_fingerprint(parent) != before:
        raise ValueError('factory changed parent tensors')
    child.initialization_report = dict(architecture=ARCHITECTURE, api_version=API_VERSION,
        parent_state_sha256=before, complete_parent_rounds=PARENT_ROUNDS,
        all_parent_tensors_exact=True, inherited_parameter_names=sorted(parent_state),
        new_parameter_names=expected_new, local_stop_enabled=local_stop,
        new_parameter_values=child.config.width if local_stop else 0,
        default32_function_preserving_expected=True, independent_birth_audit_required=True,
        initialization='new local STOP linear weight exactly zero; all parent tensors copied',
        first_step='new output weight can learn immediately; its upstream residual-path gradient starts at zero')
    child.train(parent.training)
    return child
