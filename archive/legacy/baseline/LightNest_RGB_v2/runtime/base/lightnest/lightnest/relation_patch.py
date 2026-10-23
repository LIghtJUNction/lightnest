"""Instruction graph patch experiment for the straight-line LightNest RV32 VM.

The model receives bytes, instruction fields, and ordinary last-definition/use
edges.  It receives neither execution results nor the desired patch.  EBREAK's
implicit x10 read is the documented return convention of this repository's VM.
The local control has exactly the same learned tensors and uses self edges only.

This is a separate architecture.  It is not accepted by the native byte-CNN
packers.  The graph parser does not model branches, calls, or memory aliasing.
The mechanism is informed by program graphs (Allamanis et al., ICLR 2018) and
GraphCodeBERT (Guo et al., ICLR 2021), not a reproduction of either model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import stat
import time

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'rv32-definition-use-patch-v1'
EBREAK = 0x00100073


@dataclass(frozen=True)
class RelationConfig:
    mode: str = 'dataflow'
    width: int = 64
    command_capacity: int = 384
    binary_capacity: int = 1024
    message_steps: int = 4

    def __post_init__(self):
        if self.mode not in ('local', 'dataflow'):
            raise ValueError('mode must be local or dataflow')
        if type(self.width) is not int or self.width not in (32, 48, 64, 96, 128):
            raise ValueError('unsupported relation width')
        if type(self.command_capacity) is not int or not 1 <= self.command_capacity <= 384:
            raise ValueError('command capacity must be in 1..384')
        if (type(self.binary_capacity) is not int or not 4 <= self.binary_capacity <= 1024
                or self.binary_capacity % 4):
            raise ValueError('binary capacity must be a multiple of four in 4..1024')
        if type(self.message_steps) is not int or not 1 <= self.message_steps <= 16:
            raise ValueError('message steps must be in 1..16')


@dataclass(frozen=True)
class DefinitionUseGraph:
    """Edges are (consumer instruction, last defining instruction).

    No transitive closure, target reachability, or instruction priority is
    computed.  Register zero has no defining node.  Reads precede writes, so
    overwriting a register links the old value only when the instruction uses it.
    """
    instruction_count: int
    edges: tuple[tuple[int, int], ...]


def definition_use_graph(binary: bytes) -> DefinitionUseGraph:
    if not isinstance(binary, bytes) or not binary or len(binary) % 4:
        raise ValueError('a nonempty four-byte-aligned instruction stream is required')
    last: dict[int, int] = {}
    edges: set[tuple[int, int]] = set()
    words = [int.from_bytes(binary[i:i + 4], 'little') for i in range(0, len(binary), 4)]
    for i, word in enumerate(words):
        opcode = word & 127
        rd, rs1, rs2 = (word >> 7) & 31, (word >> 15) & 31, (word >> 20) & 31
        funct3, funct7 = (word >> 12) & 7, (word >> 25) & 127
        write = None
        if word == EBREAK:
            if i != len(words) - 1:
                raise ValueError('instructions after EBREAK are outside the straight-line model')
            uses = (10,)
        elif opcode == 0x33:
            allowed = ((funct7 == 0 and 0 <= funct3 <= 7)
                       or (funct7 == 32 and funct3 in (0, 5))
                       or (funct7 == 1 and funct3 == 0))
            if not allowed:
                raise ValueError('unsupported RV32 register operation')
            uses, write = (rs1, rs2), rd
        elif opcode == 0x13:
            allowed = (funct3 == 0 or (funct3 == 1 and funct7 == 0)
                       or (funct3 == 5 and funct7 in (0, 32)))
            if not allowed:
                raise ValueError('unsupported RV32 immediate operation')
            uses, write = (rs1,), rd
        elif opcode == 0x03 and funct3 == 2:
            uses, write = (rs1,), rd
        else:
            raise ValueError('only the straight-line RV32 VM arithmetic/load subset is supported')
        for register in uses:
            if register and register in last:
                edges.add((i, last[register]))
        if write:
            last[write] = i
    return DefinitionUseGraph(len(words), tuple(sorted(edges)))


def encode_batch(commands, binaries, config=RelationConfig()):
    """Return command tokens, byte tokens, and two normalized edge matrices.

    Padding is dynamic and right aligned.  Channel zero follows definitions;
    channel one follows uses.  Each includes self edges.  Both modes receive the
    same encoding; the local model removes cross-instruction edges in forward.
    """
    if not commands or len(commands) != len(binaries):
        raise ValueError('nonempty matching command and binary batches are required')
    raws = []
    graphs = []
    for command, binary in zip(commands, binaries):
        if not isinstance(command, str):
            raise ValueError('command must be a string')
        raw = command.encode('utf-8')
        if not 1 <= len(raw) <= config.command_capacity:
            raise ValueError('command is empty or exceeds UTF-8 byte capacity')
        if not isinstance(binary, bytes) or not 4 <= len(binary) <= config.binary_capacity:
            raise ValueError('binary is empty or exceeds byte capacity')
        raws.append(raw)
        graphs.append(definition_use_graph(binary))
    cw, bw = max(map(len, raws)), max(map(len, binaries))
    c = torch.tensor([[x + 1 for x in raw] + [0] * (cw - len(raw)) for raw in raws], dtype=torch.long)
    b = torch.tensor([[x + 1 for x in raw] + [0] * (bw - len(raw)) for raw in binaries], dtype=torch.long)
    n = bw // 4
    relations = torch.zeros(len(binaries), 2, n, n, dtype=torch.float32)
    for j, graph in enumerate(graphs):
        ids = torch.arange(graph.instruction_count)
        relations[j, :, ids, ids] = 1.
        if graph.edges:
            consumers, definitions = zip(*graph.edges)
            relations[j, 0, list(consumers), list(definitions)] = 1.
            relations[j, 1, list(definitions), list(consumers)] = 1.
    relations /= relations.sum(-1, keepdim=True).clamp_min(1.)
    return c, b, relations


class RelationPatchNetwork(nn.Module):
    architecture = ARCHITECTURE
    native_packer_compatible = False

    def __init__(self, config=RelationConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.command_embedding = nn.Embedding(257, 24, padding_idx=0)
        self.command_conv1 = nn.Conv1d(24, 48, 5, padding=2)
        self.command_conv2 = nn.Conv1d(48, 48, 5, padding=2)
        self.command_fc = nn.Linear(48 * 6, d)
        self.byte_embedding = nn.Embedding(257, 16, padding_idx=0)
        self.opcode_embedding = nn.Embedding(128, 16)
        self.funct3_embedding = nn.Embedding(8, 8)
        self.funct7_embedding = nn.Embedding(128, 16)
        self.instruction_fusion = nn.Linear(104 + d, d)
        self.definition_message = nn.Linear(d, d, bias=False)
        self.use_message = nn.Linear(d, d, bias=False)
        self.message_norm = nn.LayerNorm(d)
        self.update = nn.GRUCell(d, d)
        self.byte_slots = nn.Embedding(4, 8)
        self.byte_fusion = nn.Linear(d + 16 + 8, d)
        self.pointer = nn.Linear(d, 1)
        self.replacement = nn.Linear(d, 8)
        self.stop_fusion = nn.Linear(d * 3, d)
        self.stop = nn.Linear(d, 1)

    @property
    def none_index(self):
        return self.config.binary_capacity

    def _command(self, command):
        mask = command.ne(0)[:, None, :]
        h = F.gelu(self.command_conv1(self.command_embedding(command).transpose(1, 2)))
        h = h.masked_fill(~mask, 0.)
        h = F.gelu(self.command_conv2(h)).masked_fill(~mask, 0.)
        lengths = command.ne(0).sum(1).tolist()
        pooled = torch.cat([F.adaptive_max_pool1d(h[j:j + 1, :, :length], 6)
                            for j, length in enumerate(lengths)], 0)
        return torch.tanh(self.command_fc(pooled.flatten(1)))

    def _validate(self, command, binary, relations, message_steps):
        if type(message_steps) is not int or not 1 <= message_steps <= 16:
            raise ValueError('message steps must be in 1..16')
        for tensor, cap in ((command, self.config.command_capacity), (binary, self.config.binary_capacity)):
            if (tensor.ndim != 2 or tensor.dtype != torch.long or tensor.shape[0] < 1
                    or not 1 <= tensor.shape[1] <= cap):
                raise ValueError('invalid token batch shape, dtype, or capacity')
            if bool(((tensor < 0) | (tensor > 256)).any()) or bool((tensor.ne(0).sum(1) == 0).any()):
                raise ValueError('tokens must be nonempty byte sequences in 0..256')
            if bool(((tensor[:, :-1] == 0) & (tensor[:, 1:] != 0)).any()):
                raise ValueError('padding must be right aligned')
        if binary.shape[1] % 4 or bool((binary.ne(0).sum(1) % 4 != 0).any()):
            raise ValueError('instruction bytes must be four-byte aligned')
        n = binary.shape[1] // 4
        if (command.shape[0] != binary.shape[0] or command.device != binary.device
                or relations.shape != (binary.shape[0], 2, n, n)
                or relations.dtype != torch.float32 or relations.device != binary.device
                or not bool(torch.isfinite(relations).all()) or bool((relations < 0).any())):
            raise ValueError('invalid relation tensor or batch/device mismatch')

    def forward(self, command, binary, relations, message_steps=None, return_states=False):
        steps = self.config.message_steps if message_steps is None else message_steps
        self._validate(command, binary, relations, steps)
        batch, length = binary.shape
        n, d = length // 4, self.config.width
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
        reserved = {'architecture', 'config', 'state_dict'}
        if reserved.intersection(metadata):
            raise ValueError('metadata cannot replace architecture, config, or tensors')
        torch.save(dict(architecture=ARCHITECTURE, config=asdict(self.config),
                        state_dict=self.state_dict(), **metadata), path)


def load_relation(path):
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 32 * 1024**2:
        raise ValueError('checkpoint must be a regular file within 32 MiB')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('incompatible relation architecture')
    model = RelationPatchNetwork(RelationConfig(**saved['config']))
    state, expected = saved['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('checkpoint tensor inventory mismatch')
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid checkpoint tensor: ' + name)
    model.load_state_dict(state)
    return model


class RelationPredictor:
    """One neural next-edit/STOP decision, suitable for the iterative runner.

    A confidence refusal is an empty, non-STOP decision.  It cannot silently
    become a successful STOP.  Neither the step count nor history are features.
    """
    def __init__(self, model, threshold=0., message_steps=None):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be finite and in 0..1')
        self.model = model.eval()
        self.threshold = float(threshold)
        self.message_steps = message_steps

    @torch.inference_mode()
    def __call__(self, request, current, step=0, history=()):
        from .iterative_patch import ByteEdit, PatchDecision
        c, b, relations = encode_batch([request], [current], self.model.config)
        scores, bits = self.model(c, b, relations, message_steps=self.message_steps)
        if not bool(torch.isfinite(scores).all() and torch.isfinite(bits).all()):
            raise ValueError('nonfinite neural output')
        probabilities = scores.softmax(-1)[0]
        position = int(probabilities.argmax())
        confidence = float(probabilities[position])
        if position == self.model.none_index:
            return PatchDecision(stop=True, reason='model_stop', confidence=confidence)
        if not 0 <= position < len(current):
            raise ValueError('model selected a padded byte')
        byte_probs = bits[0, position].sigmoid()
        confidence = min(confidence, float(torch.maximum(byte_probs, 1 - byte_probs).min()))
        if confidence < self.threshold:
            return PatchDecision(reason='model_uncertain', confidence=confidence)
        replacement = sum(int(byte_probs[k] >= .5) << k for k in range(8))
        return PatchDecision(edits=(ByteEdit(position, current[position:position + 1], bytes([replacement])),),
                             reason='neural_next_edit', confidence=confidence)


def load_records(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('id'), str) or not isinstance(row.get('command'), str):
            raise ValueError('record id and command must be strings')
        binary = bytes.fromhex(row['binary'])
        edits = row.get('edits')
        if not isinstance(edits, list):
            raise ValueError('record edits must be a list; empty means STOP')
        seen = set()
        for edit in edits:
            position, value = edit.get('offset'), edit.get('value')
            if (type(position) is not int or not 0 <= position < len(binary) or position in seen
                    or type(value) is not int or not 0 <= value < 256 or binary[position] == value):
                raise ValueError('invalid, duplicate, or unchanged target byte')
            seen.add(position)
        if 'target' in row:
            target = bytes.fromhex(row['target'])
            expected = {i: y for i, (x, y) in enumerate(zip(binary, target)) if x != y}
            if len(target) != len(binary) or expected != {e['offset']: e['value'] for e in edits}:
                raise ValueError('target bytes and remaining-edit labels disagree')
        records.append(dict(row, binary_bytes=binary))
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('a nonempty dataset with unique record ids is required')
    return records


def supervised_loss(scores, bits, rows, none_index, byte_weight=2.):
    """Maximize total probability of all valid next edits, or explicit STOP."""
    acceptable = torch.zeros_like(scores, dtype=torch.bool)
    bi, positions, targets = [], [], []
    for j, row in enumerate(rows):
        if not row['edits']:
            acceptable[j, none_index] = True
        for edit in row['edits']:
            p, value = edit['offset'], edit['value']
            acceptable[j, p] = True
            bi.append(j)
            positions.append(p)
            targets.append([(value >> k) & 1 for k in range(8)])
    pointer_loss = -torch.logsumexp(scores.log_softmax(-1).masked_fill(~acceptable, float('-inf')), -1).mean()
    if bi:
        byte_loss = F.binary_cross_entropy_with_logits(bits[bi, positions], torch.tensor(targets, dtype=bits.dtype, device=bits.device))
    else:
        byte_loss = bits.sum() * 0.
    return pointer_loss + byte_weight * byte_loss, pointer_loss, byte_loss


@torch.inference_mode()
def evaluate_records(model, rows, message_steps=None, batch_size=32):
    """Public one-step development metric.  It is not episode or VM success."""
    model.eval()
    groups = {}
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        encoded = encode_batch([r['command'] for r in group], [r['binary_bytes'] for r in group], model.config)
        scores, bits = model(*encoded, message_steps=message_steps)
        positions = scores.argmax(-1).tolist()
        for j, row in enumerate(group):
            p = positions[j]
            selected = next((e for e in row['edits'] if e['offset'] == p), None)
            stop_ok = not row['edits'] and p == model.none_index
            byte = None if p == model.none_index else sum(int(bits[j, p, k] >= 0) << k for k in range(8))
            exact = stop_ok or (selected is not None and byte == selected['value'])
            for category in ('all', row.get('category', 'uncategorized')):
                counter = groups.setdefault(category, dict(total=0, edit_total=0, stop_total=0, exact=0,
                                                          correct_position=0, correct_stop=0, false_edit=0))
                counter['total'] += 1
                counter['edit_total'] += bool(row['edits'])
                counter['stop_total'] += not row['edits']
                counter['exact'] += bool(exact)
                counter['correct_position'] += selected is not None
                counter['correct_stop'] += stop_ok
                counter['false_edit'] += not row['edits'] and p != model.none_index
    return groups


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_run(train_path, development_path, output, *, config=RelationConfig(), seed=104729,
              steps=800, batch_size=16, learning_rate=.001, byte_weight=2., threads=2):
    if (type(steps) is not int or not 1 <= steps <= 10000 or type(batch_size) is not int
            or not 1 <= batch_size <= 128 or threads not in (1, 2)):
        raise ValueError('invalid bounded training budget')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('refuse to overwrite an existing experiment directory')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = load_records(train_path)
    development = load_records(development_path)
    if len(rows) < batch_size:
        raise ValueError('training dataset is smaller than a batch')
    model = RelationPatchNetwork(config)
    model.save(out / 'initial.pt')
    source = Path(__file__).read_bytes()
    (out / 'relation_patch_source.py').write_bytes(source)
    protocol = dict(architecture=ARCHITECTURE, config=asdict(config), seed=seed, steps=steps,
                    batch_size=batch_size, threads=threads, learning_rate=learning_rate,
                    byte_weight=byte_weight, optimizer='AdamW', weight_decay=.01, gradient_clip=2.,
                    sampler='random.Random(seed).sample, repeated sampling across updates; same order across modes',
                    pointer_loss='negative log probability mass of any acceptable next edit, or STOP',
                    byte_loss='mean BCE of absolute bits at every acceptable next edit',
                    selection='final fixed-update checkpoint only', train_records=len(rows),
                    development_records=len(development), train_sha256=_sha256(train_path),
                    development_sha256=_sha256(development_path), source_sha256=hashlib.sha256(source).hexdigest(),
                    initial_checkpoint_sha256=_sha256(out / 'initial.pt'),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    training_kind='supervised assembler labels; not autonomous architecture improvement')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    losses, order = [], []
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for step in range(steps):
        ids = rng.sample(range(len(rows)), batch_size)
        order.append(ids)
        group = [rows[i] for i in ids]
        encoded = encode_batch([r['command'] for r in group], [r['binary_bytes'] for r in group], config)
        model.train()
        scores, bits = model(*encoded, message_steps=config.message_steps)
        loss, pointer_loss, byte_loss = supervised_loss(scores, bits, group, model.none_index, byte_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            row = dict(step=step + 1, loss=float(loss.detach()), pointer_loss=float(pointer_loss.detach()),
                       byte_loss=float(byte_loss.detach()), wall_seconds=time.perf_counter() - start_wall)
            losses.append(row)
            print(json.dumps(row), flush=True)
    train_wall, train_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    model.save(out / 'final.pt', training_steps=steps, seed=seed)
    torch.save(dict(optimizer=optimizer.state_dict(), sampler_state=rng.getstate(),
                    torch_rng_state=torch.get_rng_state(), steps=steps), out / 'optimizer-state.pt')
    order_bytes = json.dumps(order, separators=(',', ':')).encode()
    (out / 'training-order.json').write_bytes(order_bytes)
    restored = load_relation(out / 'final.pt').eval()
    metrics = evaluate_records(restored, development, message_steps=config.message_steps)
    report = dict(protocol=protocol, complete=True, training_steps=steps, training_wall_seconds=train_wall,
                  training_cpu_seconds=train_cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  checkpoint_sha256=_sha256(out / 'final.pt'), training_order_sha256=hashlib.sha256(order_bytes).hexdigest(),
                  losses=losses, development_one_step=metrics,
                  limitations=['Straight-line register data flow only; no branches, calls, or memory aliasing.',
                               'The instruction decoder and graph design are human engineering.',
                               'No native packer, default predictor promotion, or autonomous child was produced by this run.',
                               'One-step development accuracy is not complete multi-edit execution success.'])
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('protocol', 'losses')}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--development', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('local', 'dataflow'), default='dataflow')
    parser.add_argument('--seed', type=int, default=104729)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    train_run(args.train, args.development, args.output,
              config=RelationConfig(mode=args.mode, width=args.width), seed=args.seed,
              steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
