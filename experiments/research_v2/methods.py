"""Label-free predictors for restricted RV32 ADD/SUB transformations.

The symbolic and dense-mask paths have an explicit 12-command grammar and a
fixed typed patch operation (flip the ADD/SUB function bit). They do NOT learn
free-form language, arbitrary replacement bytes, or general program repair.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from . import BASE
from rgb.runtime import common, configure_torch
from rgb.network import load_rgb, RGBNetwork
from rgb.scripted_core import LoopCore, compile_rgb
from study.evaluate import action, Live, score_episode

COMMANDS: dict[str, int | None] = {}
for direction in (0, 1):
    src, dst = ('add', 'sub') if direction == 0 else ('sub', 'add')
    for text in (f'change {src} to {dst}',
                 f'replace {src} with {dst} in the returned computation',
                 f'change the output-producing {src} to {dst}',
                 f'change every {src} that contributes to the output to {dst}'):
        COMMANDS[text] = direction
    for text in (f'do not change {src} to {dst}',
                 f'do not replace {src} with {dst} in the returned computation'):
        COMMANDS[text] = None


def validate(row: dict) -> tuple[bytes, int | None]:
    if set(row) != {'command', 'binary'}:
        raise ValueError('Predictor accepts command and binary only, not answers or group IDs.')
    if row['command'] not in COMMANDS:
        raise ValueError('Unsupported request; only the documented 12 exact commands are accepted.')
    if not isinstance(row['binary'], str):
        raise ValueError('Program must be a hex string.')
    code = bytes.fromhex(row['binary'])
    if not code or len(code) % 4 or len(code) > 1024 or code[-4:] != bytes.fromhex('73001000'):
        raise ValueError('Program must be 1..256 RV32 instructions ending in EBREAK.')
    # The inherited straight-line fast parser admits computed loads, whereas
    # the branch/store parser rejects them. Enforce one boundary here first.
    for offset in range(0, len(code), 4):
        word = int.from_bytes(code[offset:offset+4], 'little')
        op = word & 127
        if op in (3, 0x23):
            if (word >> 15) & 31:
                raise ValueError('Computed memory addresses are outside this experiment.')
            immediate = word >> 20 if op == 3 else ((word >> 25) << 5) | ((word >> 7) & 31)
            address = common.sx(immediate, 12)
            if address % 4 or not (0 if op == 3 else 64) <= address < 256:
                raise ValueError('Memory access outside declared input/scratch area.')
    common.graph(code)
    return code, COMMANDS[row['command']]


def candidates(code: bytes, direction: int | None) -> list[int]:
    if direction is None:
        return []
    indices = []
    for i in range(len(code) // 4):
        word = int.from_bytes(code[4*i:4*i+4], 'little')
        if (word & 127 == 0x33 and (word >> 12) & 7 == 0
                and word >> 25 == 32 * direction and (word >> 7) & 31):
            indices.append(i)
    return indices


def patch_action(code: bytes, index: int, rounds: int = 0) -> dict:
    pos = 4 * index + 3
    return {'position': pos, 'value': code[pos] ^ 64, 'stop': False,
            'blue': False, 'rounds': rounds}


def stop_action(rounds: int = 0) -> dict:
    return {'position': 1024, 'value': None, 'stop': True, 'blue': False, 'rounds': rounds}


class BackwardSlice:
    """Static graph reachability baseline, NOT a trained neural network.

    It inherits the parser's conservative reaching-definition/control edges.
    Reachability does not prove semantic influence or arbitrary code safety.
    """
    def plan(self, rows: list[dict]) -> list[list[dict]]:
        plans = []
        for row in rows:
            code, direction = validate(row)
            predecessors = defaultdict(list)
            for user, definition in common.graph(code):
                predecessors[user].append(definition)
            seen: set[int] = set()
            pending = [len(code) // 4 - 1]
            while pending:
                node = pending.pop()
                if node not in seen:
                    seen.add(node)
                    pending.extend(predecessors[node])
            selected = [i for i in candidates(code, direction) if i in seen]
            plans.append([patch_action(code, i) for i in selected] + [stop_action()])
        return plans


class FrozenFeatures:
    def __init__(self, seed: int):
        self.seed = seed
        self.rgb = load_rgb(BASE / f'models/rgb-{seed}.pt')
        self.red = self.rgb.red
        self.loop = torch.jit.script(LoopCore(self.red).eval())
        self.red.eval()

    @torch.no_grad()
    def extract(self, rows: list[dict], depths: tuple[int, ...] = (16, 64)):
        typed = [validate(r) for r in rows]
        inputs, relations, context = self.red.encode_inputs(*common.encode(rows, self.red.config))
        h = inputs
        previous = 0
        out = {}
        for depth in depths:
            h = self.loop(h, inputs, relations, context.valid, depth - previous)
            packed = []
            for j, (code, direction) in enumerate(typed):
                ids = candidates(code, direction)
                index = torch.tensor(ids, dtype=torch.long)
                raw = torch.tensor([list(code[4*i:4*i+4]) for i in ids], dtype=torch.float32).reshape(-1, 4) / 255
                position = index.to(torch.float32)[:, None] / 256
                x = torch.cat((h[j, index], context.command[j].expand(len(ids), -1), raw, position), dim=1)
                assert x.shape == (len(ids), 133)
                packed.append((x.clone(), ids))
            out[depth] = packed
            previous = depth
        return out


class DenseEditHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(133, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer('mean', torch.zeros(133))
        self.register_buffer('scale', torch.ones(133))

    def forward(self, x):
        return self.net(((x-self.mean)/self.scale).clamp(-20, 20)).squeeze(-1)


class DensePlanner:
    """One frozen neural pass labels ALL typed edit locations simultaneously.

    STOP follows the precomputed edit list; this deliberately replaces the old
    independent STOP classifier. No target or execution feedback is consulted.
    Changing ADD to SUB preserves syntactic edges, but NOT neural hidden states;
    this is a new one-pass policy, not an exact cache of autoregressive C64.
    """
    def __init__(self, extractor: FrozenFeatures, head: DenseEditHead, depth: int, threshold: float = 0.):
        self.extractor = extractor
        self.head = head.eval()
        self.depth = depth
        self.threshold = threshold

    @torch.no_grad()
    def plan(self, rows):
        features = self.extractor.extract(rows, (self.depth,))[self.depth]
        plans = []
        for row, (x, ids) in zip(rows, features, strict=True):
            code = bytes.fromhex(row['binary'])
            scores = self.head(x).tolist()
            selected = [i for i, score in zip(ids, scores, strict=True) if score > self.threshold]
            acts = [patch_action(code, i, self.depth if k == 0 else 0) for k, i in enumerate(selected)]
            acts.append(stop_action(self.depth if not acts else 0))
            plans.append(acts)
        return plans


def evaluate_plans(planner, episodes, batch=16, score=True):
    """Same <=8 edits and explicit final STOP contract as the old evaluator."""
    records = []
    for start in range(0, len(episodes), batch):
        eps = episodes[start:start+batch]
        plans = planner.plan([{'command':r['command'], 'binary':r['binary']} for r in eps])
        for ep, plan in zip(eps, plans, strict=True):
            state = {'code':bytes.fromhex(ep['binary']), 'actions':[], 'stop':False, 'status':'budget'}
            for step, action_ in enumerate(plan[:9]):
                a = dict(action_, step=step)
                state['actions'].append(a)
                if a['stop']:
                    state['stop'] = True
                    state['status'] = 'stop'
                    break
                if step == 8:
                    break
                pos, value = a['position'], a['value']
                if not 0 <= pos < len(state['code']):
                    state['status'] = 'padded_pointer'
                    break
                a['before'] = state['code'][pos]
                if value == a['before']:
                    state['status'] = 'no_progress'
                    break
                code = bytearray(state['code'])
                code[pos] = value
                state['code'] = bytes(code)
            records.append(score_episode(ep, state) if score else {
                'id':ep['id'], 'final':state['code'].hex(), 'stop':state['stop'], 'actions':state['actions']})
    return records


class StableExit:
    """Original C64 states; three equal discrete actions permit early output.

    Confidence and repeated actions are heuristics, not correctness certificates.
    A value above one explicitly disables early exit and is the safe fallback.
    """
    def __init__(self, extractor: FrozenFeatures, threshold: float):
        self.extractor, self.threshold = extractor, threshold
        self.depths = (16, 24, 32, 48, 64)

    @torch.inference_mode()
    def __call__(self, rows):
        for row in rows:
            if set(row) != {'command', 'binary'}:
                raise ValueError('No labels in inference.')
        red = self.extractor.red
        inputs, rel, ctx = red.encode_inputs(*common.encode(rows, red.config))
        h = inputs
        ids = list(range(len(rows)))
        history = [[] for _ in rows]
        result = [None] * len(rows)
        previous = 0
        for depth in self.depths:
            h = self.extractor.loop(h, inputs, rel, ctx.valid, depth-previous)
            scores, bits = red.decode_state(h, ctx)
            keep = []
            for j, original in enumerate(ids):
                a = action(scores[j], bits[j], red.none_index)
                history[original].append((a['position'], a['value']))
                stable = len(history[original]) >= 3 and len(set(history[original][-3:])) == 1
                if depth == 64 or (stable and a['confidence'] >= self.threshold):
                    result[original] = a | {'blue':depth < 64, 'rounds':depth}
                else:
                    keep.append(j)
            if not keep:
                break
            index = torch.tensor(keep)
            h, inputs, rel = (v.index_select(0, index) for v in (h, inputs, rel))
            ctx = RGBNetwork._slice_context(ctx, index)
            ids = [ids[j] for j in keep]
            previous = depth
        assert all(r is not None for r in result)
        return result


def baseline(seed):
    return Live(compile_rgb(load_rgb(BASE / f'models/rgb-{seed}.pt')), 'red_only')
