"""Complete RGB policy computation in a single native WGT forward graph."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import math
from pathlib import Path
import struct

import weightlang as wgt
from .native import _import_gate_state


@dataclass(frozen=True)
class NativeRGBConfig:
    width: int = 64
    chunk_size: int = 64
    heads: int = 4
    landmarks: int = 4
    detail_k: int = 8
    blue_depth: int = 1
    red_depths: tuple = (2, 4, 8, 16)
    note_slots: int = 16

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "red_depths" and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be positive")
        if (
            self.width % self.heads
            or self.width % 8
            or not 8 <= self.chunk_size <= 256
            or not 2 <= self.note_slots <= 256
        ):
            raise ValueError("invalid RGB geometry")
        if (
            not self.red_depths
            or any(type(v) is not int for v in self.red_depths)
            or tuple(sorted(set(self.red_depths))) != self.red_depths
            or self.blue_depth >= self.red_depths[0]
            or self.red_depths[-1] > 64
        ):
            raise ValueError("invalid RGB thinking schedule")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _observation(image, command, frames, chunk_size):
    """Exact byte/layout preparation only; no parameter-dependent computation."""
    if not isinstance(image, bytes) or not image:
        raise ValueError("RGB image must be nonempty bytes")
    if not isinstance(command, str) or not 1 <= len(command.encode()) <= 384:
        raise ValueError("invalid RGB command")
    rows = [("image", image, 0)]
    for frame in frames:
        kind, payload, address = frame.kind, frame.payload, frame.address
        if (
            not isinstance(kind, str)
            or not isinstance(payload, bytes)
            or type(address) is not int
            or address < 0
        ):
            raise ValueError("invalid observation frame")
        rows.append((kind, payload, address))
    payloads = []
    ids = []
    addresses = []
    kinds = []
    lookup = {}
    for kind, payload, address in rows:
        index = (
            int.from_bytes(hashlib.sha256(kind.encode()).digest()[:2], "little") % 32
        )
        for offset in range(0, len(payload), chunk_size) if payload else (0,):
            raw = payload[offset : offset + chunk_size]
            if raw not in lookup:
                lookup[raw] = len(payloads)
                payloads.append(raw)
            ids.append(lookup[raw])
            addresses.append(address + offset)
            kinds.append(index)
            if len(ids) > 1 << 20:
                raise ValueError("observation capacity exceeded")
    raw = [
        [float(value + 1) for value in payload] + [0.0] * (chunk_size - len(payload))
        for payload in payloads
    ]
    features = []
    for address in addresses:
        # Match the original f32 coordinate feature preprocessing. Learned
        # position projection remains in WGT and participates in the full VJP.
        angles = [
            _f32(_f32(address) / scale) for scale in [1.0, 64.0, 4096.0, 262144.0]
        ]
        features.append(
            [_f32(math.sin(a)) for a in angles] + [_f32(math.cos(a)) for a in angles]
        )
    chunks = (len(image) + chunk_size - 1) // chunk_size
    return (
        raw,
        ids,
        features,
        kinds,
        [float(v + 1) for v in command.encode()],
        ids[:chunks],
    )


class NativeRGBPolicy(wgt.Model):
    """Both blue/red readouts, observation and all thought steps in one AD graph.

    Geometry is construction-time, inferred from the example's byte layout;
    call specialize() to change layout. Construction never predicts or trains.
    Source recipes are untrained zeros/ones. Explicitly import checkpoint weights.
    """

    def __init__(
        self, image, command, frames=(), *, config=None, depth=None, **options
    ):
        if config is None:
            config = NativeRGBConfig()
        elif not isinstance(config, NativeRGBConfig):
            config = NativeRGBConfig(
                **(asdict(config) if is_dataclass(config) else config)
            )
        depth = config.red_depths[-1] if depth is None else depth
        if (
            type(depth) is not int
            or not config.blue_depth < depth <= config.red_depths[-1]
        ):
            raise ValueError("depth must advance after blue within configured budget")
        self.config, self.depth = config, depth
        self.training = True
        data = _observation(image, command, frames, config.chunk_size)
        self._geometry = (
            len(data[0]),
            len(data[1]),
            len(data[4]),
            len(image),
            len(data[5]),
        )
        self._source = (
            Path(__file__).with_name("wgt") / "native_rgb_policy" / "model.wgt"
        ).read_text()
        definition = wgt.compile_source(
            self._source, dimensions=self._dimensions(self._geometry, depth)
        )
        options.setdefault("memory_bytes", 256 * 1024**2)
        options.setdefault("work_limit", 20_000_000)
        super().__init__(definition, **options)

    def _dimensions(self, geometry, depth):
        config = self.config
        u, n, cmd, length, t = geometry
        h = config.width
        c = config.chunk_size
        s = config.note_slots
        a = config.heads
        landmarks = config.landmarks
        return dict(
            H=h,
            HB=h // 2,
            H2=2 * h,
            H3=3 * h,
            H4=4 * h,
            H8=h + 8,
            H2P3=2 * h + 3,
            C=c,
            C8=c * 8,
            U=u,
            UC=u * c,
            N=n,
            NH=n * h,
            S=s,
            SP=s + 1,
            SH=s * h,
            NSH=(n + s) * h,
            SW=(n + s) * h + s + 1,
            A=a,
            D=h // a,
            L=landmarks,
            K=min(config.detail_k, n),
            LN=landmarks * n,
            AL=a * landmarks,
            AQ=a * n,
            Q=n,
            J=landmarks + n,
            IMG_CHUNKS=t,
            TC=t * c,
            TC8=t * c * 8,
            LEN=length,
            LEN8=length * 8,
            CMD=cmd,
            BLUE=config.blue_depth,
            RED=depth - config.blue_depth,
            B=len(config.red_depths) + 1,
            STATK=min(2, length),
        )

    def specialize(self, image, command, frames=(), *, depth=None):
        """Rebind data geometry/depth in the same session, preserving weights/Adam."""
        self._ensure_native_process()
        depth = self.depth if depth is None else depth
        if (
            type(depth) is not int
            or not self.config.blue_depth < depth <= self.config.red_depths[-1]
        ):
            raise ValueError("invalid RGB thinking depth")
        data = _observation(image, command, frames, self.config.chunk_size)
        geometry = (len(data[0]), len(data[1]), len(data[4]), len(image), len(data[5]))
        with self._lock:
            if geometry == self._geometry and depth == self.depth:
                return self
            definition = wgt.compile_source(
                self._source, dimensions=self._dimensions(geometry, depth)
            )
            result = self._call("RebindDefinition", source=definition["source"])
            definition["forward_interface"] = result["forward_interface"]
            self._definition = definition
            self.execution_backend = result.get("backend")
            self._geometry, self.depth = geometry, depth
            return self

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError("mode must be bool")
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def pack(self, image, command, frames=(), *, mode="routed"):
        if mode not in ["routed", "compressed", "dense"]:
            raise ValueError("invalid attention mode")
        raw, ids, address, kinds, tokens, image_ids = _observation(
            image, command, frames, self.config.chunk_size
        )
        geometry = (len(raw), len(ids), len(tokens), len(image), len(image_ids))
        if geometry != self._geometry:
            raise ValueError(
                "RGB input geometry changed; call specialize() before packing"
            )
        n = len(ids)
        c = self.config.chunk_size
        landmarks = self.config.landmarks
        h = self.config.width
        pooling = []
        for i in range(landmarks):
            start, end = i * n // landmarks, (i + 1) * n // landmarks
            pooling.append(
                [float(start <= j < end) / max(end - start, 1) for j in range(n)]
            )
        return (
            raw,
            [float(i) for i in ids],
            address,
            [float(i) for i in kinds],
            tokens,
            [float(i) for i in image_ids],
            [float(i) for i in range(len(image_ids)) for _ in range(c)],
            [float(i) for _ in image_ids for i in range(c)],
            [1.0] * n,
            pooling,
            [float(mode == m) for m in ["routed", "compressed", "dense"]],
            [float(not self.training)],
            [(h // self.config.heads) ** -0.5],
            [h**-0.5],
            [[0.0] * h for _ in range(self.config.note_slots)],
            [float(len(image) == 1)],
        )

    def forward(self, image, command, frames=(), *, mode="routed"):
        with self._lock:
            depth = self.depth
            values = wgt.Model.forward(
                self, *self.pack(image, command, frames, mode=mode)
            )
        blue = dict(
            zip(
                [
                    "logits",
                    "bits",
                    "risk_logit",
                    "language_logits",
                    "budget_logits",
                    "context_budget_logits",
                ],
                values[:6],
            )
        )
        red = dict(
            zip(
                [
                    "logits",
                    "bits",
                    "risk_logit",
                    "budget_logits",
                    "context_budget_logits",
                ],
                values[6:],
            )
        )
        blue.update(
            origin="blue",
            depth=self.config.blue_depth,
            risk_logit=blue["risk_logit"][0],
        )
        red.update(
            origin="red",
            depth=depth,
            risk_logit=red["risk_logit"][0],
            language_logits=None,
        )
        return blue, red

    def import_state_dict(self, state):
        direct = {
            "green_byte_weight": "green.byte.weight",
            "green_encode_weight": "green.encode_chunk.0.weight",
            "green_encode_bias": "green.encode_chunk.0.bias",
            "green_encode_norm_weight": "green.encode_chunk.2.weight",
            "green_encode_norm_bias": "green.encode_chunk.2.bias",
            "green_kind_weight": "green.kind.weight",
            "green_position_weight": "green.position.weight",
            "command_weight": "command.weight",
            "depth_weight": "depth.weight",
            "initial_weight": "initial.weight",
            "initial_bias": "initial.bias",
            "message_weight": "message.weight",
            "message_bias": "message.bias",
            "thought_norm_weight": "thought_norm.weight",
            "thought_norm_bias": "thought_norm.bias",
        }
        for name in ["query", "key", "value", "condition", "output"]:
            direct["mem_" + name + "_weight"] = "green.memory." + name + ".weight"
        direct["mem_landmark_bias"] = "green.memory.landmark_bias"
        direct.update(
            {
                "nb_slot_keys": "notebook.slot_keys",
                "nb_key_weight": "notebook.key.weight",
                "nb_query_weight": "notebook.query.weight",
            }
        )
        for native, source in [
            ("value0", "value.0"),
            ("value2", "value.2"),
            ("write_gate0", "write_gate.0"),
            ("write_gate2", "write_gate.2"),
            ("budget0", "budget.0"),
            ("budget2", "budget.2"),
            ("budget3", "budget.3"),
        ]:
            for field in ["weight", "bias"]:
                direct["nb_" + native + "_" + field] = (
                    "notebook." + source + "." + field
                )
        for prefix in ["blue", "red"]:
            for native, source in [
                ("slot", "slot"),
                ("decode", "decode.0"),
                ("norm", "decode.2"),
                ("edit", "edit"),
                ("bits", "value"),
            ]:
                for field in ["weight"] if native == "slot" else ["weight", "bias"]:
                    direct[prefix + "_" + native + "_" + field] = (
                        prefix + "." + source + "." + field
                    )
        for native, source in [
            ("red_to_blue", "red_to_blue"),
            ("red_abstraction", "red_abstraction.0"),
            ("abstraction_norm", "red_abstraction.2"),
            ("abstract_to_blue", "abstract_to_blue"),
            ("blue_language", "blue_language"),
            ("budget0", "budget.0"),
            ("budget_norm", "budget.2"),
            ("budget3", "budget.3"),
            ("risk0", "risk.0"),
            ("risk2", "risk.2"),
        ]:
            for field in ["weight", "bias"]:
                direct[native + "_" + field] = source + "." + field
        return _import_gate_state(self, state, direct, {"thought": ("thought", None)})


def load_rgb_policy(
    image, command, frames=(), *, config=None, depth=None, backend="native", **options
):
    if backend == "native":
        return NativeRGBPolicy(
            image, command, frames, config=config, depth=depth, **options
        )
    if backend == "torch":
        from .model import RGBConfig, RGBPolicy

        if options or depth is not None:
            raise TypeError("Torch runtime options/depth belong to host/forward APIs")
        if isinstance(config, dict):
            config = RGBConfig(**config)
        config = (
            RGBConfig(**asdict(config))
            if isinstance(config, NativeRGBConfig)
            else (config or RGBConfig())
        )
        return RGBPolicy(config)
    raise ValueError("backend must be 'native' or 'torch'")
