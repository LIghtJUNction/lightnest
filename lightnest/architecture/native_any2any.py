"""Native Any2Any default byte/GRU/Transformer/notebook/readout computation."""

from dataclasses import dataclass, asdict, is_dataclass
import hashlib
from pathlib import Path
import secrets

import weightlang as wgt
from .native import _import_named_state

_KEEP_DEPTH = object()


@dataclass(frozen=True)
class NativeArtifactRequest:
    kind: str
    operation: str
    specification: str = ""

    def __post_init__(self):
        _request(self)


@dataclass(frozen=True)
class NativeArtifactConfig:
    width: int = 128
    heads: int = 4
    max_bytes: int = 1 << 20
    red_depths: tuple = (2, 4, 8, 16)
    sequence_limit: int = 4096
    note_slots: int = 32
    request_encoding: str = "gru"
    semantic_chunk_size: int = 0
    semantic_bpe: object = None

    def __post_init__(self):
        if (
            any(
                type(v) is not int or v < 1
                for v in [
                    self.width,
                    self.heads,
                    self.max_bytes,
                    self.sequence_limit,
                    self.note_slots,
                ]
            )
            or self.width < 32
            or self.width % self.heads
            or self.max_bytes < 256
            or self.sequence_limit < 256
            or not 2 <= self.note_slots <= 256
        ):
            raise ValueError("invalid native ArtifactCore geometry")
        if (
            not self.red_depths
            or any(type(v) is not int for v in self.red_depths)
            or tuple(sorted(set(self.red_depths))) != tuple(self.red_depths)
            or self.red_depths[0] <= 1
        ):
            raise ValueError("invalid thinking schedule")
        if (
            self.request_encoding != "gru"
            or self.semantic_chunk_size
            or self.semantic_bpe is not None
        ):
            raise NotImplementedError(
                "native ArtifactCore currently supports GRU requests and the default byte path"
            )


def _request(request):
    kind, operation, specification = (
        request.kind,
        request.operation,
        request.specification,
    )
    if (
        not isinstance(kind, str)
        or not 1 <= len(kind) <= 128
        or not isinstance(operation, str)
        or not operation
        or len(operation.encode()) > 512
        or not isinstance(specification, str)
        or len(specification.encode()) > 2048
    ):
        raise ValueError("invalid artifact request")
    kinds = (
        "raw",
        "text",
        "json",
        "csv",
        "elf",
        "pe",
        "macho",
        "archive",
        "compressed",
        "image",
        "audio",
        "pdf",
        "sqlite",
        "bytecode",
    )
    index = (
        kinds.index(kind)
        if kind in kinds
        else 32
        + int.from_bytes(hashlib.sha256(kind.encode()).digest()[:2], "little") % 224
    )
    return index, [v + 1 for v in (operation + "\0" + specification).encode()]


class NativeArtifactCore(wgt.Model):
    """One native AD graph, including all four independent Transformer layers."""

    def __init__(
        self,
        raw,
        request,
        *,
        config=None,
        depth=None,
        chunk_size=256,
        max_chunks=16,
        **options,
    ):
        self.config = (
            config
            if isinstance(config, NativeArtifactConfig)
            else NativeArtifactConfig(
                **(asdict(config) if is_dataclass(config) else config or {})
            )
        )
        if (
            type(chunk_size) is not int
            or not 32 <= chunk_size <= self.config.sequence_limit
            or type(max_chunks) is not int
            or max_chunks < 2
        ):
            raise ValueError("invalid chunk layout")
        self.chunk_size, self.max_chunks = chunk_size, max_chunks
        self.training = True
        self.depth = depth
        self._source = (
            Path(__file__).with_name("wgt") / "native_any2any" / "model.wgt"
        ).read_text()
        self._geometry = self._layout(raw, request)[0]
        definition = wgt.compile_source(
            self._source, dimensions=self._dimensions(self._geometry, depth)
        )
        options.setdefault("memory_bytes", 256 * 1024**2)
        options.setdefault("work_limit", 100_000_000)
        super().__init__(definition, **options)

    def _layout(self, raw, request):
        if not isinstance(raw, bytes) or not 0 < len(raw) <= self.config.max_bytes:
            raise ValueError("invalid artifact bytes")
        starts = list(range(0, len(raw), self.chunk_size))
        if len(starts) > self.max_chunks:
            starts = [
                starts[int(i * (len(starts) - 1) / (self.max_chunks - 1))]
                for i in range(self.max_chunks)
            ]
        chunks = [raw[i : i + self.chunk_size] for i in starts]
        kind, tokens = _request(request)
        return (len(chunks), max(map(len, chunks)), len(tokens)), chunks, kind, tokens

    def _dimensions(self, geometry, depth):
        depth = self.config.red_depths[-1] if depth is None else depth
        if type(depth) is not int or not 1 <= depth <= self.config.red_depths[-1]:
            raise ValueError("invalid thinking depth")
        g, t, r = geometry
        h = self.config.width
        s = self.config.note_slots
        return dict(
            H=h,
            H2=2 * h,
            H4=4 * h,
            A=self.config.heads,
            D=h // self.config.heads,
            SEQ=t,
            G=g,
            GT=g * t,
            GG=g * g,
            GH=g * h,
            RQ=r,
            P=self.config.sequence_limit,
            S=s,
            SP=s + 1,
            SH=s * h,
            N=g,
            NH=g * h,
            DEPTH=depth,
            B=len(self.config.red_depths) + 1,
        )

    def train(self, mode=True):
        if type(mode) is not bool:
            raise TypeError("training must be bool")
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def specialize(self, raw, request, *, depth=_KEEP_DEPTH):
        self._ensure_native_process()
        depth = self.depth if depth is _KEEP_DEPTH else depth
        geometry = self._layout(raw, request)[0]
        with self._lock:
            dimensions = self._dimensions(geometry, depth)
            if geometry == self._geometry and depth == self.depth:
                return self
            definition = wgt.compile_source(self._source, dimensions=dimensions)
            result = self._call("RebindDefinition", source=definition["source"])
            definition["forward_interface"] = result["forward_interface"]
            self._definition = definition
            self.execution_backend = result.get("backend")
            self._geometry, self.depth = geometry, depth
        return self

    def pack(self, raw, request, *, seed=None):
        geometry, chunks, kind, tokens = self._layout(raw, request)
        if geometry != self._geometry:
            raise ValueError("artifact layout changed; call specialize first")
        if seed is None:
            seed = secrets.randbits(32) if self.training else 0
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be uint32")
        g, t, _ = geometry
        return (
            [[*chunk, *([0] * (t - len(chunk)))] for chunk in chunks],
            list(range(t)),
            [kind],
            tokens,
            [[float(i < len(chunk)) for i in range(t)] for chunk in chunks],
            [[float(i == j) for j in range(g)] for i in range(g)],
            [seed & 65535, seed >> 16, float(self.training)],
            [float(not self.training)],
            [[0.0] * self.config.width for _ in range(self.config.note_slots)],
            [float(self.depth is None)],
            [
                [
                    float(i < d)
                    for i in range(
                        self.config.red_depths[-1] if self.depth is None else self.depth
                    )
                ]
                for d in (1, *self.config.red_depths)
            ],
        )

    def forward(self, raw, request, *, seed=None):
        with self._lock:
            values = wgt.Model.forward(self, *self.pack(raw, request, seed=seed))
        return dict(
            zip(
                [
                    "encoded",
                    "abstract",
                    "guidance",
                    "blue_logits",
                    "red_logits",
                    "stop_logits",
                    "kind_logits",
                    "summary",
                    "budget_logits",
                    "copy_scores",
                    "notes",
                    "context",
                    "context_budget_logits",
                ],
                values,
            )
        )

    def import_state_dict(self, state):
        return _import_named_state(self, state, _STATE_NAMES)


def load_artifact_core(raw, request, *, backend="native", config=None, **options):
    if backend == "native":
        return NativeArtifactCore(raw, request, config=config, **options)
    if backend == "torch":
        from .any2any import ArtifactCore

        if options:
            raise TypeError("Torch runtime/depth options belong to host APIs")
        return ArtifactCore(
            **(asdict(config) if is_dataclass(config) else config or {})
        )
    raise ValueError("backend must be native or torch")


# Exact source parameter identities; packed QKV and GRU rows split in native storage.
_STATE_NAMES = {
    "byte_weight": "byte.weight",
    "kind_weight": "kind.weight",
    "position_weight": "position.weight",
    "enc0_q_weight": ("encoder.layers.0.self_attn.in_proj_weight", 0),
    "enc0_q_bias": ("encoder.layers.0.self_attn.in_proj_bias", 0),
    "enc0_k_weight": ("encoder.layers.0.self_attn.in_proj_weight", 1),
    "enc0_k_bias": ("encoder.layers.0.self_attn.in_proj_bias", 1),
    "enc0_v_weight": ("encoder.layers.0.self_attn.in_proj_weight", 2),
    "enc0_v_bias": ("encoder.layers.0.self_attn.in_proj_bias", 2),
    "enc0_out_weight": "encoder.layers.0.self_attn.out_proj.weight",
    "enc0_out_bias": "encoder.layers.0.self_attn.out_proj.bias",
    "enc0_ff1_weight": "encoder.layers.0.linear1.weight",
    "enc0_ff1_bias": "encoder.layers.0.linear1.bias",
    "enc0_ff2_weight": "encoder.layers.0.linear2.weight",
    "enc0_ff2_bias": "encoder.layers.0.linear2.bias",
    "enc0_n1_weight": "encoder.layers.0.norm1.weight",
    "enc0_n1_bias": "encoder.layers.0.norm1.bias",
    "enc0_n2_weight": "encoder.layers.0.norm2.weight",
    "enc0_n2_bias": "encoder.layers.0.norm2.bias",
    "enc1_q_weight": ("encoder.layers.1.self_attn.in_proj_weight", 0),
    "enc1_q_bias": ("encoder.layers.1.self_attn.in_proj_bias", 0),
    "enc1_k_weight": ("encoder.layers.1.self_attn.in_proj_weight", 1),
    "enc1_k_bias": ("encoder.layers.1.self_attn.in_proj_bias", 1),
    "enc1_v_weight": ("encoder.layers.1.self_attn.in_proj_weight", 2),
    "enc1_v_bias": ("encoder.layers.1.self_attn.in_proj_bias", 2),
    "enc1_out_weight": "encoder.layers.1.self_attn.out_proj.weight",
    "enc1_out_bias": "encoder.layers.1.self_attn.out_proj.bias",
    "enc1_ff1_weight": "encoder.layers.1.linear1.weight",
    "enc1_ff1_bias": "encoder.layers.1.linear1.bias",
    "enc1_ff2_weight": "encoder.layers.1.linear2.weight",
    "enc1_ff2_bias": "encoder.layers.1.linear2.bias",
    "enc1_n1_weight": "encoder.layers.1.norm1.weight",
    "enc1_n1_bias": "encoder.layers.1.norm1.bias",
    "enc1_n2_weight": "encoder.layers.1.norm2.weight",
    "enc1_n2_bias": "encoder.layers.1.norm2.bias",
    "enc2_q_weight": ("encoder.layers.2.self_attn.in_proj_weight", 0),
    "enc2_q_bias": ("encoder.layers.2.self_attn.in_proj_bias", 0),
    "enc2_k_weight": ("encoder.layers.2.self_attn.in_proj_weight", 1),
    "enc2_k_bias": ("encoder.layers.2.self_attn.in_proj_bias", 1),
    "enc2_v_weight": ("encoder.layers.2.self_attn.in_proj_weight", 2),
    "enc2_v_bias": ("encoder.layers.2.self_attn.in_proj_bias", 2),
    "enc2_out_weight": "encoder.layers.2.self_attn.out_proj.weight",
    "enc2_out_bias": "encoder.layers.2.self_attn.out_proj.bias",
    "enc2_ff1_weight": "encoder.layers.2.linear1.weight",
    "enc2_ff1_bias": "encoder.layers.2.linear1.bias",
    "enc2_ff2_weight": "encoder.layers.2.linear2.weight",
    "enc2_ff2_bias": "encoder.layers.2.linear2.bias",
    "enc2_n1_weight": "encoder.layers.2.norm1.weight",
    "enc2_n1_bias": "encoder.layers.2.norm1.bias",
    "enc2_n2_weight": "encoder.layers.2.norm2.weight",
    "enc2_n2_bias": "encoder.layers.2.norm2.bias",
    "enc3_q_weight": ("encoder.layers.3.self_attn.in_proj_weight", 0),
    "enc3_q_bias": ("encoder.layers.3.self_attn.in_proj_bias", 0),
    "enc3_k_weight": ("encoder.layers.3.self_attn.in_proj_weight", 1),
    "enc3_k_bias": ("encoder.layers.3.self_attn.in_proj_bias", 1),
    "enc3_v_weight": ("encoder.layers.3.self_attn.in_proj_weight", 2),
    "enc3_v_bias": ("encoder.layers.3.self_attn.in_proj_bias", 2),
    "enc3_out_weight": "encoder.layers.3.self_attn.out_proj.weight",
    "enc3_out_bias": "encoder.layers.3.self_attn.out_proj.bias",
    "enc3_ff1_weight": "encoder.layers.3.linear1.weight",
    "enc3_ff1_bias": "encoder.layers.3.linear1.bias",
    "enc3_ff2_weight": "encoder.layers.3.linear2.weight",
    "enc3_ff2_bias": "encoder.layers.3.linear2.bias",
    "enc3_n1_weight": "encoder.layers.3.norm1.weight",
    "enc3_n1_bias": "encoder.layers.3.norm1.bias",
    "enc3_n2_weight": "encoder.layers.3.norm2.weight",
    "enc3_n2_bias": "encoder.layers.3.norm2.bias",
    "red_xr_weight": ("red.weight_ih", 0),
    "red_xr_bias": ("red.bias_ih", 0),
    "red_xz_weight": ("red.weight_ih", 1),
    "red_xz_bias": ("red.bias_ih", 1),
    "red_xn_weight": ("red.weight_ih", 2),
    "red_xn_bias": ("red.bias_ih", 2),
    "red_hr_weight": ("red.weight_hh", 0),
    "red_hr_bias": ("red.bias_hh", 0),
    "red_hz_weight": ("red.weight_hh", 1),
    "red_hz_bias": ("red.bias_hh", 1),
    "red_hn_weight": ("red.weight_hh", 2),
    "red_hn_bias": ("red.bias_hh", 2),
    "abstract_weight": "abstract.0.weight",
    "abstract_bias": "abstract.0.bias",
    "abstract_norm_weight": "abstract.2.weight",
    "abstract_norm_bias": "abstract.2.bias",
    "to_blue_weight": "to_blue.weight",
    "to_blue_bias": "to_blue.bias",
    "blue_byte_weight": "blue_byte.weight",
    "blue_byte_bias": "blue_byte.bias",
    "red_byte_weight": "red_byte.weight",
    "red_byte_bias": "red_byte.bias",
    "stop_weight": "stop.weight",
    "stop_bias": "stop.bias",
    "kind_head_weight": "kind_head.weight",
    "kind_head_bias": "kind_head.bias",
    "summary_head_weight": "summary_head.weight",
    "summary_head_bias": "summary_head.bias",
    "budget0_weight": "budget.0.weight",
    "budget0_bias": "budget.0.bias",
    "budget_norm_weight": "budget.2.weight",
    "budget_norm_bias": "budget.2.bias",
    "budget3_weight": "budget.3.weight",
    "budget3_bias": "budget.3.bias",
    "copy_weight": "copy_query.weight",
    "nb_slot_keys": "notebook.slot_keys",
    "nb_key_weight": "notebook.key.weight",
    "nb_query_weight": "notebook.query.weight",
    "nb_value0_weight": "notebook.value.0.weight",
    "nb_value0_bias": "notebook.value.0.bias",
    "nb_write_gate0_weight": "notebook.write_gate.0.weight",
    "nb_write_gate0_bias": "notebook.write_gate.0.bias",
    "nb_write_gate2_weight": "notebook.write_gate.2.weight",
    "nb_write_gate2_bias": "notebook.write_gate.2.bias",
    "nb_budget0_weight": "notebook.budget.0.weight",
    "nb_budget0_bias": "notebook.budget.0.bias",
    "nb_budget3_weight": "notebook.budget.3.weight",
    "nb_budget3_bias": "notebook.budget.3.bias",
    "nb_value2_weight": "notebook.value.2.weight",
    "nb_value2_bias": "notebook.value.2.bias",
    "nb_budget2_weight": "notebook.budget.2.weight",
    "nb_budget2_bias": "notebook.budget.2.bias",
    "request_xr_weight": ("request_encoder.weight_ih_l0", 0),
    "request_xr_bias": ("request_encoder.bias_ih_l0", 0),
    "request_xz_weight": ("request_encoder.weight_ih_l0", 1),
    "request_xz_bias": ("request_encoder.bias_ih_l0", 1),
    "request_xn_weight": ("request_encoder.weight_ih_l0", 2),
    "request_xn_bias": ("request_encoder.bias_ih_l0", 2),
    "request_hr_weight": ("request_encoder.weight_hh_l0", 0),
    "request_hr_bias": ("request_encoder.bias_hh_l0", 0),
    "request_hz_weight": ("request_encoder.weight_hh_l0", 1),
    "request_hz_bias": ("request_encoder.bias_hh_l0", 1),
    "request_hn_weight": ("request_encoder.weight_hh_l0", 2),
    "request_hn_bias": ("request_encoder.bias_hh_l0", 2),
}
