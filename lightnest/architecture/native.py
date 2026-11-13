"""Native WGT networks; original Torch hosts remain explicit compatibility paths."""

from pathlib import Path
import weightlang as wgt


def load_recurrent(input_size, hidden_size, *, backend="native", **options):
    """Create an initialized, untrained shared GRU with explicit hidden ports.

    Native execution needs an existing wgtime service (WGTIME_ENDPOINT). The
    same model can use backend='torch'. This does not translate entire LightNest
    mask/top-k/inheritance programs into native execution.
    """
    if any(type(n) is not int or n < 1 for n in (input_size, hidden_size)):
        raise ValueError("GRU widths must be positive integers")
    return wgt.load(
        Path(__file__).with_name("wgt") / "shared_gru" / "model.wgt",
        backend=backend,
        dimensions={"I": input_size, "H": hidden_size},
        **options,
    )


class NativeELFGenerator(wgt.Model):
    """Full autoregressive ELF network on wgtime, with explicit recurrent state.

    Hosts provide discrete token/mode/position data. No Torch imports, tensor
    computations or autograd bridge occur here. Initial parameters are declared
    zeros, UNTRAINED; import_state_dict explicitly binds an existing model.
    """

    source_dir = "native_generator"
    gru_groups = {
        "decode0": ("decode", 0),
        "decode1": ("decode", 1),
        "red": ("red", None),
    }
    extra_parameters = {}

    def __init__(self, width=128, capacity=2048, **options):
        if type(width) is not int or not 1 <= width <= 1024:
            raise ValueError("native width must be in 1..1024")
        if type(capacity) is not int or not 1 <= capacity <= 16383:
            raise ValueError("native capacity must be in 1..16383")
        definition = wgt.compile(
            Path(__file__).with_name("wgt") / self.source_dir / "model.wgt",
            dimensions={"H": width, "H2": 2 * width, "C": capacity + 1},
        )
        super().__init__(definition, **options)
        self.width, self.capacity = width, capacity

    def pack_step(
        self, previous, condition, *, hidden=None, red_depth=2, position_offset=0
    ):
        """Pack one time position (one sequence) for native forward/BPTT.

        Condition is [mode, exit-byte, eight requested stdout bytes], matching
        the existing model's condition_tensor. Bits are input data, not a model
        computation; hidden consists of two H-wide decoder states.
        """
        if type(previous) is not int or not 0 <= previous <= 256:
            raise ValueError("previous token must be in 0..256")
        if (
            type(position_offset) is not int
            or not 0 <= position_offset <= self.capacity
        ):
            raise ValueError("position exceeds native generator capacity")
        if type(red_depth) is not int or not 0 <= red_depth <= 16:
            raise ValueError("red_depth must be in 0..16")
        if not isinstance(condition, (list, tuple)) or len(condition) != 10:
            raise ValueError("condition requires one mode and nine bytes")
        if (
            type(condition[0]) is not int
            or not 0 <= condition[0] < 3
            or any(type(x) is not int or not 0 <= x < 256 for x in condition[1:])
        ):
            raise ValueError("invalid condition mode or byte")
        hidden = [[0.0] * self.width, [0.0] * self.width] if hidden is None else hidden
        if len(hidden) != 2 or any(len(row) != self.width for row in hidden):
            raise ValueError("hidden requires two width-sized states")
        bits = [
            [float((byte >> bit) & 1) for bit in range(8)] for byte in condition[1:]
        ]
        return (
            [float(previous)],
            [float(position_offset)],
            [float(condition[0])],
            bits,
            hidden[0],
            hidden[1],
            [float(i < red_depth) for i in range(16)],
            [self.width**-0.5],
        )

    def step(self, previous, condition, **options):
        bits, blue, stop, h0, h1 = self(*self.pack_step(previous, condition, **options))
        return {"bits": bits, "blue_bits": blue, "stop": stop[0], "hidden": [h0, h1]}

    def generate(self, spec, *, red_depth=2, max_bytes=None):
        """Run host autoregressive scheduling over the complete native network."""
        if not isinstance(spec, dict):
            raise ValueError("generation request must be a dictionary")
        if (
            set(spec) == {"exit"}
            and type(spec["exit"]) is int
            and 0 <= spec["exit"] <= 255
        ):
            condition = [0, spec["exit"], *([0] * 8)]
        elif set(spec) == {"stdout"} and isinstance(spec["stdout"], str):
            payload = bytes.fromhex(spec["stdout"])
            if len(payload) != 8:
                raise ValueError("stdout request requires eight bytes")
            condition = [1, 0, *payload]
        else:
            raise ValueError("request must specify exit status or eight-byte stdout")
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max_bytes must be positive")
        maximum = min(self.capacity if max_bytes is None else max_bytes, self.capacity)
        result = bytearray()
        hidden = None
        previous = 256
        for position in range(maximum + 1):
            output = self.step(
                previous,
                condition,
                red_depth=red_depth,
                hidden=hidden,
                position_offset=position,
            )
            hidden = output["hidden"]
            if output["stop"] >= 0:
                return {"bytes": bytes(result), "stop": True, "steps": position + 1}
            if position == maximum:
                break
            previous = sum(
                int(value >= 0) << bit for bit, value in enumerate(output["bits"])
            )
            result.append(previous)
        return {"bytes": bytes(result), "stop": False, "steps": maximum + 1}

    def import_state_dict(self, state):
        """Copy an exact generator state dict; never execute its Python model.

        Gate rows split in reset/update/new order; independent input and hidden
        biases preserve reset-after-projection semantics. Successful replacement
        clears the service optimizer, as all native set_parameters calls do.
        """
        direct = {
            f"{name}_{field}": f"{name}.{field}"
            for name in [
                "byte",
                "position",
                "mode",
                "condition",
                "blue_bits",
                "red_bits",
                "stop",
                "copy_query",
                "copy_gate",
                "norm",
                "copy_keys",
            ]
            for field in (
                ["weight", "bias"]
                if name
                in ["condition", "blue_bits", "red_bits", "stop", "copy_gate", "norm"]
                else ["weight"]
            )
        }
        direct.update(self.extra_parameters)
        return _import_gate_state(self, state, direct, self.gru_groups)


def load_generator(width=128, capacity=2048, *, backend="native", **options):
    """Default to the complete native generator; Torch is an explicit option."""
    if backend == "native":
        return NativeELFGenerator(width, capacity, **options)
    if backend == "torch":
        from .generate import ELFGenerator

        if options:
            raise TypeError("Torch generator options belong to host PyTorch APIs")
        return ELFGenerator(width, capacity)
    raise ValueError("backend must be 'native' or 'torch'")


def _import_named_state(model, state, names):
    """Map exact existing state names into native storage; no model execution."""
    values = []
    expected = {
        source[0] if isinstance(source, tuple) else source for source in names.values()
    }
    if set(state) != expected:
        raise ValueError(
            f"state names differ: missing={expected - set(state)}, extra={set(state) - expected}"
        )

    def flatten(value, shape):
        if not shape:
            if isinstance(value, (list, tuple)):
                raise ValueError("state shape mismatch")
            return [float(value)]
        if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
            raise ValueError("state shape mismatch")
        return [v for row in value for v in flatten(row, shape[1:])]

    for p in model.parameters():
        source = names[p.name]
        key, part = source if isinstance(source, tuple) else (source, None)
        raw = state[key]
        raw = raw.tolist() if hasattr(raw, "tolist") else raw
        if part is not None:
            if len(raw) != 3 * p.shape[0]:
                raise ValueError(f"bad gate rows for {key}")
            raw = raw[part * p.shape[0] : (part + 1) * p.shape[0]]
        values.append(wgt.Parameter(p.name, p.shape, tuple(flatten(raw, p.shape))))
    return model.set_parameters(values)


class NativeNeuralNotebook(wgt.Model):
    """Complete notebook compression with native masked write and top-k budgets.

    One sequence has fixed construction-time items and slots; callers explicitly
    carry notes across invocations. Hard inference and soft training use the
    same service weights. No Python tensor computation or Torch dependency.
    """

    def __init__(self, width=128, slots=32, *, items, **options):
        if (
            any(type(v) is not int for v in [width, slots, items])
            or not 8 <= width <= 1024
            or not 2 <= slots <= 256
            or not 1 <= items <= 1024
        ):
            raise ValueError("invalid native notebook geometry")
        definition = wgt.compile(
            Path(__file__).with_name("wgt") / "native_notebook" / "model.wgt",
            dimensions={
                "H": width,
                "H2": 2 * width,
                "N": items,
                "S": slots,
                "SP": slots + 1,
                "NH": items * width,
                "SH": slots * width,
            },
        )
        super().__init__(definition, **options)
        self.width, self.slots, self.items = width, slots, items

    def pack_compress(
        self, evidence, query=None, notes=None, *, hard=False, valid=None
    ):
        if type(hard) is not bool:
            raise ValueError("hard must be bool")
        valid = [True] * self.items if valid is None else valid
        if (
            len(valid) != self.items
            or any(type(v) is not bool for v in valid)
            or not any(valid)
        ):
            raise ValueError("valid must contain at least one true evidence item")
        given = query is not None
        query = [0.0] * self.width if query is None else query
        notes = (
            [[0.0] * self.width for _ in range(self.slots)] if notes is None else notes
        )
        return (
            evidence,
            notes,
            query,
            [float(v) for v in valid],
            [float(hard)],
            [float(given)],
            [self.width**-0.5],
        )

    def compress(self, evidence, query=None, notes=None, **options):
        notes, context, weights, budget, address, gate = self(
            *self.pack_compress(evidence, query, notes, **options)
        )
        return {
            "notes": notes,
            "context": context,
            "weights": weights,
            "budget_logits": budget,
            "write": {"address": address, "write_gate": gate},
        }

    def import_state_dict(self, state):
        names = {
            "slot_keys": "slot_keys",
            "key_weight": "key.weight",
            "query_weight": "query.weight",
        }
        for native, torch in [
            ("value0", "value.0"),
            ("value2", "value.2"),
            ("write_gate0", "write_gate.0"),
            ("write_gate2", "write_gate.2"),
            ("budget0", "budget.0"),
            ("budget2", "budget.2"),
            ("budget3", "budget.3"),
        ]:
            for field in ["weight", "bias"]:
                names[native + "_" + field] = torch + "." + field
        return _import_named_state(self, state, names)


def load_notebook(width=128, slots=32, *, items, backend="native", **options):
    if backend == "native":
        return NativeNeuralNotebook(width, slots, items=items, **options)
    if backend == "torch":
        from .context import NeuralNotebook

        if options:
            raise TypeError("Torch notebook options belong to host PyTorch APIs")
        return NeuralNotebook(width, slots)
    raise ValueError("backend must be 'native' or 'torch'")


class NativeCompressRouteAttention(wgt.Model):
    """Native bank build plus dense/compressed/routed read, for one batch.

    Detail selection is represented by binary masks over the original bank,
    avoiding Python gathers. This preserves attention values/gradients, but uses
    N detail slots instead of compact K slots: no K-sized speedup is claimed.
    """

    def __init__(
        self, width=96, heads=4, landmarks=8, top_k=16, *, items, queries, **options
    ):
        values = [width, heads, landmarks, top_k, items, queries]
        if any(type(v) is not int or v <= 0 for v in values) or width % heads:
            raise ValueError("invalid native attention dimensions")
        dimensions = dict(
            H=width,
            A=heads,
            D=width // heads,
            N=items,
            Q=queries,
            L=landmarks,
            K=min(top_k, items),
            LN=landmarks * items,
            AL=heads * landmarks,
            AQ=heads * queries,
            J=landmarks + items,
        )
        super().__init__(
            wgt.compile(
                Path(__file__).with_name("wgt") / "native_attention" / "model.wgt",
                dimensions=dimensions,
            ),
            **options,
        )
        self.width, self.heads, self.landmarks, self.top_k, self.items, self.queries = (
            values
        )

    def pack_attention(self, observations, valid, condition, queries, *, mode="routed"):
        modes = ["routed", "compressed", "dense"]
        if mode not in modes:
            raise ValueError("invalid attention mode")
        if (
            len(valid) != self.items
            or any(type(v) is not bool for v in valid)
            or not any(valid)
        ):
            raise ValueError("attention requires a nonempty boolean evidence mask")
        # Validity/window pooling coefficients depend only on discrete input
        # layout, never weights or activations. Projecting/pooling learned
        # evidence, routing, normalization and all reads execute inside wgtime.
        pooling = []
        for i in range(self.landmarks):
            start, end = (
                i * self.items // self.landmarks,
                (i + 1) * self.items // self.landmarks,
            )
            count = sum(valid[start:end])
            pooling.append(
                [
                    float(start <= j < end and valid[j]) / max(count, 1)
                    for j in range(self.items)
                ]
            )
        return (
            observations,
            [float(v) for v in valid],
            condition,
            queries,
            pooling,
            [float(mode == m) for m in modes],
            [(self.width // self.heads) ** -0.5],
        )

    def attend(self, observations, valid, condition, queries, *, mode="routed"):
        output, landmarks, summaries, selected = self(
            *self.pack_attention(observations, valid, condition, queries, mode=mode)
        )
        return {
            "output": output,
            "landmarks": landmarks,
            "summaries": summaries,
            "selected": selected,
        }

    def import_state_dict(self, state):
        names = {
            f"{name}_weight": f"{name}.weight"
            for name in ["query", "key", "value", "condition", "output"]
        }
        names["landmark_bias"] = "landmark_bias"
        return _import_named_state(self, state, names)


def load_attention(
    width=96,
    heads=4,
    landmarks=8,
    top_k=16,
    *,
    items,
    queries,
    backend="native",
    **options,
):
    if backend == "native":
        return NativeCompressRouteAttention(
            width, heads, landmarks, top_k, items=items, queries=queries, **options
        )
    if backend == "torch":
        from .attention import CompressRouteAttention

        if options:
            raise TypeError("Torch attention options belong to host PyTorch APIs")
        return CompressRouteAttention(width, heads, landmarks, top_k)
    raise ValueError("backend must be 'native' or 'torch'")


def _import_gate_state(model, state, direct, groups):
    names = dict(direct)
    for native, (prefix, layer) in groups.items():
        for side, stem in [("x", "ih"), ("h", "hh")]:
            for part, gate in enumerate("rzn"):
                for field in ["weight", "bias"]:
                    source = f"{prefix}.{field}_{stem}" + (
                        f"_l{layer}" if layer is not None else ""
                    )
                    names[f"{native}_{side}{gate}_{field}"] = (source, part)
    return _import_named_state(model, state, names)


class NativeHierarchicalGRU(wgt.Model):
    """Causal dense/fixed/dynamic H-Net with explicit five-vector native state."""

    def __init__(self, input_size, width, *, mode="dynamic", target_ratio=4, **options):
        if (
            any(type(n) is not int or n < 1 for n in [input_size, width])
            or type(target_ratio) is not int
            or target_ratio < 2
            or mode not in ["dense", "fixed", "dynamic"]
        ):
            raise ValueError("invalid native H-Net geometry/mode")
        super().__init__(
            wgt.compile(
                Path(__file__).with_name("wgt")
                / "native_hierarchical_gru"
                / "model.wgt",
                dimensions={"I": input_size, "H": width},
            ),
            **options,
        )
        self.input_size, self.width, self.mode, self.target_ratio = (
            input_size,
            width,
            mode,
            target_ratio,
        )

    def pack_step(self, x, *, state=None, position_offset=0):
        if type(position_offset) is not int or position_offset < 0:
            raise ValueError("invalid H-Net offset")
        state = [[0.0] * self.width for _ in range(5)] if state is None else state
        if len(state) != 5 or any(len(row) != self.width for row in state):
            raise ValueError("H-Net needs five width-sized state vectors")
        modes = ["dynamic", "fixed", "dense"]
        return (
            x,
            *state,
            [float(position_offset == 0)],
            [float(position_offset % self.target_ratio == 0)],
            [float(self.mode == mode) for mode in modes],
        )

    def step(self, x, **options):
        output, *values = self(*self.pack_step(x, **options))
        return {
            "output": output,
            "state": values[:5],
            "probability": values[5][0],
            "boundary": bool(values[6][0]),
        }

    def import_state_dict(self, state):
        direct = {
            name + "_weight": name + ".weight" for name in ["query", "key", "skip"]
        }
        for name in ["encoder_norm", "main_norm", "output_norm"]:
            for field in ["weight", "bias"]:
                direct[name + "_" + field] = name + "." + field
        return _import_gate_state(
            self,
            state,
            direct,
            {name: (name, 0) for name in ["encoder", "main", "decoder"]},
        )


def load_hierarchical_recurrent(input_size, width, *, backend="native", **options):
    if backend == "native":
        return NativeHierarchicalGRU(input_size, width, **options)
    if backend == "torch":
        from .hnet import HierarchicalGRU

        return HierarchicalGRU(input_size, width, **options)
    raise ValueError("backend must be 'native' or 'torch'")


class NativeHNetELFGenerator(NativeELFGenerator):
    """Complete H-Net generator and confidence STE in one native AD graph."""

    source_dir = "native_hnet_generator"
    gru_groups = {
        **{name: ("decode." + name, 0) for name in ["encoder", "main", "decoder"]},
        "red": ("red", None),
    }
    extra_parameters = {
        **{
            name + "_weight": "decode." + name + ".weight"
            for name in ["query", "key", "skip"]
        },
        **{
            name + "_" + field: "decode." + name + "." + field
            for name in ["encoder_norm", "main_norm", "output_norm"]
            for field in ["weight", "bias"]
        },
    }

    def __init__(
        self,
        width=128,
        capacity=2048,
        *,
        chunk_mode="dynamic",
        target_ratio=4,
        **options,
    ):
        if (
            chunk_mode not in ["dynamic", "fixed", "dense"]
            or type(target_ratio) is not int
            or target_ratio < 2
        ):
            raise ValueError("invalid native chunk mode/ratio")
        super().__init__(width, capacity, **options)
        self.chunk_mode, self.target_ratio = chunk_mode, target_ratio

    def pack_step(
        self, previous, condition, *, hidden=None, red_depth=2, position_offset=0
    ):
        if hidden is None and position_offset != 0:
            raise ValueError("initial H-Net offset must be zero")
        hidden = [[0.0] * self.width for _ in range(5)] if hidden is None else hidden
        if len(hidden) != 5 or any(len(row) != self.width for row in hidden):
            raise ValueError("H-Net generator needs five state vectors")
        ports = super().pack_step(
            previous, condition, red_depth=red_depth, position_offset=position_offset
        )
        return (
            *ports[:4],
            *hidden,
            *ports[6:],
            [float(position_offset == 0)],
            [float(position_offset % self.target_ratio == 0)],
            [float(self.chunk_mode == mode) for mode in ["dynamic", "fixed", "dense"]],
        )

    def step(self, previous, condition, **options):
        bits, blue, stop, *values = self(
            *self.pack_step(previous, condition, **options)
        )
        return {
            "bits": bits,
            "blue_bits": blue,
            "stop": stop[0],
            "hidden": values[:5],
            "boundary_probability": values[5][0],
            "boundary": bool(values[6][0]),
        }


def load_hnet_generator(width=128, capacity=2048, *, backend="native", **options):
    if backend == "native":
        return NativeHNetELFGenerator(width, capacity, **options)
    if backend == "torch":
        from .hnet import HNetELFGenerator

        return HNetELFGenerator(width, capacity, **options)
    raise ValueError("backend must be 'native' or 'torch'")


def load_rgb_policy(image, command, frames=(), *, backend="native", **options):
    from .native_rgb import load_rgb_policy as load

    return load(image, command, frames, backend=backend, **options)


def load_artifact_core(raw, request, *, backend="native", **options):
    from .native_any2any import load_artifact_core as load
    return load(raw, request, backend=backend, **options)
