"""The same shared GRU source executes in Rust and in an explicit Torch model."""

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import torch
import weightlang as wgt
from lightnest.architecture.native import load_recurrent
from lightnest.architecture.generate import ELFGenerator


def _rgb_packed(blue, red):
    parts = [
        blue[k]
        for k in [
            "logits",
            "bits",
            "risk_logit",
            "language_logits",
            "budget_logits",
            "context_budget_logits",
        ]
    ]
    parts += [
        red[k]
        for k in [
            "logits",
            "bits",
            "risk_logit",
            "budget_logits",
            "context_budget_logits",
        ]
    ]
    return torch.cat([value.reshape(-1) for value in parts])


class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        daemon = os.environ.get("WGTIME_TEST_DAEMON")
        if not daemon:
            raise unittest.SkipTest(
                "set WGTIME_TEST_DAEMON to the matching prebuilt daemon"
            )
        cls.temp = tempfile.TemporaryDirectory(prefix="lightnest-wgtime-")
        directory = Path(cls.temp.name)
        directory.chmod(0o700)
        cls.process = subprocess.Popen(
            [daemon, "--test-directory", str(directory)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        cls.endpoint = directory / "service.sock"
        for _ in range(100):
            if cls.endpoint.exists():
                return
            if cls.process.poll() is not None:
                raise AssertionError(cls.process.stderr.read().decode())
            time.sleep(0.02)
        raise AssertionError("wgtime did not start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()
        cls.process.stderr.close()
        cls.temp.cleanup()

    def test_native_and_torch_forward_recurrent_gradients_and_training(self):
        torch.manual_seed(18)
        reference = torch.nn.GRUCell(3, 2)
        mapping = {}
        for prefix, stem in [("x", "ih"), ("h", "hh")]:
            for part, gate in enumerate("rzn"):
                mapping[prefix + gate + "_weight"] = (
                    getattr(reference, "weight_" + stem).detach().chunk(3)[part]
                )
                mapping[prefix + gate + "_bias"] = (
                    getattr(reference, "bias_" + stem).detach().chunk(3)[part]
                )
        portable = load_recurrent(3, 2, backend="torch", device="cpu")
        self.assertIsInstance(portable, wgt.Model)
        with torch.no_grad():
            for name, value in mapping.items():
                portable.to_torch().get_parameter(portable.parameter_names[name]).copy_(
                    value
                )
        with load_recurrent(3, 2, endpoint=self.endpoint, optimizer="sgd") as native:
            self.assertIsInstance(native, wgt.Model)
            self.assertEqual(native.backend, "native")
            native.set_parameters(
                [
                    wgt.Parameter(name, tuple(t.shape), tuple(t.flatten().tolist()))
                    for name, t in mapping.items()
                ]
            )
            native.set_optimizer(
                "sgd"
            )  # explicit parameter replacement resets training state
            initial = torch.tensor([0.1, -0.2], requires_grad=True)
            a = initial
            b = initial.detach().clone().requires_grad_()
            c = initial.detach().tolist()
            values = [
                torch.tensor([0.2, -0.3, 0.1], requires_grad=True),
                torch.tensor([0.3, 0.4, -0.2], requires_grad=True),
            ]
            for x in values:
                a = reference(x, a)
                b = portable(x, b)[0]
                c = native(x.detach().tolist(), c)[0]
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(
                    a.detach(), torch.tensor(c), atol=1e-6, rtol=1e-5
                )
            a.square().mean().backward()
            b.square().mean().backward()
            for prefix, stem in [("x", "ih"), ("h", "hh")]:
                for part, gate in enumerate("rzn"):
                    for suffix in ["weight", "bias"]:
                        expected = getattr(reference, suffix + "_" + stem).grad.chunk(
                            3
                        )[part]
                        actual = (
                            portable.to_torch()
                            .get_parameter(
                                portable.parameter_names[prefix + gate + "_" + suffix]
                            )
                            .grad
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=1e-6, rtol=1e-5
                        )
            rows = [
                x.detach().tolist() + ([0.1, -0.2] if i == 0 else [0.0, 0.0])
                for i, x in enumerate(values)
            ]
            loss = native.train_recurrent(
                rows,
                [None, [0.0, 0.0]],
                input_offset=3,
                output_offset=0,
                state_width=2,
                supervised_offset=0,
                supervised_width=2,
                rate=0.01,
            )
            self.assertAlmostEqual(loss, float(a.detach().square().mean()), places=6)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            updated = {
                p.name: torch.tensor(p.data).reshape(p.shape)
                for p in native.parameters()
            }
            for prefix, stem in [("x", "ih"), ("h", "hh")]:
                for part, gate in enumerate("rzn"):
                    for suffix in ["weight", "bias"]:
                        torch.testing.assert_close(
                            updated[prefix + gate + "_" + suffix],
                            getattr(reference, suffix + "_" + stem)
                            .detach()
                            .chunk(3)[part],
                            atol=1e-6,
                            rtol=1e-5,
                        )

    def test_complex_application_explicitly_uses_torch_wgt_object(self):
        model = ELFGenerator(width=8, capacity=16)
        self.assertIsInstance(model, wgt.Model)
        self.assertEqual(model.backend, "torch")
        self.assertIs(model.to_torch(), model)

    def test_complete_generator_native_forward_and_bptt_update(self):
        from lightnest.architecture.native import load_generator

        torch.manual_seed(31)
        reference = ELFGenerator(width=8, capacity=16)
        condition = [1, 0, 72, 101, 108, 108, 111, 33, 10, 0]
        cond = torch.tensor([condition])
        with load_generator(8, 16, endpoint=self.endpoint) as native:
            native.import_state_dict(reference.state_dict())
            for depth in [0, 2, 16]:
                hidden = torch.randn(2, 1, 8) * 0.1
                state = hidden[:, 0].tolist()
                previous = torch.tensor([[256, 23, 9]])
                expected = reference(previous, cond, hidden=hidden, red_depth=depth)
                outputs = []
                for i, token in enumerate(previous[0].tolist()):
                    output = native.step(
                        token,
                        condition,
                        hidden=state,
                        red_depth=depth,
                        position_offset=i,
                    )
                    outputs.append(output)
                    state = output["hidden"]
                for key in ["bits", "blue_bits", "stop"]:
                    actual = torch.tensor([o[key] for o in outputs])
                    torch.testing.assert_close(
                        actual, expected[key][0].detach(), atol=4e-6, rtol=3e-5
                    )
                torch.testing.assert_close(
                    torch.tensor(state),
                    expected["hidden"][:, 0].detach(),
                    atol=2e-6,
                    rtol=2e-5,
                )
            reference.zero_grad()
            expected = reference(torch.tensor([[256, 23]]), cond, red_depth=2)
            loss = expected["bits"][:, -1].square().mean()
            loss.backward()
            rows = []
            for i, token in enumerate([256, 23]):
                ports = native.pack_step(token, condition, position_offset=i)
                rows.append(
                    [
                        x
                        for port in ports
                        for row in port
                        for x in (row if isinstance(row, list) else [row])
                    ]
                )
            native.set_optimizer("sgd")
            actual_loss = native.train_recurrent(
                rows,
                [None, [0.0] * 8],
                input_offset=75,
                output_offset=17,
                state_width=16,
                supervised_offset=0,
                supervised_width=8,
                rate=0.01,
            )
            self.assertAlmostEqual(actual_loss, float(loss.detach()), places=5)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            actual = native.parameters()
            with load_generator(8, 16, endpoint=self.endpoint) as updated:
                updated.import_state_dict(reference.state_dict())
                for left, right in zip(actual, updated.parameters(), strict=True):
                    self.assertEqual(left.name, right.name)
                    torch.testing.assert_close(
                        torch.tensor(left.data),
                        torch.tensor(right.data),
                        atol=2e-6,
                        rtol=2e-5,
                    )

    def test_complete_notebook_native_mask_topk_and_recurrent_training(self):
        from lightnest.architecture.native import load_notebook
        from lightnest.architecture.context import NeuralNotebook

        torch.manual_seed(41)
        reference = NeuralNotebook(8, 4)
        evidence = torch.randn(1, 3, 8)
        query = torch.randn(1, 8)
        notes = torch.randn(1, 4, 8)
        valid = torch.tensor([[True, False, True]])
        with load_notebook(8, 4, items=3, endpoint=self.endpoint) as native:
            native.import_state_dict(reference.state_dict())
            for hard in [False, True]:
                for q in [None, query]:
                    expected = reference.compress(
                        evidence, q, notes, hard=hard, valid=valid
                    )
                    actual = native.compress(
                        evidence[0].tolist(),
                        None if q is None else q[0].tolist(),
                        notes[0].tolist(),
                        hard=hard,
                        valid=valid[0].tolist(),
                    )
                    for key in ["notes", "context", "weights", "budget_logits"]:
                        torch.testing.assert_close(
                            torch.tensor(actual[key]),
                            expected[key][0].detach(),
                            atol=2e-6,
                            rtol=2e-5,
                        )
                    for key in ["address", "write_gate"]:
                        torch.testing.assert_close(
                            torch.tensor(actual["write"][key]),
                            expected["write"][key][0].detach(),
                            atol=2e-6,
                            rtol=2e-5,
                        )
            a = reference.compress(evidence, query, notes, valid=valid)
            b = reference.compress(evidence * 0.7, query, a["notes"], valid=valid)
            loss = b["context"].square().mean()
            loss.backward()
            rows = []
            for e in [evidence, evidence * 0.7]:
                ports = native.pack_compress(
                    e[0].tolist(),
                    query[0].tolist(),
                    notes[0].tolist(),
                    valid=valid[0].tolist(),
                )
                rows.append(
                    [
                        x
                        for port in ports
                        for row in port
                        for x in (row if isinstance(row, list) else [row])
                    ]
                )
            native.set_optimizer("sgd")
            got = native.train_recurrent(
                rows,
                [None, [0.0] * 8],
                input_offset=24,
                output_offset=0,
                state_width=32,
                supervised_offset=32,
                supervised_width=8,
                rate=0.01,
            )
            self.assertAlmostEqual(got, float(loss.detach()), places=6)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            actual = native.parameters()
            with load_notebook(8, 4, items=3, endpoint=self.endpoint) as updated:
                updated.import_state_dict(reference.state_dict())
                for left, right in zip(actual, updated.parameters(), strict=True):
                    self.assertEqual(left.name, right.name)
                    torch.testing.assert_close(
                        torch.tensor(left.data),
                        torch.tensor(right.data),
                        atol=2e-6,
                        rtol=2e-5,
                    )

    def test_complete_native_routed_attention_all_modes_and_parameter_update(self):
        for geometry in [
            (8, 2, 4, 2, 3, 2, [True, False, True]),
            (16, 4, 3, 2, 7, 4, [True, False, True, True, False, True, True]),
        ]:
            with self.subTest(geometry=geometry):
                self._check_routed_attention(*geometry)

    def _check_routed_attention(
        self, width, heads, landmarks, top_k, items, query_count, valid_rows
    ):
        from lightnest.architecture.native import load_attention
        from lightnest.architecture.attention import CompressRouteAttention

        torch.manual_seed(51)
        reference = CompressRouteAttention(width, heads, landmarks, top_k)
        observations = torch.randn(1, items, width)
        condition = torch.randn(1, width)
        queries = torch.randn(1, query_count, width)
        valid = torch.tensor([valid_rows])
        with load_attention(
            width,
            heads,
            landmarks,
            top_k,
            items=items,
            queries=query_count,
            endpoint=self.endpoint,
        ) as native:
            native.import_state_dict(reference.state_dict())
            bank = reference.build(observations, valid, condition)
            selected = torch.zeros(1, heads, landmarks, items).scatter_(
                -1, bank.detail_indices, 1
            )
            for mode in ["dense", "compressed", "routed"]:
                expected = reference.read(queries, bank, mode=mode)
                actual = native.attend(
                    observations[0].tolist(),
                    valid[0].tolist(),
                    condition[0].tolist(),
                    queries[0].tolist(),
                    mode=mode,
                )
                for value, want in [
                    (actual["output"], expected[0]),
                    (actual["landmarks"], bank.landmarks[0]),
                    (actual["summaries"], bank.summaries[0]),
                    (actual["selected"], selected[0]),
                ]:
                    torch.testing.assert_close(
                        torch.tensor(value), want.detach(), atol=3e-6, rtol=3e-5
                    )
            output = reference.read(queries, bank, mode="routed")
            packed = torch.cat(
                [
                    value.flatten()
                    for value in [output, bank.landmarks, bank.summaries, selected]
                ]
            )
            loss = packed.square().mean()
            loss.backward()
            ports = native.pack_attention(
                observations[0].tolist(),
                valid[0].tolist(),
                condition[0].tolist(),
                queries[0].tolist(),
            )
            rows = [
                [
                    x
                    for port in ports
                    for row in port
                    for x in (row if isinstance(row, list) else [row])
                ]
            ]
            native.set_optimizer("sgd")
            got = native.train_step(rows, [[0.0] * len(packed)], rate=0.01)
            self.assertAlmostEqual(got, float(loss.detach()), places=6)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            actual = native.parameters()
            with load_attention(
                width,
                heads,
                landmarks,
                top_k,
                items=items,
                queries=query_count,
                endpoint=self.endpoint,
            ) as updated:
                updated.import_state_dict(reference.state_dict())
                for left, right in zip(actual, updated.parameters(), strict=True):
                    self.assertEqual(left.name, right.name)
                    torch.testing.assert_close(
                        torch.tensor(left.data),
                        torch.tensor(right.data),
                        atol=2e-6,
                        rtol=2e-5,
                    )

    def test_native_hierarchical_gru_modes_state_and_bptt(self):
        from lightnest.architecture.native import load_hierarchical_recurrent
        from lightnest.architecture.hnet import HierarchicalGRU

        for mode in ["dense", "fixed", "dynamic"]:
            torch.manual_seed(61)
            reference = HierarchicalGRU(16, 8, mode=mode, target_ratio=2)
            x = torch.randn(1, 4, 16)
            with load_hierarchical_recurrent(
                16, 8, mode=mode, target_ratio=2, endpoint=self.endpoint
            ) as native:
                native.import_state_dict(reference.state_dict())
                expected, state = reference(x)
                actual = []
                hidden = None
                for i in range(4):
                    result = native.step(
                        x[0, i].tolist(), state=hidden, position_offset=i
                    )
                    actual.append(result["output"])
                    hidden = result["state"]
                    self.assertEqual(
                        result["boundary"], bool(reference.last_boundaries[0, i])
                    )
                    self.assertAlmostEqual(
                        result["probability"],
                        float(reference.last_probabilities[0, i].detach()),
                        places=5,
                    )
                torch.testing.assert_close(
                    torch.tensor(actual), expected[0].detach(), atol=3e-6, rtol=3e-5
                )
                for got, want in zip(
                    hidden,
                    [
                        state.encoder[0, 0],
                        state.main[0, 0],
                        state.decoder[0, 0],
                        state.key[0],
                        state.smooth[0],
                    ],
                    strict=True,
                ):
                    torch.testing.assert_close(
                        torch.tensor(got), want.detach(), atol=3e-6, rtol=3e-5
                    )
                loss = expected[:, -1].square().mean()
                loss.backward()
                rows = []
                for i in range(4):
                    ports = native.pack_step(x[0, i].tolist(), position_offset=i)
                    rows.append([item for port in ports for item in port])
                native.set_optimizer("sgd")
                got = native.train_recurrent(
                    rows,
                    [None, None, None, [0.0] * 8],
                    input_offset=16,
                    output_offset=8,
                    state_width=40,
                    supervised_offset=0,
                    supervised_width=8,
                    rate=0.01,
                )
                self.assertAlmostEqual(got, float(loss.detach()), places=5)
                torch.optim.SGD(reference.parameters(), lr=0.01).step()
                actual = native.parameters()
                with load_hierarchical_recurrent(
                    16, 8, mode=mode, target_ratio=2, endpoint=self.endpoint
                ) as updated:
                    updated.import_state_dict(reference.state_dict())
                    for left, right in zip(actual, updated.parameters(), strict=True):
                        self.assertEqual(left.name, right.name)
                        torch.testing.assert_close(
                            torch.tensor(left.data),
                            torch.tensor(right.data),
                            atol=3e-6,
                            rtol=3e-5,
                        )

    def test_complete_native_hnet_generator_and_cross_state_training(self):
        from lightnest.architecture.native import load_hnet_generator
        from lightnest.architecture.hnet import HNetELFGenerator

        condition = [1, 0, 65, 66, 67, 68, 69, 70, 71, 72]
        cond = torch.tensor([condition])
        tokens = [256, 23, 9]
        for mode in ["dense", "fixed", "dynamic"]:
            torch.manual_seed(71)
            reference = HNetELFGenerator(8, 16, chunk_mode=mode, target_ratio=2)
            expected = reference(torch.tensor([tokens]), cond, red_depth=2)
            with load_hnet_generator(
                8, 16, chunk_mode=mode, target_ratio=2, endpoint=self.endpoint
            ) as native:
                native.import_state_dict(reference.state_dict())
                state = None
                actual = []
                for i, token in enumerate(tokens):
                    out = native.step(token, condition, hidden=state, position_offset=i)
                    state = out["hidden"]
                    actual.append(out)
                    self.assertEqual(
                        out["boundary"], bool(expected["boundaries"][0, i])
                    )
                    self.assertAlmostEqual(
                        out["boundary_probability"],
                        float(expected["boundary_probabilities"][0, i].detach()),
                        places=5,
                    )
                for key in ["bits", "blue_bits", "stop"]:
                    torch.testing.assert_close(
                        torch.tensor([out[key] for out in actual]),
                        expected[key][0].detach(),
                        atol=4e-6,
                        rtol=4e-5,
                    )
                loss = expected["bits"][:, -1].square().mean()
                loss.backward()
                rows = []
                for i, token in enumerate(tokens):
                    initial = None if i == 0 else [[0.0] * 8 for _ in range(5)]
                    ports = native.pack_step(
                        token, condition, hidden=initial, position_offset=i
                    )
                    rows.append(
                        [
                            x
                            for port in ports
                            for row in port
                            for x in (row if isinstance(row, list) else [row])
                        ]
                    )
                native.set_optimizer("sgd")
                got = native.train_recurrent(
                    rows,
                    [None, None, [0.0] * 8],
                    input_offset=75,
                    output_offset=17,
                    state_width=40,
                    supervised_offset=0,
                    supervised_width=8,
                    rate=0.01,
                )
                self.assertAlmostEqual(got, float(loss.detach()), places=5)
                torch.optim.SGD(reference.parameters(), lr=0.01).step()
                actual = native.parameters()
                with load_hnet_generator(
                    8, 16, chunk_mode=mode, target_ratio=2, endpoint=self.endpoint
                ) as updated:
                    updated.import_state_dict(reference.state_dict())
                    for left, right in zip(actual, updated.parameters(), strict=True):
                        self.assertEqual(left.name, right.name)
                        torch.testing.assert_close(
                            torch.tensor(left.data),
                            torch.tensor(right.data),
                            atol=4e-6,
                            rtol=4e-5,
                        )

    def test_full_default_width_native_generator_capacity(self):
        from lightnest.architecture.native import load_generator, load_hnet_generator

        for factory in [load_generator, load_hnet_generator]:
            with factory(endpoint=self.endpoint) as model:
                self.assertEqual(model.width, 128)
                result = model.step(256, [0] * 10, red_depth=0)
                self.assertEqual(len(result["bits"]), 8)
                self.assertTrue(all(value == value for value in result["bits"]))

    def test_complete_native_generation_scheduling_matches_torch(self):
        from lightnest.architecture.native import load_generator, load_hnet_generator
        from lightnest.architecture.hnet import HNetELFGenerator

        for factory, cls in [
            (load_generator, ELFGenerator),
            (load_hnet_generator, HNetELFGenerator),
        ]:
            torch.manual_seed(81)
            reference = cls(8, 16).eval()
            with factory(8, 16, endpoint=self.endpoint) as native:
                for bias in [-5.0, 5.0]:
                    with torch.no_grad():
                        reference.stop.weight.zero_()
                        reference.stop.bias.fill_(bias)
                    native.import_state_dict(reference.state_dict())
                    for request in [{"exit": 7}, {"stdout": bytes(range(8)).hex()}]:
                        self.assertEqual(
                            native.generate(request, max_bytes=5),
                            reference.generate(request, max_bytes=5),
                        )

    def test_complex_wgt_sources_also_run_on_explicit_torch_backend(self):
        from lightnest.architecture.native import (
            load_generator,
            load_hnet_generator,
            load_notebook,
            load_attention,
            load_hierarchical_recurrent,
        )

        torch.manual_seed(91)
        factories = [
            (
                lambda: load_generator(8, 16, endpoint=self.endpoint),
                lambda m: m.pack_step(256, [0] * 10),
            ),
            (
                lambda: load_hnet_generator(8, 16, endpoint=self.endpoint),
                lambda m: m.pack_step(256, [0] * 10),
            ),
            (
                lambda: load_notebook(8, 4, items=3, endpoint=self.endpoint),
                lambda m: m.pack_compress([[0.1] * 8] * 3, [0.2] * 8),
            ),
            (
                lambda: load_attention(
                    8, 2, 4, 2, items=3, queries=2, endpoint=self.endpoint
                ),
                lambda m: m.pack_attention(
                    [[0.1] * 8] * 3, [True, False, True], [0.2] * 8, [[0.3] * 8] * 2
                ),
            ),
            (
                lambda: load_hierarchical_recurrent(16, 8, endpoint=self.endpoint),
                lambda m: m.pack_step([0.1] * 16),
            ),
        ]
        for factory, pack in factories:
            with factory() as native:
                values = [
                    wgt.Parameter(
                        p.name,
                        p.shape,
                        tuple((torch.randn(p.shape) * 0.1).flatten().tolist()),
                    )
                    for p in native.parameters()
                ]
                native.set_parameters(values)
                portable = wgt.from_ir(native.definition, backend="torch", device="cpu")
                raw = portable.to_torch().base
                with torch.no_grad():
                    for p in values:
                        raw.get_parameter(raw.parameter_names[p.name]).copy_(
                            torch.tensor(p.data).reshape(p.shape)
                        )
                ports = pack(native)
                actual = native(*ports)
                expected = portable(
                    *[torch.tensor(value, dtype=torch.float32) for value in ports]
                )
                for left, right in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        torch.tensor(left), right.detach(), atol=4e-6, rtol=4e-5
                    )

    def test_complete_rgb_native_observation_thinking_readouts_and_training(self):
        from lightnest.architecture.native import load_rgb_policy
        from lightnest.architecture.native_rgb import NativeRGBConfig
        from lightnest.architecture.model import RGBConfig, RGBPolicy
        from lightnest.architecture.observe import Frame

        torch.manual_seed(101)
        config = NativeRGBConfig(
            width=8,
            chunk_size=8,
            heads=2,
            landmarks=2,
            detail_k=1,
            blue_depth=1,
            red_depths=(2, 3),
            note_slots=3,
        )
        reference = RGBPolicy(RGBConfig(**vars(config)))
        self.assertIsInstance(
            load_rgb_policy(b"a", "x", config=vars(config), backend="torch"), RGBPolicy
        )
        with self.assertRaises(TypeError):
            load_rgb_policy(b"a", "x", config=config, backend="torch", depth=3)
        # Exercise guidance and language gradients despite their cold-start zeros.
        with torch.no_grad():
            for module in [
                reference.red_to_blue,
                reference.abstract_to_blue,
                reference.blue_language,
            ]:
                module.weight.normal_(std=0.1)
                module.bias.normal_(std=0.1)
        image = bytes(range(9))
        command = "fix"
        frames = (
            Frame("stdout", image[:8], 128),
            Frame("feedback", bytes([9, 10, 11]), 4096),
            Frame("empty", b"", 7),
        )

        with load_rgb_policy(
            image, command, frames, config=config, endpoint=self.endpoint
        ) as native:
            native.import_state_dict(reference.state_dict())
            for training in [True, False]:
                native.train(training)
                reference.train(training)
                for mode in ["routed", "dense", "compressed"]:
                    blue, red, _ = reference(image, command, frames, depth=3, mode=mode)
                    actual = wgt.Model.forward(
                        native, *native.pack(image, command, frames, mode=mode)
                    )
                    want = _rgb_packed(blue, red)
                    got = torch.cat([torch.tensor(value).flatten() for value in actual])
                    torch.testing.assert_close(got, want.detach(), atol=5e-6, rtol=5e-5)
            reference.train()
            native.train()
            reference.zero_grad()
            blue, red, _ = reference(image, command, frames, depth=3)
            expected = _rgb_packed(blue, red)
            loss = expected.square().mean()
            loss.backward()
            ports = native.pack(image, command, frames)

            def flatten(x):
                return [
                    v
                    for item in x
                    for v in (flatten(item) if isinstance(item, list) else [item])
                ]

            native.set_optimizer("sgd")
            actual_loss = native.train_step(
                [[v for port in ports for v in flatten(port)]],
                [[0.0] * len(expected)],
                rate=0.01,
            )
            self.assertAlmostEqual(actual_loss, float(loss.detach()), places=5)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            actual = native.parameters()
            with load_rgb_policy(
                image, command, frames, config=config, endpoint=self.endpoint
            ) as updated:
                updated.import_state_dict(reference.state_dict())
                for left, right in zip(actual, updated.parameters(), strict=True):
                    self.assertEqual(left.name, right.name)
                    torch.testing.assert_close(
                        torch.tensor(left.data),
                        torch.tensor(right.data),
                        atol=4e-6,
                        rtol=4e-5,
                    )
            # The very same ordinary WGT definition also executes in Torch.
            portable = wgt.from_ir(native.definition, backend="torch", device="cpu")
            raw = portable.to_torch().base
            with torch.no_grad():
                for p in actual:
                    raw.get_parameter(raw.parameter_names[p.name]).copy_(
                        torch.tensor(p.data).reshape(p.shape)
                    )
            native_values = wgt.Model.forward(native, *ports)
            torch_values = portable(
                *[torch.tensor(v, dtype=torch.float32) for v in ports]
            )
            for got, want in zip(native_values, torch_values, strict=True):
                torch.testing.assert_close(
                    torch.tensor(got), want.detach(), atol=5e-6, rtol=5e-5
                )

            # Risk targets must not train the byte/guidance/observer pathways.
            before = native.parameters()
            reference.zero_grad()
            blue, red, _ = reference(image, command, frames, depth=3)
            loss = (
                blue["risk_logit"].square().sum() + red["risk_logit"].square().sum()
            ) / len(expected)
            loss.backward()
            target = flatten(wgt.Model.forward(native, *ports))
            first = 9 * len(image)
            second = (
                first
                + 1
                + 257
                + len(config.red_depths)
                + 1
                + config.note_slots
                + 1
                + 9 * len(image)
            )
            target[first] = target[second] = 0.0
            native.train_step([flatten(ports)], [target], rate=0.01)
            torch.optim.SGD(reference.parameters(), lr=0.01).step()
            after = native.parameters()
            for old, new in zip(before, after, strict=True):
                if not new.name.startswith("risk"):
                    self.assertEqual(old.data, new.data, new.name)
            with load_rgb_policy(
                image, command, frames, config=config, endpoint=self.endpoint
            ) as oracle:
                oracle.import_state_dict(reference.state_dict())
                for got, want in zip(after, oracle.parameters(), strict=True):
                    torch.testing.assert_close(
                        torch.tensor(got.data),
                        torch.tensor(want.data),
                        atol=4e-6,
                        rtol=4e-5,
                    )

    def test_rgb_default_size_and_rebinding_preserve_optimizer(self):
        from lightnest.architecture.native import load_rgb_policy
        from lightnest.architecture.model import RGBPolicy, RGBConfig

        torch.manual_seed(109)
        reference = RGBPolicy(RGBConfig())
        image, command = bytes(range(9)), "a"
        with load_rgb_policy(image, command, endpoint=self.endpoint) as native:
            native.import_state_dict(reference.state_dict())
            for training in [False, True]:
                reference.train(training)
                native.train(training)
                blue, red, _ = reference(image, command, depth=16)
                ports = native.pack(image, command)
                actual = wgt.Model.forward(native, *ports)
                torch.testing.assert_close(
                    torch.cat([torch.tensor(v).flatten() for v in actual]),
                    _rgb_packed(blue, red).detach(),
                    atol=1e-5,
                    rtol=1e-4,
                )

            # Default depth=16 and width=64 must also fit a real native AD tape.
            def flat(values):
                return torch.cat([torch.tensor(v).flatten() for v in values]).tolist()

            native.set_optimizer("adam")
            zeros = [0.0] * _rgb_packed(blue, red).numel()
            native.train_step([flat(ports)], [zeros], rate=0.0001)
            optimizer = torch.optim.Adam(reference.parameters(), lr=0.0001)
            reference.zero_grad()
            _rgb_packed(blue, red).square().mean().backward()
            optimizer.step()
            # Rebinding must keep the existing Adam step/moments, not re-import weights.
            # Check optimizer continuity with a same-session geometry roundtrip.
            # Importing a checkpoint would be a different, unsupported contract.
            before = native.parameters()
            native.specialize(bytes(range(65)), "longer", depth=8)
            self.assertEqual(native.parameters(), before)
            changed = wgt.Model.forward(
                native, *native.pack(bytes(range(65)), "longer")
            )
            self.assertEqual(len(changed[0]), 65)
            with torch.no_grad():
                changed_blue, changed_red, _ = reference(
                    bytes(range(65)), "longer", depth=8
                )
            torch.testing.assert_close(
                torch.tensor(flat(changed)),
                _rgb_packed(changed_blue, changed_red),
                atol=1e-5,
                rtol=1e-4,
            )
            native.specialize(image, command, depth=16)
            self.assertEqual(native.parameters(), before)
            # Explicit invalid geometry leaves the last usable definition/session.
            with self.assertRaises(ValueError):
                native.specialize(b"", command)
            self.assertEqual(native.parameters(), before)
            self.assertEqual(native.work_limit, 20_000_000)
            native.train_step([flat(ports)], [zeros], rate=0.0001)
            reference.zero_grad()
            blue, red, _ = reference(image, command, depth=16)
            _rgb_packed(blue, red).square().mean().backward()
            optimizer.step()
            with load_rgb_policy(image, command, endpoint=self.endpoint) as oracle:
                oracle.import_state_dict(reference.state_dict())
                for got, want in zip(
                    native.parameters(), oracle.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        torch.tensor(got.data),
                        torch.tensor(want.data),
                        atol=1e-5,
                        rtol=1e-4,
                    )

    def test_complete_any2any_native_encoder_dropout_thinking_heads_and_sgd(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from lightnest.architecture.native import load_artifact_core
        from lightnest.architecture.native_any2any import NativeArtifactConfig
        from lightnest.architecture.any2any import ArtifactCore, ArtifactRequest
        import torch.nn.functional as F

        torch.manual_seed(133)
        config = NativeArtifactConfig(
            width=32, heads=4, sequence_limit=256, note_slots=3, red_depths=(2, 4)
        )
        reference = ArtifactCore(**vars(config))
        raw = bytes(range(35))
        request = ArtifactRequest("json", "x", "")
        chunks = [raw[:32], raw[32:]]
        seed = 0x43211234

        # Independently implement the public counter contract, leaving all original
        # PyTorch Transformer/GRU/notebook/head parameters and autograd in place.
        def apply_mask(value, site, p):
            output = []
            for chunk, tensor in enumerate(value):
                values = []
                for i in range(tensor.numel()):
                    h = (
                        seed
                        ^ (((chunk * 16 + site) * 0x9E3779B9) & 0xFFFFFFFF)
                        ^ ((i * 0x85EBCA6B) & 0xFFFFFFFF)
                    )
                    h ^= h >> 16
                    h = (h * 0x7FEB352D) & 0xFFFFFFFF
                    h ^= h >> 15
                    h = (h * 0x846CA68B) & 0xFFFFFFFF
                    h ^= h >> 16
                    values.append(float((h >> 8) / 16777216 >= p))
                output.append(
                    tensor * torch.tensor(values).reshape(tensor.shape) / (1 - p)
                )
            return torch.stack(output)

        @contextmanager
        def seeded_dropout():
            sites = [0, 0]

            def sdpa(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                **kwargs,
            ):
                self.assertFalse(is_causal)
                logits = (q @ k.transpose(-1, -2)) * (
                    scale if scale is not None else q.shape[-1] ** -0.5
                )
                if attn_mask is not None:
                    logits = logits + attn_mask
                probability = logits.softmax(-1)
                site = sites[0] * 4
                sites[0] += 1
                if dropout_p:
                    probability = apply_mask(probability, site, dropout_p)
                return probability @ v

            def dropout(value, p=0.5, training=True, inplace=False):
                if not training:
                    return value
                index = sites[1]
                sites[1] += 1
                return apply_mask(value, (index // 3) * 4 + index % 3 + 1, p)

            with (
                patch.object(F, "scaled_dot_product_attention", sdpa),
                patch.object(F, "dropout", dropout),
            ):
                yield

        def outputs(depth=2):
            encoded, valid, _ = reference.encode_batch(chunks, [request] * len(chunks))
            pooled = (encoded * valid[..., None]).sum(1) / valid.sum(1)[:, None]
            contextual, notes = reference.contextualize(pooled[None])
            if depth is None:
                abstract, guidance, budget = reference.think_budgeted(contextual)
                heads = reference.heads(
                    contextual, guidance, budget_logits=budget["logits"]
                )
            else:
                abstract, guidance = reference.think(contextual, depth=depth)
                heads = reference.heads(contextual, guidance)
            return [
                contextual[0],
                abstract[0],
                guidance[0],
                heads["blue_logits"][0],
                heads["red_logits"][0],
                heads["stop_logits"][0],
                heads["kind_logits"][0],
                heads["summary"][0],
                heads["budget_logits"][0],
                heads["copy_paste"]["source_start"][0],
                notes["notes"][0],
                notes["context"][0],
                notes["budget_logits"][0],
            ]

        def flat(values):
            return torch.cat([torch.as_tensor(v).flatten() for v in values])

        with load_artifact_core(
            raw, request, config=config, depth=2, chunk_size=32, endpoint=self.endpoint
        ) as native:
            native.import_state_dict(reference.state_dict())
            reference.eval()
            native.eval()
            expected = outputs()
            actual = wgt.Model.forward(native, *native.pack(raw, request, seed=seed))
            torch.testing.assert_close(
                flat(actual), flat(expected).detach(), atol=1e-5, rtol=1e-4
            )
            reference.train()
            native.train()
            with seeded_dropout():
                expected = outputs()
            ports = native.pack(raw, request, seed=seed)
            actual = wgt.Model.forward(native, *ports)
            torch.testing.assert_close(
                flat(actual), flat(expected).detach(), atol=1e-5, rtol=1e-4
            )
            loss = flat(expected).square().mean()
            loss.backward()
            native.set_optimizer("sgd")
            got = native.train_step(
                [flat(ports).tolist()], [[0.0] * flat(expected).numel()], rate=0.001
            )
            self.assertAlmostEqual(got, float(loss.detach()), places=5)
            torch.optim.SGD(reference.parameters(), lr=0.001).step()
            with load_artifact_core(
                raw,
                request,
                config=config,
                depth=2,
                chunk_size=32,
                endpoint=self.endpoint,
            ) as oracle:
                oracle.import_state_dict(reference.state_dict())
                for actual, want in zip(
                    native.parameters(), oracle.parameters(), strict=True
                ):
                    self.assertEqual(actual.name, want.name)
                    torch.testing.assert_close(
                        torch.tensor(actual.data),
                        torch.tensor(want.data),
                        atol=4e-6,
                        rtol=4e-5,
                    )
            # The same definition and seed execute via the explicit Torch backend.
            portable = wgt.from_ir(native.definition, backend="torch", device="cpu")
            base = portable.to_torch().base
            base.max_work = 100_000_000
            base.memory_bytes = 256 * 1024**2
            with torch.no_grad():
                for parameter in native.parameters():
                    base.get_parameter(base.parameter_names[parameter.name]).copy_(
                        torch.tensor(parameter.data).reshape(parameter.shape)
                    )
            actual = wgt.Model.forward(native, *ports)
            same = portable(
                *[torch.tensor(p, dtype=torch.float32) for p in ports],
            )
            torch.testing.assert_close(
                flat(actual), flat(same).detach(), atol=1e-5, rtol=1e-4
            )

            # Learned depth 0 means blue-only; the last index continues shared red.
            native.specialize(raw, request, depth=None)
            for selected in [0, len(config.red_depths)]:
                with torch.no_grad():
                    reference.budget[-1].bias.fill_(-100.0)
                    reference.budget[-1].bias[selected] = 100.0
                native.import_state_dict(reference.state_dict())
                with seeded_dropout():
                    expected = outputs(None)
                ports = native.pack(raw, request, seed=seed)
                actual = wgt.Model.forward(native, *ports)
                torch.testing.assert_close(
                    flat(actual), flat(expected).detach(), atol=1e-5, rtol=1e-4
                )
            reference.zero_grad()
            loss = flat(expected).square().mean()
            loss.backward()
            native.set_optimizer("sgd")
            native.train_step(
                [flat(ports).tolist()], [[0.0] * flat(expected).numel()], rate=0.0001
            )
            torch.optim.SGD(reference.parameters(), lr=0.0001).step()
            with load_artifact_core(
                raw,
                request,
                config=config,
                depth=None,
                chunk_size=32,
                endpoint=self.endpoint,
            ) as oracle:
                oracle.import_state_dict(reference.state_dict())
                for actual, want in zip(
                    native.parameters(), oracle.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        torch.tensor(actual.data),
                        torch.tensor(want.data),
                        atol=4e-6,
                        rtol=4e-5,
                    )

    def test_any2any_default_geometry_and_live_rebinding(self):
        from lightnest.architecture.native_any2any import (
            NativeArtifactRequest,
            NativeArtifactConfig,
            load_artifact_core,
        )
        from lightnest.architecture.any2any import ArtifactCore

        torch.manual_seed(141)
        config = NativeArtifactConfig()
        request = NativeArtifactRequest("image", "x")
        raw = b"abc"
        reference = ArtifactCore(**vars(config)).eval()
        with load_artifact_core(raw, request, endpoint=self.endpoint) as native:
            native.import_state_dict(reference.state_dict())
            native.eval()
            self.assertTrue(native._compact_parameters)
            expected = reference.summarize(raw, request, depth=None)[1]
            actual = native(raw, request)
            for name in [
                "blue_logits",
                "red_logits",
                "stop_logits",
                "kind_logits",
                "summary",
                "budget_logits",
            ]:
                torch.testing.assert_close(
                    torch.tensor(actual[name]), expected[name][0], atol=2e-5, rtol=2e-4
                )
            # One owner, different sequence/request length and thinking depth.
            before = native.parameters()
            native.specialize(b"abcdef", request, depth=2)
            actual = native(b"abcdef", request)
            expected = reference.summarize(b"abcdef", request, depth=2)[1]
            torch.testing.assert_close(
                torch.tensor(actual["blue_logits"]),
                expected["blue_logits"][0],
                atol=2e-5,
                rtol=2e-4,
            )
            self.assertEqual(native.parameters(), before)
            self.assertEqual(native.work_limit, 100_000_000)
