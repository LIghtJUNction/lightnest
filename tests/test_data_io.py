import unittest
from lightnest_rgb.data_io import pack, unpack
from lightnest_rgb.any2any import ArtifactRequest, ArtifactCore
from lightnest_rgb.tools import ToolCall, ToolRegistry
import asyncio

class DataIOTests(unittest.TestCase):
    def test_arbitrary_named_format_and_bytes_round_trip(self):
        value = pack(b"\x00\xff", format="application/x-custom-binary", name="x.bin")
        self.assertEqual(unpack(value), b"\x00\xff")
        ArtifactRequest("application/x-custom-binary", "transform")
    def test_json_and_text_round_trip(self):
        self.assertEqual(unpack(pack({"x": 1})), {"x": 1})
        self.assertEqual(unpack(pack("hello")), "hello")
    def test_unknown_format_can_enter_model(self):
        model=ArtifactCore(width=32,heads=4,max_bytes=256)
        encoded,_=model.encode(b"abc", ArtifactRequest("vendor/custom", "understand"))
        self.assertEqual(encoded.shape[-1], 32)
    def test_async_tool_calls_are_parallelizable_and_typed(self):
        registry=ToolRegistry(); registry.register("length", lambda value: len(value))
        results=asyncio.run(registry.call_many([ToolCall("length", {"value":"abc"}, "1"), ToolCall("length", {"value":"xy"}, "2")]))
        self.assertEqual([r.output for r in results], [3, 2]); self.assertEqual(registry.describe(), ["length"])
if __name__ == '__main__': unittest.main()
