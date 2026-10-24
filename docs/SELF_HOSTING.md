# Self-hosting ELF

LightNest can be built as one Linux x86-64 ELF that reads `/proc/self/exe` at
runtime. The file contains the trained Any2Any checkpoint, a GGUF language
model, the llama.cpp server, and all local CPU backend variants. The runtime
passes a compact ELF observation and the learned Any2Any observation to the
embedded language model through the native `/v1/responses` protocol; it does
not assemble an explanation from a template or use the legacy chat endpoint.

Build with `uv` and provide the checkpoint and GGUF explicitly:

```bash
mkdir -p /home/lightjunction/lightnest-self-build-tmp
TMPDIR=/home/lightjunction/lightnest-self-build-tmp \
  uv run --locked python tools/build_self.py \
  --any2any-checkpoint /path/to/any2any.pt \
  --language-model /path/to/language-model.gguf \
  --output /path/to/lightnest-self
```

Run the resulting ELF directly:

```bash
/path/to/lightnest-self
```

The current local validation uses the Gemma4 e2b GGUF and the v12 Any2Any
checkpoint. The output reports facts parsed from the executable, the learned
artifact type, sampled byte count, learned context slots, and learned compute
depth, then separates model inference from claims unsupported by the
observation. The model can explain the executable's observed structure; this
does not prove that it has recovered every source-level purpose or behavior.
