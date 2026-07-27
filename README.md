# nanbeige-mlx

An MLX port of the **Nanbeige4.2-3B *Looped Transformer*** for Apple Silicon,
plus the HF→MLX conversion and publishing helpers. The model definition in
[`nanbeige_mlx/model.py`](nanbeige_mlx/model.py) is the **single source of truth**
for the port — it is also copied verbatim into each converted weight directory
as the `model_file` mlx-lm loads.

This is an independent project; it is not affiliated with or endorsed by the
Nanbeige team.

## What's here

| module | purpose |
|---|---|
| `nanbeige_mlx/model.py` | the port — also the shipped `model_file` |
| `nanbeige_mlx/convert.py` | HF → MLX quant (non-mutating staging, tokenizer verify) |
| `nanbeige_mlx/upload.py` | model-card + LICENSE + NOTICE + (opt-in) HF upload |
| `nanbeige_mlx/pull.py` | `pull("4bit")` → local path for `mlx_lm.load` |

## Load a published quant (one line)

```python
from nanbeige_mlx import pull
import mlx_lm
model, tok = mlx_lm.load(pull("4bit"))
```

## Convert from the BF16 checkpoint yourself

```bash
nanbeige-mlx-convert --src /path/to/Nanbeige4.2-3B --out ./nanbeige-mlx-4bit --bits 4
```

The source directory is never mutated; the tokenizer round-trip is asserted.

## The 44-slot KV cache (a real cost of the looped design)

The looped architecture needs `num_loops * num_hidden_layers = 44` KV slots.
At full context that is 44 × 8 KV-heads × 128 dim × 2 (K+V) × 262 144 positions
× 2 bytes ≈ **47 GB** — unreachable on a 16 GB machine. Because this model
supplies `make_cache`, mlx-lm's `--max-kv-size` knob is inert; use `--kv-bits`
to reduce KV precision instead. This is a quantifiable cost of the looped
design, not a bug — see `make_cache()`'s docstring.

## Dependencies

Upper-bounded to `mlx>=0.32,<0.34` and `mlx-lm>=0.31,<0.33`. The bounds are
deliberate: `model.py` uses `mlx_lm.models.base` / `.cache` internals and is
copied verbatim into every published weight repo as the `model_file`, where it
is frozen — a user who already downloaded a quant cannot receive a patch.
Relaxing the bound is a decision to make after testing against a new mlx-lm.

## Fidelity status

This port is validated by behaviour, not bit-parity: per-layer arithmetic
agrees with the reference's own `NanbeigeDecoderLayer` to fp32 precision, the
44-slot loop-aware KV cache passes a prefill-vs-incremental-decode equality
test, and the port's two code paths are bit-identical. End-to-end next-token
logit cosine against the HF reference is 0.847 (top-1 agreement 83%), lower
than a faithful port should give; six candidate causes have been eliminated by
measurement. The full record — including every falsified hypothesis — is in
the evaluation harness repo's
[`docs/investigation-log.md`](https://github.com/jishnuvenugopal/nanbeige-mlx-eval/blob/main/docs/investigation-log.md).
Behaviour on the bilingual agentic suite is unaffected (26–28/30 across
4/6/8-bit).

## License

MIT for the code in this package. The Nanbeige model weights are governed by
the upstream Apache-2.0 license; convert and redistribute them per that license.
