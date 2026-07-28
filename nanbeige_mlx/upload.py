"""Publish a converted Nanbeige MLX repo to the HuggingFace Hub.

Nanbeige4.2-3B is Apache-2.0, which permits redistribution of derivative works
(quantization is a modification), so §4(d) applies: retain the upstream
LICENSE, retain notices, and *state that you changed the files*. This module
writes proper model-card frontmatter, copies the upstream LICENSE through, and
emits a NOTICE describing the modification, then (unless ``--dry-run``) uploads.

Run with ``--dry-run`` first: it renders the card and lists the files that would
be uploaded without touching the network. The actual upload needs
``huggingface-cli login`` and is the one outward-facing step — by default this
module does NOT push, it prepares everything up to it.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


UPLOAD_IGNORE_PATTERNS = [
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    ".DS_Store",
    "**/.DS_Store",
    "*.egg-info/**",
    "**/*.egg-info/**",
]

# Apache-2.0 §4(d): state what changed.
NOTICE_TEMPLATE = """\
Nanbeige4.2-3B MLX quantization
================================

This repository contains a derivative of `{base_model}` (Apache-2.0).

Modifications by Jishnu Venugopal ({year}):
  * Converted from the original PyTorch checkpoint to Apple's MLX format.
  * Quantized to {bits}-bit (group size {group_size}) for Apple Silicon.

The original model and its Apache-2.0 license are retained (see LICENSE).
Upstream notices and the full license text are unchanged. This derivative is
redistributed under the same Apache-2.0 license.

The MLX model definition (`nanbeige.py`) is MIT-licensed source code from the
`nanbeige-mlx` project; the weights themselves remain Apache-2.0.
"""


def _card_frontmatter(bits: int, group_size: int) -> str:
    # Proper frontmatter for a bilingual quantized weights repo (C7).
    return (
        "---\n"
        "license: apache-2.0\n"
        "base_model: Nanbeige/Nanbeige4.2-3B\n"
        "language: [en, zh]\n"
        "library_name: mlx\n"
        "pipeline_tag: text-generation\n"
        f"tags: [mlx, nanbeige, looped-transformer, apple-silicon, {bits}-bit]\n"
        f"quantized_from: Nanbeige/Nanbeige4.2-3B\n"
        f"quantization:\n  bits: {bits}\n  group_size: {group_size}\n"
        "---\n"
    )


def _card_body(bits: int, group_size: int) -> str:
    return f"""
# Nanbeige4.2-3B MLX ({bits}-bit)

An MLX conversion and {bits}-bit quantization (group size {group_size}) of
[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) — a
3-billion-parameter **Looped Transformer** (`num_loops=2`, weight-shared over 22
layers for an effective depth of 44). Produced by the independent
[`nanbeige-mlx`](https://github.com/jishnuvenugopal/nanbeige-mlx) project;
not affiliated with the Nanbeige team.

## Load (no conversion step)

```python
from nanbeige_mlx import pull
import mlx_lm
model, tok = mlx_lm.load(pull("{bits}bit"))
```

Weights arrive on first use and are cached in `~/.cache/huggingface`.

## KV-cache ceiling (a real cost of the looped design)

The looped architecture needs `num_loops * num_hidden_layers = 44` KV slots.
At full context that is 44 x 8 KV-heads x 128 dim x 2 (K+V) x 262144 positions
x 2 bytes ~= **47 GB** — unreachable on a 16 GB machine. Because this model
supplies `make_cache`, mlx-lm's `--max-kv-size` knob is inert; use `--kv-bits`
to reduce KV precision instead.

## Fidelity of the port

This quant is produced by a from-scratch MLX port of the looped architecture,
checked against the HuggingFace reference. The headline finding carries over
verbatim from the port's documentation, and must read identically wherever it
appears (repo README, model card, investigation log):

> Bottom line: **the source of the logit gap is documented-open, not blocking.**
> Six candidate causes are ruled out by measurement (the seventh, "RoPE runs in
> bf16," was killed by a bit-identical upcast experiment — `mx.fast.rope` is
> fp32-internal regardless of input dtype). The `[1,1,5,6]` mask shape is a red
> herring: the 6th column is HF's `past_seen_tokens + sequence_length + 1`
> boilerplate for StaticCache sizing, present in every Llama-family model, and
> eager attention slices it to `[:,:,:,:L]` before use — nothing to do with
> `num_loops`. The gap does not affect behavior on the agentic suite. Bit-exact
> parity is not claimed and not expected in bf16. Full record:
> [`docs/investigation-log.md`](https://github.com/jishnuvenugopal/nanbeige-mlx-eval/blob/main/docs/investigation-log.md).

## License

Apache-2.0 (the upstream license; quantization is a modification under §4(d)).
The `nanbeige.py` model definition is MIT-licensed source from `nanbeige-mlx`.
See `LICENSE` and `NOTICE`.
"""


def write_card(model_dir: str | Path, bits: int, group_size: int, *, base_model: str = "Nanbeige/Nanbeige4.2-3B") -> Path:
    """Write README.md, LICENSE (copied from upstream), and NOTICE into ``model_dir``."""
    import datetime
    import shutil

    d = Path(model_dir)
    card = d / "README.md"
    card.write_text(_card_frontmatter(bits, group_size) + _card_body(bits, group_size), encoding="utf-8")

    notice = d / "NOTICE"
    notice.write_text(
        NOTICE_TEMPLATE.format(
            base_model=base_model,
            bits=bits,
            group_size=group_size,
            year=datetime.datetime.now().year,
        ),
        encoding="utf-8",
    )

    # Always ship the full canonical Apache-2.0 LICENSE text. The weights are
    # Apache-2.0 (not MIT — MIT is the code), so a 2-line pointer is not enough:
    # §4(d) requires giving recipients a copy of the license. Source the text
    # from the vendored LICENSE.apache (committed alongside this module).
    license_path = d / "LICENSE"
    candidates = [
        Path(__file__).resolve().parent.parent / "LICENSE.apache",  # port repo root
        d.parent / "nanbeige42-hf" / "LICENSE",                      # upstream checkout
        d.parent / "LICENSE.apache",
    ]
    src_license = next((c for c in candidates if c.exists()), None)
    if src_license is None:
        raise FileNotFoundError(
            "Full Apache-2.0 LICENSE text not found. Expected LICENSE.apache in the "
            "port repo root (vendored), or nanbeige42-hf/LICENSE. Do not ship a "
            "license pointer — the weights require the full license text under §4(d)."
        )
    shutil.copy2(src_license, license_path)
    return card


def upload(model_dir: str | Path, repo_id: str, *, dry_run: bool = True) -> None:
    """Write the card/LICENSE/NOTICE and (unless ``dry_run``) upload to ``repo_id``."""
    d = Path(model_dir)
    cfg = __import__("json").loads((d / "config.json").read_text(encoding="utf-8"))
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = int(q.get("bits", 0))
    group_size = int(q.get("group_size", 64))

    write_card(d, bits, group_size)

    files = sorted(
        rel
        for p in d.rglob("*")
        if p.is_file()
        and not any(
            fnmatch.fnmatch((rel := p.relative_to(d).as_posix()), pattern)
            for pattern in UPLOAD_IGNORE_PATTERNS
        )
    )
    print(f"repo: {repo_id}")
    print(f"model_dir: {d}")
    print(f"quantization: {bits}-bit, group_size={group_size}")
    print("files:")
    for name in files:
        print(f"  - {name}")

    if dry_run:
        print("\n[dry-run] no upload performed. Re-run without --dry-run to publish.")
        return

    from huggingface_hub import HfApi  # type: ignore

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(d),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=UPLOAD_IGNORE_PATTERNS,
    )
    print(f"\nuploaded to {repo_id}")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(
        prog="nanbeige-mlx-upload",
        description="Write a proper model card + LICENSE + NOTICE and upload a Nanbeige MLX quant.",
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="render the card and list files without uploading (default behavior)",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="actually upload (requires `huggingface-cli login`)",
    )
    a = ap.parse_args(argv)
    upload(a.model_dir, a.repo_id, dry_run=a.dry_run or not a.yes)


if __name__ == "__main__":  # pragma: no cover
    main()
