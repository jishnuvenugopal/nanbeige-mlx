"""nanbeige-mlx: an MLX port of the Nanbeige4.2-3B Looped Transformer.

The model definition lives in :mod:`nanbeige_mlx.model` and is the single
source of truth — it is also copied verbatim into each converted weight
directory as the ``model_file`` mlx-lm loads. Conversion (HF -> MLX quant) is
in :mod:`nanbeige_mlx.convert`; publishing helpers in
:mod:`nanbeige_mlx.upload` and :mod:`nanbeige_mlx.pull`.
"""

from __future__ import annotations

from .pull import pull

__version__ = "0.2.1"

__all__ = ["pull", "__version__"]
