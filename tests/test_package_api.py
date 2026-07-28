from __future__ import annotations

import nanbeige_mlx


def test_top_level_pull_export_is_callable() -> None:
    from nanbeige_mlx import pull

    assert callable(pull)
    assert pull is nanbeige_mlx.pull
