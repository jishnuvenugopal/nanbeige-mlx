"""Regression coverage for the opt-in Hugging Face upload helper."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import nanbeige_mlx.upload as upload_module


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"quantization": {"bits": 6, "group_size": 64}}),
        encoding="utf-8",
    )
    (model_dir / "weights").mkdir()
    (model_dir / "weights" / "model.safetensors").write_bytes(b"weights")
    return model_dir


@pytest.mark.parametrize(
    ("flags", "expected_dry_run"),
    [
        ([], True),
        (["--dry-run"], True),
        (["--yes"], False),
        (["--yes", "--dry-run"], True),
    ],
)
def test_cli_flag_precedence(monkeypatch, tmp_path, flags, expected_dry_run):
    calls = []

    def fake_upload(model_dir, repo_id, *, dry_run):
        calls.append((model_dir, repo_id, dry_run))

    monkeypatch.setattr(upload_module, "upload", fake_upload)
    upload_module.main(
        [
            "--model-dir",
            str(tmp_path),
            "--repo-id",
            "owner/model",
            *flags,
        ]
    )

    assert calls == [(str(tmp_path), "owner/model", expected_dry_run)]


def test_dry_run_manifest_excludes_ignored_files(monkeypatch, tmp_path, capsys):
    model_dir = _model_dir(tmp_path)
    ignored = [
        model_dir / "__pycache__" / "nanbeige.cpython-312.pyc",
        model_dir / "nested" / "__pycache__" / "helper.pyc",
        model_dir / ".DS_Store",
        model_dir / "nested" / ".DS_Store",
        model_dir / "nanbeige_mlx.egg-info" / "PKG-INFO",
        model_dir / "nested" / "other.egg-info" / "PKG-INFO",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(upload_module, "write_card", lambda *_args, **_kwargs: None)
    upload_module.upload(model_dir, "owner/model", dry_run=True)

    output = capsys.readouterr().out
    assert "config.json" in output
    assert "weights/model.safetensors" in output
    assert "__pycache__" not in output
    assert ".pyc" not in output
    assert ".DS_Store" not in output
    assert ".egg-info" not in output
    assert "[dry-run] no upload performed" in output


def test_upload_uses_model_repo_folder_path_and_ignore_patterns(
    monkeypatch, tmp_path
):
    model_dir = _model_dir(tmp_path)

    class FakeApi:
        def __init__(self):
            self.create_calls = []
            self.upload_calls = []

        def create_repo(self, **kwargs):
            self.create_calls.append(kwargs)

        def upload_folder(self, **kwargs):
            self.upload_calls.append(kwargs)

    api = FakeApi()
    monkeypatch.setattr(upload_module, "write_card", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api)
    )

    upload_module.upload(model_dir, "owner/model", dry_run=False)

    assert api.create_calls == [
        {"repo_id": "owner/model", "repo_type": "model", "exist_ok": True}
    ]
    assert api.upload_calls == [
        {
            "folder_path": str(model_dir),
            "repo_id": "owner/model",
            "repo_type": "model",
            "ignore_patterns": upload_module.UPLOAD_IGNORE_PATTERNS,
        }
    ]
