"""Tests for repository path resolution and configuration loading."""
from pathlib import Path

import numpy as np
import pytest

from vqa_lab.config import atomic_path, find_repo_root, load_encoders, load_paths


def test_find_repo_root_precedence(tmp_path, monkeypatch):
    """Verify repository root discovery precedence (env var > ancestor pyproject.toml > error).

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").touch()
    nested, other = root / "sub", tmp_path / "other"
    nested.mkdir()
    other.mkdir()

    monkeypatch.chdir(other)
    monkeypatch.setenv("VQA_LAB_ROOT", str(root))
    assert find_repo_root() == root.resolve()

    monkeypatch.delenv("VQA_LAB_ROOT", raising=False)
    monkeypatch.chdir(nested)
    assert find_repo_root() == root.resolve()

    monkeypatch.chdir(other)
    with pytest.raises(FileNotFoundError, match="VQA_LAB_ROOT"):
        find_repo_root()


def test_atomic_path(tmp_path):
    """Test that atomic_path writes to hidden temporary file and renames cleanly on success.

    Also verifies cleanup on exception and compatibility with numpy save.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    target = tmp_path / "sub" / "results.parquet"
    with atomic_path(target) as tmp:
        assert tmp.name == ".partial-results.parquet" and not tmp.match("results*.parquet")
        tmp.write_text("v1")
    assert target.read_text() == "v1" and not tmp.exists()
    with pytest.raises(KeyboardInterrupt):     # a stopped notebook in the middle of a write
        with atomic_path(target) as tmp:
            tmp.write_text("half")
            raise KeyboardInterrupt
    assert target.read_text() == "v1" and not tmp.exists()   # the previous file survives and the partial file is cleaned up
    with atomic_path(tmp_path / "scores.npy") as tmp:
        np.save(tmp, np.arange(3))                # numpy keeps the name: no "x.partial.npy.npy"
    assert np.load(tmp_path / "scores.npy").tolist() == [0, 1, 2]


def test_config_file(tmp_path):
    """Test loading paths and encoder registries from vqa_lab.yaml configuration file.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/vqa_lab.yaml").write_text(
        "paths: {zip_dir: data/z, raw_dir: /abs/raw, processed_dir: data/p, artifacts_dir: art}\n"
        "classical:\n  image:\n    - {key: m1, family: F, tier: 1}\n    - {key: m2, name: timm-m2, family: F, tier: 2, note: kept}\n"
        "  text:\n    - {key: t1, kind: static, tier: 1}\n")
    p = load_paths(tmp_path)
    assert (p.root, p.zip_dir, p.raw_dir, p.processed_dir, p.artifacts_dir, p.configs_dir, p.tables_dir) == (
        tmp_path, tmp_path / "data/z", Path("/abs/raw"), tmp_path / "data/p", tmp_path / "art", tmp_path / "configs",
        tmp_path / "data/p/tables")
    assert (p.image_features_dir, p.text_features_dir, p.runs_dir) == (
        tmp_path / "data/p/features/image", tmp_path / "data/p/features/text", tmp_path / "art/runs")
    assert load_encoders(p, "image") == [{"key": "m1", "name": "m1", "family": "F", "tier": 1},
                                         {"key": "m2", "name": "timm-m2", "family": "F", "tier": 2, "note": "kept"}]
    assert load_encoders(p, "text") == [{"key": "t1", "name": "t1", "kind": "static", "tier": 1}]
    assert load_encoders(p, "modern_text") == []
