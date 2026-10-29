"""Pytest fixtures shared across the test suite."""
import pytest

from vqa_lab.config import Paths


@pytest.fixture
def paths(tmp_path):
    """Provide a mock project Paths layout rooted under a temporary directory.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Paths: Mock Paths configuration instance.
    """
    return Paths(tmp_path, tmp_path / "zip", tmp_path / "raw", tmp_path / "processed", tmp_path / "artifacts", tmp_path / "configs")
