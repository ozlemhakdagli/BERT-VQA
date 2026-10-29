"""VQA Lab: Controlled benchmark of visual encoders and question representations for VQA v2."""
from __future__ import annotations

__version__ = "0.1.0"

from vqa_lab.experiment import context, forget, run  # noqa: E402,F401

__all__ = ["__version__", "context", "forget", "run"]
