"""Project paths and the benchmark seed. Every path derives from the repository root; nothing is hard-coded."""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT_ENV_VAR = "VQA_LAB_ROOT"


@dataclass(frozen=True)
class Paths:
    """Project filesystem paths resolved relative to the repository root."""
    root: Path
    zip_dir: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    configs_dir: Path

    @property
    def tables_dir(self) -> Path:
        """Return the directory containing processed question and annotation tables.

        Returns:
            Path: Path to the processed tables directory (`processed_dir / 'tables'`).
        """
        return self.processed_dir / "tables"

    @property
    def image_features_dir(self) -> Path:   # decoded448/, pooled/<encoder>/, grid/<encoder>/ (each with train2014/, val2014/)
        """Return the root directory for decoded images and extracted image features.

        Returns:
            Path: Path to the image features directory (`processed_dir / 'features/image'`).
        """
        return self.processed_dir / "features/image"

    @property
    def text_features_dir(self) -> Path:   # <question arm>/ with its cached states or sentence vectors
        """Return the directory containing cached question token states and sentence vectors.

        Returns:
            Path: Path to the text features directory (`processed_dir / 'features/text'`).
        """
        return self.processed_dir / "features/text"

    @property
    def runs_dir(self) -> Path:   # <benchmark>/ with results.parquet, failed.csv and scores/
        """Return the directory containing benchmark run tables, logs, and score arrays.

        Returns:
            Path: Path to the runs artifacts directory (`artifacts_dir / 'runs'`).
        """
        return self.artifacts_dir / "runs"


def find_repo_root(start: Path | None = None) -> Path:
    """Find and return the root directory of the repository.

    Checks the `VQA_LAB_ROOT` environment variable first. If unset, searches
    ancestors of `start` (or current working directory) for `pyproject.toml`.

    Args:
        start: Directory to start searching upwards from. Defaults to the current
            working directory.

    Returns:
        Path: Resolved absolute path to the repository root directory.

    Raises:
        FileNotFoundError: If `pyproject.toml` is not found in any ancestor directory.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        f"No pyproject.toml found above {here}; run inside the repository or set {ROOT_ENV_VAR}."
    )


CONFIG_FILE = "vqa_lab.yaml"   # in <root>/configs: the `paths` section and the encoder registries
REGISTRIES = {"image": ("classical", "image"), "text": ("classical", "text"),
              "modern_image": ("modern", "image"), "modern_text": ("modern", "text")}


def load_paths(root: Path | None = None) -> Paths:
    """Load filesystem paths from the configuration file.

    Reads the `paths` section of `<root>/configs/vqa_lab.yaml`. Relative entries
    are resolved against `root`, while absolute entries are preserved.

    Args:
        root: Optional repository root path. If None, resolved via `find_repo_root()`.

    Returns:
        Paths: Dataclass instance containing all resolved project directories.
    """
    root = Path(root) if root is not None else find_repo_root()
    configs_dir = root / "configs"
    cfg = yaml.safe_load((configs_dir / CONFIG_FILE).read_text())["paths"]
    return Paths(
        root=root,
        configs_dir=configs_dir,
        **{k: root / cfg[k] for k in ("zip_dir", "raw_dir", "processed_dir", "artifacts_dir")},
    )


def load_encoders(paths: Paths, kind: str) -> list[dict]:
    """Load encoder entries from the YAML configuration registry.

    Reads the registry corresponding to `kind` ('image', 'text', 'modern_image',
    or 'modern_text') from `<configs_dir>/vqa_lab.yaml`.

    Args:
        paths: Project paths object providing `configs_dir`.
        kind: Registry identifier, one of 'image', 'text', 'modern_image',
            or 'modern_text'.

    Returns:
        list[dict]: List of encoder configuration dictionaries, where each entry
            has a 'name' field defaulted to 'key'.
    """
    path = paths.configs_dir / CONFIG_FILE
    group, part = REGISTRIES[kind]
    entries = (((yaml.safe_load(path.read_text()) or {}).get(group) or {}).get(part) or []) if path.is_file() else []
    return [{"name": e["key"], **e} for e in entries]


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary file path that replaces the target path atomically upon completion.

    Creates a hidden temporary file (`.partial-<filename>`) in the parent directory.
    When the context block exits cleanly, the temporary file replaces `path`.
    If an exception occurs, the temporary file is unlinked.

    Args:
        path: Final destination path for the file.

    Yields:
        Path: Temporary file path to write to safely before atomic rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".partial-" + path.name)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


SEED = 123   # the seed of every benchmark run, so results are comparable across encoders
