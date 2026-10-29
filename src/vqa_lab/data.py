"""From the VQA v2 JSON files to `Data`: question tables, the image-disjoint fit/holdout split, the answer list and soft targets."""
from __future__ import annotations

import itertools
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch

from vqa_lab.config import SEED, Paths, atomic_path
from vqa_lab.metrics import normalize_answer, soft_score


SPLIT_FILES = {  # split -> (questions file, annotations file or None)
    "train": ("v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json"),
    "val": ("v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json"),
    "test_dev": ("v2_OpenEnded_mscoco_test-dev2015_questions.json", None),
    "test": ("v2_OpenEnded_mscoco_test2015_questions.json", None),
}


class TableBuildError(ValueError):
    """Questions and annotations do not describe the same questions."""


def build_split_table(split: str, questions_path: Path, annotations_path: Path | None = None) -> pd.DataFrame:
    """Build a consolidated DataFrame for a dataset split from raw questions and annotations.

    Each row represents one question. When annotations are provided, includes question_type,
    answer_type, majority_answer, raw human answers, answer IDs, and normalized answers.

    Args:
        split: Dataset split name (e.g., 'train', 'val', 'test_dev', 'test').
        questions_path: Path to the raw questions JSON file.
        annotations_path: Path to the annotations JSON file, or None for test splits.

    Returns:
        pd.DataFrame: Table with questions and optional merged annotation columns.

    Raises:
        TableBuildError: If question and annotation IDs do not match or reference
            different image IDs.
    """
    df = pd.DataFrame(json.loads(Path(questions_path).read_text())["questions"])
    df.insert(2, "split_name", split)
    if annotations_path is None:
        return df
    ann = json.loads(Path(annotations_path).read_text())["annotations"]
    a = pd.DataFrame({
        "question_id": [x["question_id"] for x in ann],
        "ann_image_id": [x["image_id"] for x in ann],
        "question_type": [x["question_type"] for x in ann],
        "answer_type": [x["answer_type"] for x in ann],
        "majority_answer": [x["multiple_choice_answer"] for x in ann],
        "answers_raw": [[y["answer"] for y in x["answers"]] for x in ann],
        "answer_ids": [[y["answer_id"] for y in x["answers"]] for x in ann],
    })
    if len(a) != len(df) or set(a["question_id"]) != set(df["question_id"]):
        raise TableBuildError(f"{split}: questions ({len(df)}) and annotations ({len(a)}) cover different question ids")
    df = df.merge(a, on="question_id", validate="one_to_one")  # by id, never by list position
    bad = df[df["image_id"] != df["ann_image_id"]]
    if len(bad):
        raise TableBuildError(f"{split}: {len(bad)} questions name a different image than their annotation, "
                              f"e.g. question_id {bad['question_id'].iloc[0]}")
    df["answers_norm"] = [[normalize_answer(x) for x in xs] for xs in df["answers_raw"]]
    return df.drop(columns="ann_image_id")


def build_all_tables(paths: Paths) -> dict[str, pd.DataFrame]:
    """Build DataFrames for all standard splits from raw JSON files.

    Processes 'train', 'val', 'test_dev', and 'test' splits using files
    found in `paths.raw_dir / 'questions'` and `paths.raw_dir / 'annotations'`.

    Args:
        paths: Project paths object containing raw directory paths.

    Returns:
        dict[str, pd.DataFrame]: Mapping from split name to its built DataFrame.
    """
    questions, annotations = paths.raw_dir / "questions", paths.raw_dir / "annotations"
    return {split: build_split_table(split, questions / q, annotations / a if a else None)
            for split, (q, a) in SPLIT_FILES.items()}


def write_tables(paths: Paths, tables: Mapping[str, pd.DataFrame]) -> None:
    """Write split tables to parquet files atomically in the processed tables directory.

    Args:
        paths: Project paths object providing `tables_dir`.
        tables: Mapping from split name (e.g. 'train', 'val') to DataFrame.
    """
    for split, df in tables.items():
        with atomic_path(paths.tables_dir / f"{split}.parquet") as tmp:
            df.to_parquet(tmp)


def read_table(paths: Paths, split: str) -> pd.DataFrame:
    """Read a cached split table from a parquet file.

    List columns are restored as numpy arrays (strings, or integers for `answer_ids`).

    Args:
        paths: Project paths object.
        split: Split name (e.g., 'train', 'val').

    Returns:
        pd.DataFrame: Loaded dataset table.
    """
    return pd.read_parquet(paths.tables_dir / f"{split}.parquet")


def load_tables(paths: Paths) -> dict[str, pd.DataFrame]:
    """Load split tables from parquet cache if present, otherwise build and cache from raw JSON.

    Args:
        paths: Project paths object.

    Returns:
        dict[str, pd.DataFrame]: Mapping from split name to split DataFrame.
    """
    if all((paths.tables_dir / f"{split}.parquet").exists() for split in SPLIT_FILES):
        return {split: read_table(paths, split) for split in SPLIT_FILES}
    tables = build_all_tables(paths)
    write_tables(paths, tables)
    return tables


def fetch_hub(repo: str, target: Path, *, needed: Sequence[str] | None = None, patterns: Sequence[str] | None = None) -> Path:
    """Ensure a Hugging Face Hub repository is downloaded to the target directory.

    Resumes interrupted downloads. Skips downloading if all `needed` files exist
    or if no unfinished downloads are detected.

    Args:
        repo: Hugging Face Hub repository identifier (e.g. 'timm/resnet50').
        target: Local directory to store the repository files.
        needed: Optional list of required filenames to check for completion.
        patterns: Optional filename patterns to filter downloads.

    Returns:
        Path: The target folder.
    """
    target = Path(target)
    if needed is not None:
        finished = all((target / f).is_file() for f in needed)
    else:   # huggingface_hub keeps the parts of an unfinished file as .cache/huggingface/download/**/*.incomplete
        finished = target.is_dir() and not any((target / ".cache/huggingface/download").rglob("*.incomplete"))
    if not finished:
        import huggingface_hub

        huggingface_hub.snapshot_download(repo, local_dir=str(target), allow_patterns=list(patterns) if patterns else None)
    return target


def fetch_archive(url: str, target: Path) -> Path:
    """Download a remote archive file to `target` atomically if not already present.

    Args:
        url: URL of the archive file to download.
        target: Local destination file path.

    Returns:
        Path: The target file.
    """
    target = Path(target)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(target))
    return target


def extract_archive(zip_path: Path, dest: Path) -> bool:
    """Unpack a zip archive into the destination directory unless already extracted.

    Checks if all non-directory files from the archive exist in `dest` with matching sizes.

    Args:
        zip_path: Path to the source zip file.
        dest: Destination directory to extract contents into.

    Returns:
        bool: True if files were extracted, False if extraction was skipped.
    """
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as z:
        if all((dest / m.filename).is_file() and (dest / m.filename).stat().st_size == m.file_size
               for m in z.infolist() if not m.is_dir()):
            return False
        z.extractall(dest)
        return True


HOLDOUT_PERCENT = 5


def holdout_mask(image_ids: pd.Series, percent: int = HOLDOUT_PERCENT) -> pd.Series:
    """Generate a boolean mask assigning images to the holdout validation split.

    Uses `GroupShuffleSplit` on sorted unique image IDs to guarantee image-disjoint splits.

    Args:
        image_ids: Series of integer image IDs for each question.
        percent: Approximate percentage of images to assign to holdout. Defaults to 5.

    Returns:
        pd.Series: Boolean mask where True indicates rows belonging to the holdout set.
    """
    unique = np.sort(image_ids.unique())
    _, hold_idx = next(GroupShuffleSplit(n_splits=1, test_size=percent / 100, random_state=SEED).split(unique, groups=unique))
    return image_ids.isin(set(unique[hold_idx])).astype(bool)


def add_subset_columns(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign 'subset' labels to training and validation splits.

    Adds a 'subset' column to `train` partitioned into 'fit' and 'holdout' by image,
    and labels `val` rows as 'report'.

    Args:
        train: Training split DataFrame.
        val: Validation split DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Updated (train, val) DataFrames with 'subset' column.
    """
    train = train.assign(subset=np.where(holdout_mask(train["image_id"]), "holdout", "fit"))
    return train, val.assign(subset="report")


def assert_image_disjoint(**tables: pd.DataFrame) -> None:
    """Verify that no two provided dataset tables share any common image IDs.

    Args:
        **tables: Named keyword arguments mapping table names to DataFrames.

    Raises:
        ValueError: If any pair of tables contains overlapping image IDs.
    """
    images = {name: set(df["image_id"]) for name, df in tables.items()}
    for (a, ids_a), (b, ids_b) in itertools.combinations(images.items(), 2):
        if shared := ids_a & ids_b:
            raise ValueError(f"'{a}' and '{b}' share {len(shared)} images, e.g. {sorted(shared)[:5]}")


def build_vocab(train: pd.DataFrame, k: int = 1000) -> list[str]:
    """Construct the fixed answer vocabulary of top-k frequent answers from fit rows.

    Counts normalized majority answers on rows where subset == 'fit', breaking ties
    alphabetically.

    Args:
        train: Training DataFrame with 'subset' and 'majority_answer' columns.
        k: Maximum number of vocabulary answers to retain. Defaults to 1000.

    Returns:
        list[str]: The top-k normalized answers, most frequent first (ties alphabetical).

    Raises:
        ValueError: If the 'subset' column is missing from `train`.
    """
    if "subset" not in train.columns:
        raise ValueError("train needs a 'subset' column (see add_subset_columns)")
    counts = Counter(normalize_answer(a) for a in train.loc[train["subset"] == "fit", "majority_answer"])
    return sorted(counts, key=lambda a: (-counts[a], a))[:k]


def soft_target(answers_norm: Sequence[str], vocab_index: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute sparse soft target scores for vocabulary answers given by human annotators.

    Calculates the official leave-one-out soft score for every answer present in
    the vocabulary. Answers outside the vocabulary receive zero score.

    Args:
        answers_norm: Sequence of normalized human answers for a single question.
        vocab_index: Mapping from normalized answer string to vocabulary index.

    Returns:
        tuple[np.ndarray, np.ndarray]: Ascending int32 vocabulary indices and
            corresponding float32 soft target values.
    """
    n = len(answers_norm)
    scores = {vocab_index[a]: soft_score(c, n) for a, c in Counter(answers_norm).items() if a in vocab_index}
    idx = sorted(scores)
    return np.array(idx, dtype=np.int32), np.array([scores[i] for i in idx], dtype=np.float32)


@dataclass
class Data:
    """`fit` and `hold` are the train rows of each subset; `fit` keeps only questions whose majority answer is in `vocab`, and
    its index is the row position in `train`. `targets` [len(fit), len(vocab)] holds the soft targets of the `fit` rows.
    `questions` is the train, val and test-dev question text, read only to build the word list of the static word vectors."""
    train: pd.DataFrame
    val: pd.DataFrame
    fit: pd.DataFrame
    hold: pd.DataFrame
    questions: pd.Series
    vocab: list[str]
    targets: torch.Tensor


def prepare_data(paths: Paths, limit: int | None = None, answers: int = 1000) -> Data:
    """Load, split, and prepare all dataset tables, answer vocabulary, and soft targets.

    Loads tables, performs image-disjoint fit/holdout partitioning on the training set,
    builds the answer vocabulary, filters fit questions to vocabulary answers, and
    precomputes soft target tensors.

    Args:
        paths: Project paths configuration.
        limit: Optional integer limit for sampling questions (for fast debugging/smoke runs).
        answers: Vocabulary size (number of top answers). Defaults to 1000.

    Returns:
        Data: Container with train, val, fit, holdout splits, vocab, and targets tensor.
    """
    tables = load_tables(paths)
    train, val = add_subset_columns(tables["train"], tables["val"])
    if limit:
        train = train.sample(limit, random_state=0).reset_index(drop=True)
        val = val.sample(limit // 4, random_state=0).reset_index(drop=True)
    fit = train[train["subset"] == "fit"]
    hold = train[train["subset"] == "holdout"]
    assert_image_disjoint(fit=fit, hold=hold, val=val)
    vocab = build_vocab(train, k=answers)
    index = {a: i for i, a in enumerate(vocab)}
    fit = fit[fit["majority_answer"].map(lambda a: normalize_answer(a) in index)]
    targets = torch.zeros((len(fit), len(vocab)), dtype=torch.float16)
    per_row = [soft_target(answers, index) for answers in fit["answers_norm"]]
    if any(lengths := [len(c) for c, _ in per_row]):
        rows = np.repeat(np.arange(len(fit)), lengths)
        cols = np.concatenate([c for c, _ in per_row]).astype(np.int64)
        targets[rows, cols] = torch.as_tensor(np.concatenate([s for _, s in per_row]), dtype=torch.float16)
    questions = pd.concat(
        [tables["train"]["question"], tables["val"]["question"], tables["test_dev"]["question"]],
        ignore_index=True,
    )
    return Data(
        train=train,
        val=val,
        fit=fit,
        hold=hold,
        questions=questions,
        vocab=vocab,
        targets=targets,
    )
