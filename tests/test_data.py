"""Tests for vqa_lab.data: tables, splits, vocabulary, soft targets, and prepare_data."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vqa_lab.config import SEED, Paths
from vqa_lab.data import (
    extract_archive, fetch_archive, fetch_hub,
    SPLIT_FILES, TableBuildError, add_subset_columns, assert_image_disjoint,
    build_split_table, build_vocab, holdout_mask,
    prepare_data, soft_target,
)
from vqa_lab.metrics import normalize_answer, score_question


def q_dict(qid: int, img_id: int) -> dict:
    """Generate a synthetic question dictionary for tests.

    Args:
        qid: Question identifier.
        img_id: Associated image identifier.

    Returns:
        dict[str, Any]: Question metadata dictionary.
    """
    return {"question_id": qid, "image_id": img_id, "question": f"Q {qid}?"}


def a_dict(qid: int, img_id: int, answers: list[str], atype: str = "other") -> dict:
    """Generate a synthetic annotation dictionary for tests.

    Args:
        qid: Question identifier.
        img_id: Associated image identifier.
        answers: List of raw answer strings.
        atype: Answer category type string.

    Returns:
        dict[str, Any]: Annotation metadata dictionary.
    """
    return {"question_id": qid, "image_id": img_id, "question_type": "what is", "answer_type": atype,
            "multiple_choice_answer": answers[0],
            "answers": [{"answer": a, "answer_id": i + 1} for i, a in enumerate(answers)]}


def write_qa(path: Path, questions: list, annotations: list | None = None) -> tuple[Path, Path | None]:
    """Write synthetic question and annotation JSON files.

    Args:
        path: Directory where files should be written.
        questions: List of question dictionaries.
        annotations: Optional list of annotation dictionaries.

    Returns:
        tuple[Path, Path | None]: Paths to the written question and annotation files.
    """
    q, a = path / "q.json", (path / "a.json") if annotations is not None else None
    q.write_text(json.dumps({"questions": questions}))
    if a:
        a.write_text(json.dumps({"annotations": annotations}))
    return q, a


def write_synthetic_data(raw_dir: Path) -> None:
    """Write a complete synthetic VQA dataset split into raw_dir.

    Args:
        raw_dir: Target directory for synthetic question and annotation files.
    """
    (raw_dir / "questions").mkdir(parents=True, exist_ok=True)
    (raw_dir / "annotations").mkdir(parents=True, exist_ok=True)
    pool, offset = ["cat", "dog", "yes", "no", "red", "blue", "white", "black", "1", "2"], 0
    counts = {"train": 400, "val": 50, "test_dev": 10, "test": 10}
    for split, (q_name, a_name) in SPLIT_FILES.items():
        n = counts[split]
        ids, offset = list(range(offset + 1, offset + n + 1)), offset + n
        (raw_dir / "questions" / q_name).write_text(json.dumps({"questions": [q_dict(i, i) for i in ids]}))
        if a_name:
            ann = [a_dict(i, i, [pool[i % 10]] * 10) for i in ids]
            (raw_dir / "annotations" / a_name).write_text(json.dumps({"annotations": ann}))


def test_tables_join_annotations_by_question_id(tmp_path: Path) -> None:
    """Verify build_split_table joins questions and annotations with normalized answers.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    q, a = write_qa(
        tmp_path, [q_dict(1, 10), q_dict(2, 10), q_dict(3, 11)],
        [a_dict(3, 11, ["cat"] * 10), a_dict(1, 10, ["The Dog."] * 10, "yes/no"), a_dict(2, 10, ["2"] * 10, "number")],
    )
    df = build_split_table("val", q, a).set_index("question_id")
    cols = ["question_id", "image_id", "split_name", "question", "question_type",
            "answer_type", "majority_answer", "answers_raw", "answer_ids", "answers_norm"]
    assert list(df.reset_index().columns) == cols
    assert df.loc[1, "split_name"] == "val" and df.loc[1, "answers_norm"] == ["dog"] * 10
    assert df.loc[1, "answer_ids"] == list(range(1, 11)) and df.loc[2, "answer_type"] == "number"
    assert df.loc[3, "majority_answer"] == "cat"
    assert list(build_split_table("test", q).columns) == ["question_id", "image_id", "split_name", "question"]


@pytest.mark.parametrize("annotations", [
    [a_dict(1, 10, ["a"] * 10)],
    [a_dict(1, 10, ["a"] * 10), a_dict(2, 10, ["a"] * 10), a_dict(9, 10, ["a"] * 10)],
    [a_dict(1, 10, ["a"] * 10), a_dict(2, 99, ["a"] * 10)],
])
def test_inconsistent_files_raise(tmp_path: Path, annotations: list) -> None:
    """Verify build_split_table raises TableBuildError on inconsistent annotations.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        annotations: Inconsistent annotation test fixture list.
    """
    q, a = write_qa(tmp_path, [q_dict(1, 10), q_dict(2, 10)], annotations)
    with pytest.raises(TableBuildError):
        build_split_table("train", q, a)


def test_holdout_is_image_disjoint_and_row_order_free() -> None:
    """Verify holdout split is strictly image-disjoint, ratio-accurate, and order-invariant."""
    from sklearn.model_selection import GroupShuffleSplit

    ids = pd.Series([*range(1, 2001), 5, 5, 100, 100])
    mask = holdout_mask(ids)
    assert mask.dtype == bool and len(mask) == len(ids)
    hold_imgs = set(ids[mask])
    fit_imgs = set(ids[~mask])
    assert hold_imgs.isdisjoint(fit_imgs)   # no image on both sides
    assert 0.045 <= len(hold_imgs) / ids.nunique() <= 0.055   # within 0.5 points of 5 %

    shuffled = ids.sample(frac=1, random_state=0)
    shuf_mask = holdout_mask(shuffled)
    assert set(shuffled[shuf_mask]) == hold_imgs   # same set of holdout images after shuffling rows
    assert shuf_mask.to_dict() == mask.loc[shuffled.index].to_dict()

    unique = np.sort(ids.unique())
    _, expected_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=SEED).split(unique, groups=unique))
    assert hold_imgs == set(unique[expected_idx])

    mask_10 = holdout_mask(ids, percent=10)
    assert 0.095 <= len(set(ids[mask_10])) / ids.nunique() <= 0.105


def test_subsets_are_per_image_and_overlaps_raise() -> None:
    """Verify add_subset_columns partitions per-image and assert_image_disjoint validates overlap."""
    train, val = add_subset_columns(pd.DataFrame({"image_id": np.repeat(np.arange(1, 2001), 3)}),
                                    pd.DataFrame({"image_id": [10_001, 10_002]}))
    assert set(train["subset"]) == {"fit", "holdout"} and (val["subset"] == "report").all()
    assert train.groupby("image_id")["subset"].nunique().max() == 1
    assert_image_disjoint(a=pd.DataFrame({"image_id": [1, 2]}), b=pd.DataFrame({"image_id": [3]}))
    with pytest.raises(ValueError, match="'a' and 'c'"):
        assert_image_disjoint(a=pd.DataFrame({"image_id": [1, 2]}), c=pd.DataFrame({"image_id": [2, 9]}))


def test_answer_list_counts_fit_rows_only_by_frequency() -> None:
    """Verify build_vocab filters to fit subset, normalizes answers, and breaks ties alphabetically."""
    rows = [("yes", "fit")] * 20 + [("no", "fit")] * 20 + [("Two", "fit")] * 7 + [("2", "fit")] * 5 \
        + [("dog", "fit")] * 9 + [("cat", "fit")] * 8 + [("holdout_only", "holdout")] * 50
    df = pd.DataFrame(rows, columns=["majority_answer", "subset"])
    assert build_vocab(df) == ["no", "yes", "2", "dog", "cat"]   # fit rows only, "Two" counted as "2", ties A-Z
    assert build_vocab(df, k=3) == ["no", "yes", "2"] and build_vocab(df, k=2) == ["no", "yes"]
    with pytest.raises(ValueError):
        build_vocab(df.drop(columns="subset"))


def test_soft_target_equals_the_official_answer_score() -> None:
    """Verify soft_target sparse target vector computation matches official scoring rules."""
    idx, sc = soft_target(["yes"] * 3 + ["no"] * 7, {"yes": 0, "no": 1})
    assert idx.tolist() == [0, 1] and sc.tolist() == pytest.approx([0.9, 1.0])
    idx, sc = soft_target(["cat"] + ["dog"] * 9, {"cat": 0, "dog": 1})
    assert sc.tolist() == pytest.approx([0.3, 1.0])
    assert soft_target(["fish"] * 10, {"cat": 0})[0].size == 0
    assert soft_target(["yes"] * 10, {"yes": 0})[1].tolist() == [1.0] and score_question("yes", ["Yes"] * 10) == 0.0
    rng, words = random.Random(0), ["a", "b", "c", "d", "e", "f"]
    index = {w: i for i, w in enumerate(words)}
    for _ in range(200):
        answers = [rng.choice(words) for _ in range(10)]
        idx, sc = soft_target(answers, index)
        for i, s in zip(idx, sc):
            assert score_question(words[i], answers) == pytest.approx(s)


def test_prepare_data_and_parquet_cache(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify prepare_data pipeline, target tensor construction, and parquet caching.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for intercepting table rebuilding.
    """
    write_synthetic_data(paths.raw_dir)
    data = prepare_data(paths)
    assert len(data.hold) > 0 and set(data.fit["image_id"]).isdisjoint(set(data.hold["image_id"]))
    assert len(data.val) == 50
    assert len(data.vocab) == 10
    assert data.targets.shape == (len(data.fit), len(data.vocab)) and data.targets.dtype == torch.float16
    assert (data.targets.max(dim=1).values > 0).all()
    assert all(normalize_answer(a) in data.vocab for a in data.fit["majority_answer"])
    index = {a: i for i, a in enumerate(data.vocab)}
    for r, answers in enumerate(data.fit["answers_norm"]):   # each target row is the official soft score of its answers
        idx, scores = soft_target(answers, index)
        expected = torch.zeros(len(data.vocab), dtype=torch.float16)
        if len(idx):
            expected[torch.as_tensor(idx, dtype=torch.long)] = torch.as_tensor(scores, dtype=torch.float16)
        assert torch.equal(data.targets[r], expected)
    data_ans = prepare_data(paths, answers=3)
    assert len(data_ans.vocab) == 3 and data_ans.targets.shape[1] == 3
    data_lim = prepare_data(paths, limit=50)
    assert len(data_lim.train) == 50 and len(data_lim.val) == 12
    assert all((paths.tables_dir / f"{split}.parquet").is_file() for split in SPLIT_FILES)
    assert not list(paths.tables_dir.glob(".partial-*"))   # no temporary file left behind
    def _fail_build(p):
        """Sentinel mock that raises AssertionError if parquet cache is bypassed.

        Args:
            p: Paths configuration object.

        Raises:
            AssertionError: Unconditionally to verify parquet cache usage.
        """
        raise AssertionError("parquet cache should be used")
    monkeypatch.setattr("vqa_lab.data.build_all_tables", _fail_build)
    data2 = prepare_data(paths)
    assert data.fit["question_id"].tolist() == data2.fit["question_id"].tolist()
    assert data.val["question_id"].tolist() == data2.val["question_id"].tolist()
    assert torch.equal(data.targets, data2.targets)


def test_fetch_hub_skips_finished_and_resumes_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fetch_hub skips existing completed repositories and resumes interrupted downloads.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        monkeypatch: Monkeypatch fixture for mocking snapshot_download.
    """
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kw: calls.append((repo, Path(kw["local_dir"]), kw.get("allow_patterns"))))
    done = tmp_path / "bert"
    done.mkdir()
    (done / "model.safetensors").write_text("w")
    assert fetch_hub("org/bert", done, needed=["model.safetensors"]) == done and calls == []
    assert fetch_hub("org/bert", done) == done and calls == []   # an existing folder counts as finished ...
    marker = done / ".cache/huggingface/download/sub/abc.etag.incomplete"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    fetch_hub("org/bert", done)   # ... unless a download in it was interrupted
    fetch_hub("org/vlm", tmp_path / "vlm", patterns=["*.json"])   # a missing folder
    fetch_hub("org/bert", done, needed=["model.safetensors", "config.json"])   # a needed file is missing
    assert calls == [("org/bert", done, None), ("org/vlm", tmp_path / "vlm", ["*.json"]), ("org/bert", done, None)]


def test_fetch_and_extract_archive(tmp_path: Path) -> None:
    """Verify archive downloading, caching, idempotency, and integrity-checked extraction.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    import zipfile

    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w") as z:
        z.writestr("sub/", "")
        z.writestr("sub/a.txt", "hello world")
        z.writestr("b.txt", "second")

    target_zip = tmp_path / "zips/archive.zip"
    assert fetch_archive(src_zip.as_uri(), target_zip) == target_zip
    assert target_zip.read_bytes() == src_zip.read_bytes()

    src_zip.write_bytes(b"changed")   # existing target is skipped, not re-downloaded
    assert fetch_archive(src_zip.as_uri(), target_zip) == target_zip
    assert target_zip.read_bytes() != b"changed"

    dest = tmp_path / "raw"
    assert extract_archive(target_zip, dest) is True
    assert (dest / "sub/a.txt").read_text() == "hello world" and (dest / "b.txt").read_text() == "second"
    assert extract_archive(target_zip, dest) is False   # finished extraction is skipped

    (dest / "sub/a.txt").write_text("short")   # truncated file from an interrupted extraction -> redone
    assert extract_archive(target_zip, dest) is True
    assert (dest / "sub/a.txt").read_text() == "hello world"

