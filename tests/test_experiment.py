"""Tests for high-level experiment API."""
import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from vqa_lab.config import SEED, Paths
from vqa_lab.data import Data
import vqa_lab.experiment as ex
from vqa_lab.experiment import (
    HEADS, Context, _image_source, _question_source, _resolve_name, _targets,
    context, forget, run,
)
from vqa_lab.image import grid_split, pooled_split
from vqa_lab.train import Rows


@pytest.fixture(autouse=True)
def small_h1(monkeypatch):
    """A small H1 (the default is 13M parameters): the tests check the plumbing, not the model."""
    monkeypatch.setattr(ex, "H1", functools.partial(ex.H1, hidden=16, fc=32))


def make_synthetic_context(tmp_path: Path | Paths) -> Context:
    """Build a synthetic Experiment Context with small feature files and tables.

    Args:
        tmp_path: Root temporary directory or Paths object.

    Returns:
        Context: Synthetic experiment execution context.
    """
    paths = tmp_path if isinstance(tmp_path, Paths) else Paths(
        tmp_path, tmp_path / "zip", tmp_path / "raw", tmp_path / "processed", tmp_path / "artifacts", tmp_path / "configs"
    )
    rng = np.random.default_rng(42)
    for split, n, ids in [
        ("train2014", 12, np.arange(1, 13, dtype=np.int64)),
        ("val2014", 5, np.arange(101, 106, dtype=np.int64)),
    ]:
        p_dir = paths.processed_dir / f"features/image/pooled/tiny/{split}"
        d_dir = paths.processed_dir / f"features/image/decoded448/{split}"
        p_dir.mkdir(parents=True, exist_ok=True)
        d_dir.mkdir(parents=True, exist_ok=True)
        np.save(p_dir / "pooled.npy", rng.standard_normal((n, 8)).astype(np.float16))
        np.save(p_dir / "ids.npy", ids)
        np.save(d_dir / "ids.npy", ids)

    vocab = ["yes", "no", "1", "2", "cat", "dog"]
    index = {a: i for i, a in enumerate(vocab)}
    type_pool = [("is this", "yes/no")] * 2 + [("how many", "number")] * 2 + [("what is", "other")] * 2

    def _make_df(q_ids, img_ids, prefix, subset):
        """Construct a synthetic subset DataFrame with questions and answers.

        Args:
            q_ids: Sequence of question IDs.
            img_ids: Sequence of image IDs.
            prefix: String prefix to prepend to questions.
            subset: Data subset split label.

        Returns:
            pd.DataFrame: Formatted VQA annotations DataFrame.
        """
        maj = [vocab[i % len(vocab)] for i in range(len(q_ids))]
        return pd.DataFrame({
            "question_id": q_ids, "image_id": img_ids, "question": [f"{prefix}question {i}?" for i in q_ids],
            "majority_answer": maj, "answers_raw": [[a] * 10 for a in maj], "answers_norm": [[a] * 10 for a in maj],
            "question_type": [type_pool[i % 6][0] for i in range(len(q_ids))],
            "answer_type": [type_pool[i % 6][1] for i in range(len(q_ids))], "subset": [subset] * len(q_ids),
        })

    fit_df = _make_df(list(range(1, 49)), [(i % 9) + 1 for i in range(48)], "", "fit")
    hold_df = _make_df(list(range(49, 61)), [10 + (i % 3) for i in range(12)], "", "holdout")
    train_df = pd.concat([fit_df, hold_df], ignore_index=True)
    val_df = _make_df(list(range(101, 121)), [101 + (i % 5) for i in range(20)], "val ", "val")

    targets = torch.eye(len(vocab), dtype=torch.float16)[[index[a] for a in fit_df["majority_answer"]]]
    questions = pd.concat([train_df["question"], val_df["question"]], ignore_index=True)
    data = Data(train=train_df, val=val_df, fit=fit_df, hold=hold_df, questions=questions, vocab=vocab, targets=targets)
    return Context(paths=paths, device="cpu", data=data, limit=None)


def test_run_complete_row_and_scores(tmp_path):
    """Verify that experiment.run produces complete metrics row and saves score array.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ctx = make_synthetic_context(tmp_path)
    scores_dir = tmp_path / "scores"
    row = run(ctx, "tiny", "random", "product", epochs=2, patience=2, scores_dir=scores_dir)
    expected_keys = {
        "text", "backbone", "fusion", "seed", "seed_used", "lr", "lr_used",
        "epochs", "holdout", "val", "yes_no", "number", "other", "secs",
        "restarted", "curve", "discarded_seed", "discarded_lr",
        "discarded_holdout", "discarded_curve",
    }
    assert set(row.keys()) == expected_keys
    assert row["fusion"] == "product" and row["backbone"] == "tiny" and row["text"] == "random"
    assert row["seed"] == row["seed_used"] == SEED and row["lr"] == row["lr_used"] == HEADS["product"]["lr"]
    assert row["epochs"] == 2 and row["restarted"] is False
    assert row["discarded_seed"] == row["discarded_lr"] == row["discarded_holdout"] == row["discarded_curve"] == ""
    assert isinstance(row["curve"], str) and len(row["curve"]) > 0
    for k in ("holdout", "val", "yes_no", "number", "other", "secs"):
        assert isinstance(row[k], (int, float)) and math.isfinite(row[k])

    score_file = scores_dir / "random_tiny_product_0.0008_123.npz"   # text_image_head_lr_seed: benchmark.confidence_table finds the files by this name
    assert score_file.is_file()
    with np.load(score_file) as z:   # compressed: five distinct values per question, about ten times smaller than a plain .npy
        scores = z["scores"]
    assert len(scores) == len(ctx.data.val) and scores.dtype == np.float32


@pytest.mark.parametrize(
    ("holdouts", "healthy", "exp_seeds", "exp_lrs", "exp_holdout", "exp_restarted",
     "exp_seed_used", "exp_lr_used", "exp_disc_seed", "exp_disc_lr", "exp_disc_holdout"),
    [
        ([1.0], 1, [SEED], [1e-3], 1.0, False, SEED, 1e-3, "", "", ""),
        ([1.0, 5.0], 2, [SEED, SEED + 101], [1e-3, 1e-3], 5.0, True, SEED + 101, 1e-3, str(SEED), "0.001", "1.00"),
        ([10.0, 5.0], 2, [SEED, SEED + 101], [1e-3, 1e-3], 10.0, True, SEED, 1e-3, str(SEED + 101), "0.001", "5.00"),
        ([1.0, 2.0, 8.0], 3, [SEED, SEED + 101, SEED], [1e-3, 1e-3, 5e-4], 8.0, True, SEED, 5e-4,
         f"{SEED}|{SEED + 101}", "0.001|0.001", "1.00|2.00"),
        ([1.0, 5.0, 3.0], 0, [SEED, SEED + 101, SEED], [1e-3, 1e-3, 5e-4], 5.0, True, SEED + 101, 1e-3,
         f"{SEED}|{SEED}", "0.001|0.0005", "1.00|3.00"),
        ([2.0, 1.0, 8.0], 3, [SEED, SEED + 101, SEED], [1e-3, 1e-3, 5e-4], 8.0, True, SEED, 5e-4,
         f"{SEED}|{SEED + 101}", "0.001|0.001", "2.00|1.00"),
    ],
    ids=["healthy first attempt", "second attempt wins", "earlier attempt kept when better", "third attempt wins",
         "no healthy start: best of three", "discarded in attempt order"],
)
def test_attempt_logic(
    tmp_path, monkeypatch, holdouts, healthy, exp_seeds, exp_lrs, exp_holdout,
    exp_restarted, exp_seed_used, exp_lr_used, exp_disc_seed, exp_disc_lr, exp_disc_holdout
):
    """`holdouts`: each attempt's final holdout score; `healthy`: the attempt (1-based) whose first 3 epochs score 30, above
    STALL, so no attempt follows it (0: none does)."""
    ctx = make_synthetic_context(tmp_path)
    seeds, lrs = [], []
    monkeypatch.setattr(ex, "set_seed", seeds.append)

    real_evaluate = ex.evaluate
    def fake_evaluate(preds, gt):
        """Mock evaluation returning score above stall threshold for healthy attempt.

        Args:
            preds: Model prediction array or tensor.
            gt: Ground truth evaluation DataFrame or structure.

        Returns:
            Any: Evaluation result object.
        """
        res = real_evaluate(preds, gt)
        if len(seeds) == healthy:
            return type(res)(overall=30.0, per_question=res.per_question,
                             per_answer_type=res.per_answer_type)
        return res
    monkeypatch.setattr(ex, "evaluate", fake_evaluate)

    h_iter = iter(holdouts)
    def fake_train(model, epoch_batches, holdout_score, **kw):
        """Mock train loop calling holdout_score and yielding scores from generator.

        Args:
            model: PyTorch model being trained.
            epoch_batches: Iterable of epoch batches.
            holdout_score: Callback evaluating the holdout score.
            **kw: Keyword arguments including learning rate.

        Returns:
            tuple[float, int]: Final holdout score and epoch count.
        """
        lrs.append(kw.get("lr"))
        for _ in range(3):
            holdout_score(model)
        return next(h_iter), 3
    monkeypatch.setattr(ex, "train", fake_train)

    row = ex.run(ctx, "tiny", "random", "product", lr=1e-3, epochs=3)
    assert seeds == exp_seeds and lrs == exp_lrs and row["holdout"] == exp_holdout and row["restarted"] is exp_restarted
    assert (row["seed_used"], row["lr_used"], row["discarded_seed"], row["discarded_lr"], row["discarded_holdout"]) == (
        exp_seed_used, exp_lr_used, exp_disc_seed, exp_disc_lr, exp_disc_holdout
    )


def test_forget_and_image_source_memoised(tmp_path):
    """Verify in-memory caching of targets, image features, and selective cache eviction.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ctx = make_synthetic_context(tmp_path)
    targets = _targets(ctx)
    assert torch.equal(targets.cpu(), ctx.data.targets.cpu())
    run(ctx, "tiny", "random", "product", epochs=1)
    assert ("tiny", "product") in ctx.images and _targets(ctx) is targets and ctx.targets is targets   # moved to the device once, reused by every run
    n1 = (len(ctx.images), len(ctx.questions))
    run(ctx, "tiny", "random", "product", epochs=1)
    assert (len(ctx.images), len(ctx.questions)) == n1
    pooled = ctx.paths.processed_dir / "features/image/pooled/tiny"
    assert pooled.exists()
    forget(ctx, "tiny", disk=False)
    assert pooled.exists() and ("tiny", "product") not in ctx.images
    run(ctx, "tiny", "random", "product", epochs=1)
    forget(ctx, "tiny")
    assert not pooled.exists() and ("tiny", "product") not in ctx.images


def test_question_arm_placement_and_eviction(tmp_path, monkeypatch):
    """Verify question representations are loaded on demand and evicted to preserve VRAM.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        monkeypatch: Monkeypatch fixture for substituting question_inputs.
    """
    ctx = make_synthetic_context(tmp_path)
    tags = []
    states = lambda: {
        "fit": Rows(np.zeros((4, 2, 3), np.float16)),
        "hold": Rows(np.zeros((2, 2, 3), np.float16)),
        "val": Rows(np.zeros((2, 2, 3), np.float16)),
        "kwargs": {"input_dim": 3},
    }
    def fake_question_inputs(text, *a, **k):
        """Mock question feature loading and record limit tagging.

        Args:
            text: Text representation key.
            *a: Positional arguments.
            **k: Keyword arguments including limit tag.

        Returns:
            dict[str, Any]: Mock question tensor dictionary.
        """
        tags.append(k.get("tag"))
        if text == "bert_large":
            assert "bert" not in ctx.questions
        return states()

    monkeypatch.setattr(ex, "question_inputs", fake_question_inputs)
    ctx.questions["qwen3_emb_0.6b"] = {"fit": Rows(torch.zeros(4, 16)), "kwargs": {"vector_dim": 16}}
    inp = _question_source(ctx, "bert", gpu_states=True)
    assert isinstance(inp["fit"].data, torch.Tensor) and isinstance(inp["hold"].data, torch.Tensor)
    assert "qwen3_emb_0.6b" not in ctx.questions   # one large question arm on the GPU at a time (the modern run ran out of memory)
    ctx.limit = 500
    _question_source(ctx, "bert_large", gpu_states=True)
    assert tags == ["", "_limit500"] and "bert" not in ctx.questions and "bert_large" in ctx.questions
    _question_source(ctx, "qwen3_emb_4b", gpu_states=True)
    assert "bert_large" not in ctx.questions and "qwen3_emb_4b" in ctx.questions
    loads = len(tags)
    # reused for the next head, not reloaded
    assert _question_source(ctx, "qwen3_emb_4b", gpu_states=True) is ctx.questions["qwen3_emb_4b"] and len(tags) == loads
    small = _question_source(ctx, "qwen3_emb_0.6b_tokens", gpu_states=False)   # token arms: placement from TOKEN_ARMS, not gpu_states
    assert isinstance(small["fit"].data, torch.Tensor) and isinstance(small["hold"].data, torch.Tensor)
    big = _question_source(ctx, "qwen3_emb_4b_tokens", gpu_states=True)
    assert isinstance(big["fit"].data, np.ndarray) and isinstance(big["hold"].data, np.ndarray)
    assert "qwen3_emb_0.6b_tokens" not in ctx.questions


def test_context_answers_and_resolve_name(paths, monkeypatch):
    """Verify context initialization parameters and encoder name resolution logic.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for substituting data and encoder loaders.
    """
    recorded = []
    monkeypatch.setattr(ex, "prepare_data", lambda p, limit=None, answers=1000: recorded.append((limit, answers)) or "data")
    c = context(limit=100, device="cpu", paths=paths, answers=3129)
    assert c.limit == 100 and c.data == "data" and recorded == [(100, 3129)]
    c_def = context(device="cpu", paths=paths)
    assert c_def.limit is None and recorded[-1] == (None, 1000)

    monkeypatch.setattr(ex, "load_encoders", lambda p, k: [{"key": f"{k}_k", "name": f"{k}_n"}])
    assert _resolve_name(c_def, "image_k") == "image_n"
    assert _resolve_name(c_def, "modern_image_k") == "modern_image_n"
    assert _resolve_name(c_def, "other") == "other"


def test_feature_folders(tmp_path):
    """The product head reads the pooled vectors a grid extraction left; a folder without ids.npy is not a finished cache."""
    ctx = make_synthetic_context(tmp_path)
    feats = ctx.paths.processed_dir / "features/image"
    (feats / "grid").mkdir(parents=True, exist_ok=True)
    (feats / "pooled/tiny").rename(feats / "grid/tiny")
    assert pooled_split(ctx.paths, "tiny", "train2014") is not None
    row = run(ctx, "tiny", "random", "product", epochs=1)
    assert row["fusion"] == "product" and row["backbone"] == "tiny"
    for sub, name, shape in (("pooled", "pooled.npy", (4, 8)), ("grid", "features.npy", (4, 4, 8))):
        d = feats / sub / "tiny/missing"
        d.mkdir(parents=True)
        np.save(d / name, np.zeros(shape, dtype=np.float16))
    assert pooled_split(ctx.paths, "tiny", "missing") is None and grid_split(ctx.paths, "tiny", "missing") is None


def test_image_rows_follow_the_ids_and_attention_runs_on_cpu(tmp_path):
    """Verify image features align with question image IDs and attention head executes on CPU.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ctx = make_synthetic_context(tmp_path)
    key = "rev"
    for split, ids in [
        ("train2014", np.arange(12, 0, -1, dtype=np.int64)),
        ("val2014", np.arange(105, 100, -1, dtype=np.int64)),
    ]:
        p_dir = ctx.paths.processed_dir / f"features/image/pooled/{key}/{split}"
        g_dir = ctx.paths.processed_dir / f"features/image/grid/{key}/{split}"
        p_dir.mkdir(parents=True, exist_ok=True)
        g_dir.mkdir(parents=True, exist_ok=True)
        np.save(p_dir / "ids.npy", ids)
        np.save(g_dir / "ids.npy", ids)
        pooled = np.repeat(ids[:, None].astype(np.float32), 8, axis=1).astype(np.float16)
        np.save(p_dir / "pooled.npy", pooled)
        grid = np.repeat(np.repeat(ids[:, None, None].astype(np.float32), 8, axis=2), 4, axis=1).astype(np.float16)
        np.save(g_dir / "features.npy", grid)

    src_tr, _, rows_fit, _, _, _ = _image_source(ctx, key, "product")
    fit_vecs = src_tr.take(rows_fit)
    img_ids = ctx.data.fit.image_id.to_numpy()
    unique_ids = torch.tensor(np.unique(img_ids), dtype=torch.float32)
    mu, sd = unique_ids.mean(), unique_ids.std(correction=0)
    expected = (torch.tensor(img_ids, dtype=torch.float32) - mu) / sd
    assert torch.allclose(fit_vecs[:, 0], expected, atol=1e-3)
    for i in range(len(img_ids) - 1):
        assert (fit_vecs[i, 0] < fit_vecs[i + 1, 0]).item() == (img_ids[i] < img_ids[i + 1])
        assert (fit_vecs[i, 0] == fit_vecs[i + 1, 0]).item() == (img_ids[i] == img_ids[i + 1])

    row = run(ctx, key, "random", "attention", epochs=1, patience=1)
    assert row["fusion"] == "attention" and row["backbone"] == key and row["epochs"] == 1
    assert isinstance(row["val"], float) and math.isfinite(row["val"])
    forget(ctx, key)
    assert not (ctx.paths.processed_dir / f"features/image/grid/{key}").exists()
    assert not (ctx.paths.processed_dir / f"features/image/pooled/{key}").exists()


def test_run_rejects_unknown_head_and_text(tmp_path):
    """Verify early validation rejects invalid fusion heads and text representations.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    ctx = make_synthetic_context(tmp_path)
    with pytest.raises(ValueError, match="product.*attention"):
        run(ctx, "tiny", "random", "atention")
    with pytest.raises(ValueError, match="glove"):
        run(ctx, "tiny", "glvoe", "product")
    assert not (ctx.paths.processed_dir / "features/image/grid").exists()   # nothing was extracted before the check
