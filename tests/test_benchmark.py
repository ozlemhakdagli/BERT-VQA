"""Tests for the benchmark funnel logic, RunTable and the report tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vqa_lab.benchmark import (
    RUN_COLUMNS, TOKEN_COLUMNS, RunTable, attention_plan, collect_runs, confidence_table, mark_screen,
    screen_board, screen_scores, select_survivors, token_table, write_runs,
)
from vqa_lab.config import Paths
from vqa_lab.experiment import STALL


def _screen(extra: list[dict] | None = None, **holdouts) -> pd.DataFrame:
    """Product-head screen rows: one glove and one bert row per keyword (encoder = holdout, or (holdout, status))."""
    rows = [
        {"backbone": k, "text": t, "fusion": "product", "holdout": s, "status": st}
        for k, v in holdouts.items()
        for s, st in [(v if isinstance(v, tuple) else (v, "done"))]
        for t in ("glove", "bert")
    ]
    return pd.DataFrame(rows + (extra or []))


def test_runtable_finished(paths: Paths) -> None:
    """Verify RunTable.finished correctly identifies completed, failed, or stalled runs.

    Args:
        paths: Project paths fixture.
    """
    table = RunTable(paths, paths.artifacts_dir / "runs/benchmark")
    table.results = pd.DataFrame([
        {"backbone": "enc1", "text": "glove", "fusion": "product", "holdout": STALL - 1.0, "status": "untrained", "discarded_seed": "123"},
        {"backbone": "enc2", "text": "glove", "fusion": "product", "holdout": STALL + 5.0, "status": "done", "discarded_seed": ""},
        {"backbone": "enc4", "text": "glove", "fusion": "product", "holdout": STALL - 1.0, "status": "untrained", "discarded_seed": "123|224"},
        {"backbone": "enc5", "text": "glove", "fusion": "product", "holdout": STALL - 1.0, "status": "untrained", "discarded_seed": "", "curve": "24.0;24.2"},
    ]).reindex(columns=RUN_COLUMNS)
    assert [table.finished(f"enc{i}", "glove", "product") for i in range(1, 6)] == [False, True, False, True, True]


def test_survivors_selection_exclusions_and_marks() -> None:
    """Verify survivor selection cutoff, exclusion of untrained runs, and status marking."""
    extra = [   # one screen embedding never learned: ranked by the trained one (the documented rule)
        {"backbone": "enc_partial", "text": "glove", "fusion": "product", "holdout": 50.0, "status": "done"},
        {"backbone": "enc_partial", "text": "bert", "fusion": "product", "holdout": 21.0, "status": "untrained"},
        {"backbone": "enc_unfinished", "text": "glove", "fusion": "product", "holdout": 58.0, "status": "done"},   # screen still running
        {"backbone": "e0", "text": "fasttext", "fusion": "product", "holdout": 50.0, "status": "done"},
        {"backbone": "e0", "text": "word2vec", "fusion": "product", "holdout": 50.0, "status": "done"},
    ]
    df = _screen(extra, **{f"e{i}": 50.0 + i for i in range(10)}, enc_untrained=(20.0, "untrained"))
    scores, untrained = screen_scores(df, ["glove", "bert"])
    assert untrained == ["enc_untrained"] and scores == {**{f"e{i}": 50.0 + i for i in range(10)}, "enc_partial": 50.0}

    survivors, stopped, kept, cut = select_survivors(df, ["glove", "bert"], ["fasttext", "word2vec"], 0.34, [])
    assert cut == 56.0 and survivors == ["e9", "e8", "e7", "e6"]
    assert set(stopped) == {"e1", "e2", "e3", "e4", "e5", "enc_partial"}
    assert "e0" not in stopped and "enc_untrained" not in survivors + stopped + kept

    df = _screen(e1=85.0, e2=75.0, e_always=45.0)   # an always-kept encoder survives below the cut
    survivors, stopped, kept, cut = select_survivors(df, ["glove", "bert"], ["fasttext"], 0.33, always=["e_always"])
    assert cut == 85.0 and survivors == ["e1", "e_always"] and stopped == ["e2"] and kept == ["e_always"]
    assert select_survivors(df, ["glove", "bert"], ["fasttext"], 0.0, always=["e_always"])[:3] == (["e_always"], ["e1", "e2"], ["e_always"])
    assert select_survivors(df, ["glove", "bert"], ["fasttext"], 1.5, always=[])[0] == ["e1", "e2", "e_always"]
    marked = mark_screen(df, ["glove", "bert"], stopped, kept, cut)
    for key, status, ref in (("e_always", "kept", cut), ("e2", "stopped", cut), ("e1", "done", None)):
        rows = marked[marked["backbone"] == key]
        assert (rows["status"] == status).all() and (rows["stop_ref"].isna().all() if ref is None else (rows["stop_ref"] == ref).all())


def test_attention_plan_rules() -> None:
    """Verify attention plan construction follows top performers, always-kept, and wildcards."""
    df = pd.DataFrame([
        ("top", "bert", "product", 60.0, "done"), ("top", "glove", "product", 58.0, "done"),
        ("top", "random", "product", 99.0, "done"), ("stop_a", "glove", "product", 45.0, "stopped"),
        ("stop_b", "bert", "product", 47.0, "stopped"), ("always", "glove", "product", 44.0, "kept"),
    ], columns=["backbone", "text", "fusion", "holdout", "status"])
    plan = attention_plan(df, screen=["glove", "bert"], top_n=1, always=["always"], berts=["bert"], wildcards=1)
    assert ("top", "random") not in plan and plan == [("top", "bert"), ("always", "glove"), ("always", "bert"), ("stop_b", "bert")]


def test_runtable_lifecycle(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RunTable execution caching, failure logging, and independent benchmark tables.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for substituting the experiment runner.
    """
    paths.configs_dir.mkdir(parents=True, exist_ok=True)
    (paths.configs_dir / "vqa_lab.yaml").write_text("classical:\n  image:\n    - {key: res50, family: ResNet}\n")
    calls = []

    def fake_run(ctx, image, text, head="product", **kw):
        """Mock experiment.run returning predefined metrics for the benchmark run.

        Args:
            ctx: Experiment context dictionary or None.
            image: Image backbone identifier.
            text: Text representation identifier.
            head: Fusion head architecture.
            **kw: Additional keyword arguments.

        Returns:
            dict[str, Any]: Simulated experiment result dictionary.
        """
        calls.append((image, text, head))
        return {"backbone": image, "text": text, "fusion": head, "seed": 123, "holdout": 55.0, "val": 56.0, "secs": 12.0}

    monkeypatch.setattr("vqa_lab.experiment.run", fake_run)

    smoke = RunTable(paths, paths.artifacts_dir / "runs/benchmark_smoke")
    smoke.run(None, "res50", "glove", "product")
    assert (smoke.run_dir / "results.parquet").is_file()
    calls.clear()

    run_dir = paths.artifacts_dir / "runs/benchmark"
    table = RunTable(paths, run_dir)
    assert (run_dir / "scores").is_dir()

    r1 = table.run(None, "res50", "glove", "product")
    assert len(calls) == 1 and r1["family"] == "ResNet" and r1["status"] == "done"
    assert (run_dir / "results.parquet").is_file()   # the record is the parquet

    r2 = table.run(None, "res50", "glove", "product")
    assert len(calls) == 1 and r2["backbone"] == "res50" and r2["holdout"] == 55.0


    table.fail("res50", "funnel", RuntimeError("disk full"))
    failed_csv = run_dir / "failed.csv"
    assert failed_csv.is_file()
    df_fail = pd.read_csv(failed_csv)
    assert len(df_fail) == 1 and df_fail.iloc[0]["key"] == "res50" and "disk full" in df_fail.iloc[0]["error"]

    table.fail("res101", "funnel", RuntimeError("oom"))
    assert len(pd.read_csv(failed_csv)) == 2

    (paths.configs_dir / "vqa_lab.yaml").write_text("classical:\n  image:\n    - {key: res50, family: ResNet}\n"
                                                    "modern:\n  image:\n    - {key: siglip, family: SigLIP}\n  text:\n    - {key: qwen, kind: sentence}\n")
    modern = RunTable(paths, paths.artifacts_dir / "runs/modern")
    assert table.name == "benchmark" and modern.name == "modern" and smoke.name is None and modern.families.get("siglip") == "SigLIP"
    modern.run(None, "siglip", "qwen", "product")
    assert modern.results["backbone"].tolist() == ["siglip"] and "siglip" not in table.results["backbone"].tolist()   # each benchmark keeps its own table


def test_write_runs(tmp_path: Path) -> None:
    """Verify atomic parquet serialization of benchmark runs with serialized string columns.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    old = pd.DataFrame([
        {"backbone": "a", "text": "glove", "fusion": "product", "val": 50.0, "discarded_seed": 224.0, "discarded_holdout": 24.4, "curve": "1;2"},
        {"backbone": "b", "text": "glove", "fusion": "product", "val": 51.0, "discarded_seed": np.nan, "discarded_holdout": np.nan, "curve": "3;4"},
    ])
    new = pd.DataFrame([{"backbone": "c", "text": "glove", "fusion": "product", "val": 52.0, "discarded_seed": "224|123", "discarded_holdout": "24.40|24.38", "curve": "5;6"}])
    write_runs(pd.concat([old, new], ignore_index=True), tmp_path / "r.parquet")
    back = pd.read_parquet(tmp_path / "r.parquet")
    assert back["discarded_seed"].tolist() == ["224", "", "224|123"]
    assert back["discarded_holdout"].tolist() == ["24.4", "", "24.40|24.38"] and back["val"].tolist() == [50.0, 51.0, 52.0]
    assert not list(tmp_path.glob(".partial-*"))   # no temporary file left behind


def test_collect_runs(paths: Paths) -> None:
    """Verify collection of the benchmark run table while ignoring temporary partial files.

    Args:
        paths: Project paths fixture.
    """
    bench = paths.artifacts_dir / "runs" / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"backbone": "res152", "text": "glove", "fusion": "product", "val": 52.0, "secs": 120, "restarted": False},
        {"backbone": "vgg16", "text": "glove", "fusion": "product", "val": 49.0},
        {"backbone": "enc_stop", "text": "glove", "fusion": "product", "val": 30.0, "status": "stopped", "stop_ref": 48.0},
        {"backbone": "res152", "text": "bert", "fusion": "attention", "val": 55.0},
    ]).to_parquet(bench / "results.parquet")
    pd.DataFrame([{"backbone": "ignored", "text": "bert", "fusion": "attention", "val": 99.0}]).to_parquet(bench / ".partial-results.parquet")   # what an interrupted atomic write leaves

    runs = collect_runs(paths)
    assert list(runs.columns) == RUN_COLUMNS and len(runs) == 4 and "ignored" not in runs["backbone"].values
    assert runs.loc[runs["backbone"] == "res152", "secs"].iloc[1] == 120
    assert runs.loc[runs["backbone"] == "vgg16", "val"].iloc[0] == pytest.approx(49.0)
    stop = runs.loc[runs["backbone"] == "enc_stop"].iloc[0]
    assert stop["status"] == "stopped" and stop["stop_ref"] == 48.0 and runs["backbone"].tolist() == ["enc_stop", "res152", "res152", "vgg16"]


    mod_bench = paths.artifacts_dir / "runs" / "modern"
    mod_bench.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"backbone": "siglip", "text": "qwen", "fusion": "product", "val": 61.0}]).to_parquet(mod_bench / "results.parquet")
    assert collect_runs(paths, name="modern")["backbone"].tolist() == ["siglip"]


def test_confidence_table(tmp_path: Path) -> None:
    """Verify bootstrap confidence interval computation and paired delta intervals.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    image_ids = np.repeat(np.arange(10), 4)

    for name, ones in [
        ("bert_res152_attention_0.001_123.npz", 30), ("glove_res152_product_0.001_123.npz", 26),
        ("fasttext_res152_product_0.001_123.npz", 22), ("bert_convnext_product_0.001_123.npz", 36),
    ]:
        np.savez_compressed(scores_dir / name, scores=np.array([1.0] * ones + [0.0] * (40 - ones), dtype=np.float32))

    runs = pd.DataFrame([
        {"backbone": b, "text": t, "fusion": f, "lr": 0.001, "seed": 123, "val": v, "status": s} for b, t, f, v, s in [
            ("res152", "word2vec", "product", 80.0, "done"), ("convnext", "bert", "product", 90.0, "untrained"),
            ("res152", "glove", "product", 65.0, "done"), ("res152", "bert", "attention", 75.0, "done"),
            ("res152", "fasttext", "product", 55.0, "done"),
        ]
    ])

    table = confidence_table(runs, scores_dir, image_ids, top=5, n_boot=200, seed=42)
    assert len(table) == 3 and table["text"].tolist() == ["bert", "glove", "fasttext"] and table["val"].tolist() == [75.0, 65.0, 55.0]
    best_row = table.iloc[0]
    assert best_row["diff"] == 0 and best_row["diff_lo"] == 0 and best_row["diff_hi"] == 0
    for _, row in table.iterrows():
        assert row["lo"] <= row["val"] <= row["hi"]
    assert table.iloc[1]["diff"] < 0 and table.iloc[2]["diff"] < 0
    assert "word2vec" not in table["text"].values and "convnext" not in table["backbone"].values
    empty = confidence_table(runs, tmp_path / "empty_dir", image_ids)
    assert len(empty) == 0 and list(empty.columns) == ["text", "backbone", "fusion", "val", "lo", "hi", "diff", "diff_lo", "diff_hi"]


def test_token_table() -> None:
    """Verify token vs vector comparison table generation and delta column calculations."""
    runs = pd.DataFrame([{"backbone": b, "text": t, "fusion": f, "val": v, "status": s} for b, t, f, v, s in [
        ("enc", "bert", "attention", 64.0, "done"), ("enc", "qwen3_emb_0.6b", "attention", 63.0, "done"),
        ("enc", "qwen3_emb_0.6b_tokens", "attention", 64.5, "done"), ("enc", "qwen3_emb_4b", "attention", 65.0, "done"),
        ("enc", "qwen3_emb_4b_tokens", "attention", 66.25, "done"),
        ("enc", "bert", "product", 58.0, "done"), ("enc", "qwen3_emb_4b_tokens", "product", 57.0, "done"),
        ("enc", "qwen3_emb_0.6b_tokens", "product", 24.0, "untrained"),
        ("other", "bert", "attention", 61.0, "done"),   # no token run: no row
    ]]).reindex(columns=RUN_COLUMNS)
    t = token_table(runs)
    assert list(t.columns) == ["backbone", "fusion", *TOKEN_COLUMNS, "d_0.6b_tokens_vs_vector", "d_4b_tokens_vs_vector", "d_4b_tokens_vs_bert"]
    assert t[["backbone", "fusion"]].values.tolist() == [["enc", "attention"], ["enc", "product"]]
    a, p = t.iloc[0], t.iloc[1]
    assert (a["d_0.6b_tokens_vs_vector"], a["d_4b_tokens_vs_vector"], a["d_4b_tokens_vs_bert"]) == (1.5, 1.25, 2.25)
    assert pd.isna(p["qwen3_emb_0.6b_tokens"]) and pd.isna(p["qwen3_emb_4b"]) and pd.isna(p["d_4b_tokens_vs_vector"]) and p["d_4b_tokens_vs_bert"] == -1.0
    assert token_table(runs[~runs["text"].str.endswith("_tokens")]).empty


def test_visit_logs_failure_and_always_forgets(paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RunTable.visit logs errors to CSV and guarantees cache eviction on exit.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for substituting run and forget functions.
    """
    calls, forgotten = [], []

    def fake_run(ctx, image, text, head="product", **kw):
        """Mock experiment runner that raises an error on a specific text representation.

        Args:
            ctx: Experiment context dictionary or None.
            image: Image backbone identifier.
            text: Text representation identifier.
            head: Fusion head architecture.
            **kw: Additional keyword arguments.

        Returns:
            dict[str, Any]: Simulated experiment result dictionary.

        Raises:
            RuntimeError: If text is 'boom'.
        """
        calls.append((image, text, kw.get("epochs")))
        if text == "boom":
            raise RuntimeError("out of memory")
        return {"backbone": image, "text": text, "fusion": head, "seed": 123, "holdout": 55.0, "val": 56.0, "secs": 1.0}

    monkeypatch.setattr("vqa_lab.experiment.run", fake_run)
    monkeypatch.setattr("vqa_lab.experiment.forget", lambda ctx, key: forgotten.append(key))
    table = RunTable(paths, paths.artifacts_dir / "runs/benchmark_smoke")
    table.visit(None, "res50", [("glove", "product"), ("bert", "product")], "funnel", epochs=2)
    assert calls == [("res50", "glove", 2), ("res50", "bert", 2)]
    assert forgotten == ["res50"]
    table.visit(None, "res18", [("boom", "product"), ("glove", "product")], "funnel")   # a failure ends this encoder's visit ...
    failed = pd.read_csv(table.run_dir / "failed.csv")
    assert failed[["key", "mode"]].values.tolist() == [["res18", "funnel"]] and "out of memory" in failed["error"].iloc[0]
    assert forgotten == ["res50", "res18"] and calls[-1][:2] == ("res18", "boom")   # ... and still frees its features
    table.visit(None, "res50", [("glove", "product")], "funnel")   # a finished cell is reused, not retrained
    assert calls[-1][:2] == ("res18", "boom") and forgotten[-1] == "res50"


def test_screen_board_pivots_labels_and_sorts() -> None:
    """Verify screen board pivots text representation columns, joins families, and ranks."""
    res = pd.DataFrame([
        ("e_done", "glove", "product", 55.0, 56.0, "done"),
        ("e_done", "bert", "product", 57.0, 58.0, "done"),
        ("e_done", "bert", "attention", 59.0, 60.0, "done"),
        ("e_stop", "glove", "product", 24.0, 24.1, "untrained"),
        ("e_stop", "bert", "product", 52.0, 53.0, "stopped"),
        ("e_kept", "glove", "product", 50.0, 51.0, "kept"),
        ("e_kept", "bert", "product", 51.0, 52.0, "kept"),
        ("e_untr", "glove", "product", 24.0, 24.0, "untrained"),
        ("e_untr", "bert", "product", 24.2, 24.3, "untrained"),
    ], columns=["backbone", "text", "fusion", "holdout", "val", "status"])
    screen = ["glove", "bert"]
    families = {"e_done": "CLIP", "e_stop": "ResNet", "e_kept": "Xception", "e_untr": "VGG"}
    nan = float("nan")
    expected = pd.DataFrame(
        [("CLIP", 56.0, 55.0, 58.0, 57.0, 58.0, "done"),   # the attention row is not part of the screen
         ("ResNet", nan, nan, 53.0, 52.0, 53.0, "stopped"),   # an untrained cell shows no numbers
         ("Xception", 51.0, 50.0, 52.0, 51.0, 52.0, "kept"),
         ("VGG", nan, nan, nan, nan, nan, "untrained")],
        index=pd.Index(["e_done", "e_stop", "e_kept", "e_untr"], name="backbone"),
        columns=["family", "glove_val", "glove_holdout", "bert_val", "bert_holdout", "better_val", "status"],
    )
    pd.testing.assert_frame_equal(screen_board(res, screen, families), expected)

