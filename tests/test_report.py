import numpy as np
import pandas as pd
import pytest

from vqa_lab.benchmark import RUN_COLUMNS, write_runs
from vqa_lab.report import discarded_runs, pipe_table, write_report
from vqa_lab.vlm import out_dir, question_file


def _runs() -> pd.DataFrame:   # one run of every kind: done, stopped, retried, never learned
    rows = [
        ("res50", "bert", "product", "done", 61.234, 80.0, 40.0, 50.0, 60.0, False, "", np.nan),
        ("res50", "glove", "product", "done", 55.0, 75.0, 30.0, 45.0, 54.0, False, "", np.nan),
        ("vgg16", "glove", "product", "stopped", 49.0, 70.0, 25.0, 40.0, 49.5, False, "", 53.9),
        ("res152", "bert", "product", "done", 58.0, 78.0, 33.0, 47.0, 57.0, True, "24.5", np.nan),
        ("dense", "glove", "product", "untrained", 24.9, 60.0, 1.0, 2.0, 24.8, True, "24.6|24.7", np.nan),
    ]
    cols = ["backbone", "text", "fusion", "status", "val", "yes_no", "number", "other", "holdout", "restarted", "discarded_holdout", "stop_ref"]
    df = pd.DataFrame(rows, columns=cols).reindex(columns=RUN_COLUMNS)
    df["family"], df["epochs"], df["secs"] = "ResNet", 12, 345.6
    df["seed_used"], df["discarded_seed"], df["discarded_lr"] = [123, 123, 123, 224, 123], ["", "", "", "123", ""], ""   # res152 was retried with another seed
    return df


def _failed() -> pd.DataFrame:
    return pd.DataFrame({"key": ["bad|enc", "plain"], "mode": ["funnel", "run"], "error": ["CUDA out of memory.\nTried to allocate 8 GiB", ""]})


def _benchmark(paths, name: str = "benchmark") -> None:   # what RunTable.save and RunTable.fail leave behind
    write_runs(_runs(), paths.runs_dir / name / "results.parquet")
    _failed().to_csv(paths.runs_dir / name / "failed.csv", index=False)


def _val_table(paths) -> pd.DataFrame:
    val = pd.DataFrame({"question_id": [21, 12, 11], "image_id": [2, 1, 1], "question": ["What color?", "How many?", "Is it?"],
                        "question_type": ["what color", "how many", "is"], "answer_type": ["other", "number", "yes/no"],
                        "answers_raw": [np.array(["red"] * 3 + ["blue"] * 7, dtype=object), np.array(["2"] * 10, dtype=object),
                                        np.array(["yes"] * 10, dtype=object)]})
    paths.tables_dir.mkdir(parents=True)
    val.to_parquet(paths.tables_dir / "val.parquet")
    return val


def test_pipe_table_escapes_pipes_and_blanks_missing():
    table = pipe_table(pd.DataFrame({"a": ["x|y", None], "b": [1.5, np.nan]}))
    assert table.splitlines() == ["| a | b |", "| --- | --- |", "| x\\|y | 1.5 |", "|  |  |"]


def test_discarded_runs_names_every_kind():
    d = discarded_runs(_runs(), _failed())
    assert d["reason"].tolist() == ["stopped", "retried", "never learned", "failed", "failed"]
    stopped, retried, never, failed, empty = d.to_dict("records")
    assert (stopped["backbone"], stopped["text"], stopped["detail"]) == ("vgg16", "glove", "holdout 49.50 < 53.90, val 49.00")
    assert (retried["backbone"], retried["detail"]) == ("res152", "kept seed 224 holdout 57.00; first tries seed 123 holdout 24.5")   # a retried run that learned
    assert (never["backbone"], never["detail"]) == ("dense", "best try 24.80, other tries 24.6|24.7")   # the best attempt, not the last
    assert failed["backbone"] == "bad|enc" and failed["detail"] == "CUDA out of memory."   # first line of the error only
    assert empty["detail"] == "" and pd.isna(empty["text"])
    assert discarded_runs(_runs().iloc[:2], _failed().iloc[:0]).empty   # nothing to report


def test_write_report_is_rebuilt_from_the_files_it_finds(paths, capsys):
    val = _val_table(paths)
    _benchmark(paths)
    target = write_report(paths)
    assert target == paths.root / "notebooks" / "benchmark_results.md"   # written by the notebooks, so it lives next to them
    assert sorted(p.name for p in paths.artifacts_dir.iterdir()) == ["runs"]
    assert [p.name for p in target.parent.iterdir()] == ["benchmark_results.md"]   # one new file, nothing else
    text = target.read_text()
    assert [l for l in text.splitlines() if l.startswith("#")] == [
        "# Benchmark results", "## Headline", "## Classical benchmark", "## Discarded, retried and failed runs"]   # modern and VLM never ran
    assert "5 runs, 4 trained" in text
    assert "| res50 | ResNet | bert | product | done | 12 | 60.0 | 61.23 | 80.0 | 40.0 | 50.0 | 345.6 |  |" in text   # best run first, with epochs, holdout and time
    assert "| benchmark | vgg16 | glove | product | stopped | holdout 49.50 < 53.90, val 49.00 |" in text
    assert "bad\\|enc" in text and "Tried to allocate" not in text

    _benchmark(paths, "modern")   # a benchmark finishes: the next call grows the one file instead of appending to it
    questions = question_file(val, paths)
    out = out_dir(paths, "gemma4_e4b", questions)
    out.mkdir(parents=True)
    pd.DataFrame({"question_id": [11, 12, 21], "answer": ["yes", "2", "red"], "raw_output": ["yes", "2", "red"],
                  "prompt_tokens": [300] * 3}).to_parquet(out / "answers_000.parquet")
    text = write_report(paths).read_text()
    assert [l for l in text.splitlines() if l.startswith("## ")] == [
        "## Headline", "## Classical benchmark", "## Modern encoder benchmark", "## VLM zero-shot baselines",
        "## Discarded, retried and failed runs"]
    assert "| Gemma 4 E4B-it | free text | 96.67 |" in text   # the VLM joins the headline next to the best classical and modern runs
    assert "| modern | vgg16 | glove | product | stopped |" in text and "| benchmark | vgg16 | glove | product | stopped |" in text
    assert [p.name for p in target.parent.iterdir()] == ["benchmark_results.md"]

    questions.unlink()   # a cleaned data folder: the answers survive, their question list does not
    assert "## VLM zero-shot baselines" not in write_report(paths).read_text()
    assert "skipped gemma4_e4b/full" in capsys.readouterr().out   # said aloud, not left out of the report in silence


def test_a_benchmark_that_only_failed_still_gets_a_report(paths):
    _val_table(paths)
    (paths.runs_dir / "modern").mkdir(parents=True)
    _failed().to_csv(paths.runs_dir / "modern" / "failed.csv", index=False)   # every cell failed: failed.csv, no results.parquet
    text = write_report(paths).read_text()
    assert [l for l in text.splitlines() if l.startswith("#")] == ["# Benchmark results", "## Discarded, retried and failed runs"]
    assert "| modern | bad\\|enc |  |  | failed | CUDA out of memory. |" in text


def test_write_report_without_runs_raises(paths):
    _val_table(paths)
    with pytest.raises(ValueError, match="No finished runs"):
        write_report(paths)
    assert not paths.artifacts_dir.exists() and not (paths.root / "notebooks").exists()
