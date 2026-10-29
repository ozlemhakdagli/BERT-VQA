"""VLM zero-shot baselines: question file preparation, answering commands, collection, scoring and the run tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vqa_lab.config import SEED, Paths, atomic_path
from vqa_lab.data import read_table
from vqa_lab.image import image_path
from vqa_lab.metrics import EvalResult, evaluate, normalize_answer
from vqa_lab.benchmark import collect_runs, trained_runs

VLM_MODELS: dict[str, dict] = {
    "gemma4_e4b": {
        "weights": "google/gemma-4-E4B-it",
        "name": "Gemma 4 E4B-it",
        "precision": "bf16",
        "mm_processor_kwargs": {"max_soft_tokens": 280},
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "qwen35_9b_fp8": {
        "weights": "RedHatAI/Qwen3.5-9B-FP8-dynamic",
        "name": "Qwen3.5-9B (RedHatAI FP8-dynamic)",
        "precision": "FP8 W8A8",
        "mm_processor_kwargs": {},
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


def question_file(val: pd.DataFrame, paths: Paths, limit: int | None = None) -> Path:
    """Write sorted question parquet file for VLM answering; return its path.

    Prepares questions with absolute image paths and sorts them by image ID and question ID.

    Args:
        val: Validation DataFrame containing questions.
        paths: Project paths configuration.
        limit: Optional maximum number of questions to sample.

    Returns:
        Path: Destination path of the generated question parquet file.
    """
    df = val.sample(n=limit, random_state=SEED) if limit is not None else val
    df = df.sort_values(["image_id", "question_id"]).reset_index(drop=True)
    val_dir = paths.raw_dir / "val2014"
    out_df = df[["question_id", "question"]].assign(image=[str(image_path(val_dir, "val2014", i)) for i in df["image_id"]])
    filename = "questions_full.parquet" if limit is None else f"questions_n{limit}.parquet"
    target = paths.processed_dir / "vlm" / filename
    with atomic_path(target) as tmp:
        out_df.to_parquet(tmp, index=False)
    return target


def out_dir(paths: Paths, key: str, questions: Path, prefix_caching: bool = True) -> Path:
    """Determine the output directory for a VLM model run.

    Args:
        paths: Project paths configuration.
        key: Model key from `VLM_MODELS`.
        questions: Path to the question parquet file.
        prefix_caching: Whether prefix caching was enabled for this run.

    Returns:
        Path: Output directory for the answer chunks, `timing.csv` and `environment.json`.
    """
    tag = Path(questions).stem.removeprefix("questions_")
    return paths.runs_dir / "vlm" / key / (tag if prefix_caching else f"{tag}_nocache")


def answer_command(
    paths: Paths,
    key: str,
    questions: Path,
    *,
    batch: int = 512,
    prefix_caching: bool = True,
) -> list[str]:
    """Construct command-line arguments to run `vlm_answer.py` in the vqa-lab conda environment.

    Runs `vlm_answer.py` as an isolated script with `-P` to prevent workspace shadowing.

    Args:
        paths: Project paths configuration.
        key: Model key identifier in `VLM_MODELS`.
        questions: Path to the question parquet file.
        batch: Inference batch size. Defaults to 512.
        prefix_caching: Whether to enable prompt prefix caching. Defaults to True.

    Returns:
        list[str]: Command line argument list suitable for subprocess execution.

    Raises:
        KeyError: If `key` is not found in `VLM_MODELS`.
    """
    info = VLM_MODELS[key]   # KeyError for an unknown model
    out = out_dir(paths, key, questions, prefix_caching=prefix_caching)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", "vqa-lab",
        "python", "-P", str(paths.root / "src/vqa_lab/vlm_answer.py"),
        "--model", str(paths.raw_dir / "models" / info["weights"]),
        "--questions", str(questions),
        "--out-dir", str(out),
        "--batch", str(batch),
        "--mm-processor-kwargs", json.dumps(info["mm_processor_kwargs"]),
        "--chat-template-kwargs", json.dumps(info["chat_template_kwargs"]),
    ]
    if not prefix_caching:
        cmd.append("--no-prefix-caching")
    return cmd


def collect_answers(questions: Path, out: Path) -> pd.DataFrame:
    """Collect and concatenate all chunk answer parquet files in question-file order.

    Args:
        questions: Path to the reference question parquet file.
        out: Directory containing output answer chunk parquet files.

    Returns:
        pd.DataFrame: Combined answers aligned with the questions.

    Raises:
        ValueError: If answer files are missing, contain duplicate question IDs,
            or don't match the expected set of question IDs.
    """
    q_df = pd.read_parquet(questions, columns=["question_id"])
    expected_qids = q_df["question_id"].tolist()

    files = sorted(Path(out).glob("answers_*.parquet"))   # the script's unfinished chunks end in .partial and never match
    if not files:
        raise ValueError(f"No answer files found in {out}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if dups := df.loc[df["question_id"].duplicated(), "question_id"].tolist():
        raise ValueError(f"Duplicate question_id in answers: {dups[:5]}")

    ans_set, exp_set = set(df["question_id"]), set(expected_qids)
    if missing := [qid for qid in expected_qids if qid not in ans_set]:
        raise ValueError(f"Missing {len(missing)} questions in answers: {missing[:5]}")
    if extra := [qid for qid in ans_set if qid not in exp_set]:
        raise ValueError(f"Extra {len(extra)} questions found in answers: {extra[:5]}")

    return df.set_index("question_id").loc[expected_qids].reset_index()


def score_answers(answers: pd.DataFrame, gt: pd.DataFrame) -> EvalResult:
    """Score collected VLM answers against ground-truth annotations after official normalization.

    Args:
        answers: DataFrame containing model predictions with question_id and answer columns.
        gt: Ground-truth DataFrame containing question_id and human answers.

    Returns:
        EvalResult: Evaluation result containing accuracy breakdown.
    """
    ans_str = answers["answer"].fillna("").astype(str)
    norm_cache = {a: normalize_answer(a) for a in ans_str.unique()}
    pred_map = dict(zip(answers["question_id"].tolist(), ans_str.map(norm_cache)))
    gt_sub = gt.set_index("question_id").loc[answers["question_id"]].reset_index()
    return evaluate(pred_map, gt_sub)


def vlm_tables(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every finished VLM run and return benchmark comparison tables.

    Scores every finished run against the val answers and returns both the VLM runs table
    and a comparison against top classifiers.

    Args:
        paths: Project paths configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Pair of (runs_df, comp_df) DataFrames.
    """
    val_gt = read_table(paths, "val")
    vlm_artifacts = paths.runs_dir / "vlm"

    run_rows: list[dict] = []

    for run_dir in sorted(p for p in vlm_artifacts.glob("*/*") if p.is_dir()):   # <model>/<question set>, model by model
        key, tag = run_dir.parent.name, run_dir.name
        info = VLM_MODELS.get(key, {"name": key, "precision": "unknown"})
        q_file = paths.processed_dir / "vlm" / f"questions_{tag.removesuffix('_nocache')}.parquet"
        try:
            answers = collect_answers(q_file, run_dir)
        except (ValueError, FileNotFoundError) as e:   # an unfinished run, or one whose question file is gone, is reported, not exported
            print(f"skipped {key}/{tag}: {e}")
            continue

        r = score_answers(answers, val_gt)

        empty_count = int((answers["answer"].isna() | (answers["answer"].str.strip() == "")).sum())
        empty_pct = 100.0 * empty_count / len(answers)

        timing_file = run_dir / "timing.csv"
        if timing_file.is_file():
            t_df = pd.read_csv(timing_file).drop_duplicates("chunk", keep="last")
            total_secs = float(t_df["seconds"].sum())
            total_q = float(t_df["questions"].sum())
            minutes = round(total_secs / 60.0, 2)
            q_per_s = round(total_q / total_secs, 2) if total_secs > 0 else 0.0
        else:
            minutes = q_per_s = np.nan
        prompt_tokens = round(float(answers["prompt_tokens"].mean()), 1)

        run_rows.append({
            "model": key,
            "name": info["name"],
            "precision": info["precision"],
            "set": tag,
            "questions": int(len(answers)),
            "overall": round(r.overall, 2),
            "yes_no": round(r.per_answer_type.get("yes/no", 0.0), 2),
            "number": round(r.per_answer_type.get("number", 0.0), 2),
            "other": round(r.per_answer_type.get("other", 0.0), 2),
            "empty": round(empty_pct, 2),
            "minutes": minutes,
            "questions_per_s": q_per_s,
            "prompt_tokens": prompt_tokens,
        })

    run_cols = [
        "model", "name", "precision", "set", "questions",
        "overall", "yes_no", "number", "other", "empty",
        "minutes", "questions_per_s", "prompt_tokens",
    ]
    runs_df = pd.DataFrame(run_rows, columns=run_cols)

    comp_rows: list[dict] = []
    for name, label in (("benchmark", "classical, 1,000-answer list"), ("modern", "modern, 3,129-answer list")):
        if not (valid := trained_runs(collect_runs(paths, name))).empty:
            top = valid.sort_values("val", ascending=False).iloc[0]
            comp_rows.append({"system": f"{top['text']} + {top['backbone']} ({top['fusion']})", "answers": label,
                              **{k: round(float(top[k]), 2) for k in ("val", "yes_no", "number", "other")}})

    for row in run_rows:
        if row["set"] == "full":
            comp_rows.append({"system": row["name"], "answers": "free text", "val": round(float(row["overall"]), 2),
                              **{k: round(float(row[k]), 2) for k in ("yes_no", "number", "other")}})

    comp_df = pd.DataFrame(comp_rows, columns=["system", "answers", "val", "yes_no", "number", "other"])
    return runs_df, comp_df
