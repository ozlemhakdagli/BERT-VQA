"""The merged results file: every finished benchmark in one Markdown file, `notebooks/benchmark_results.md`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from vqa_lab.benchmark import RUN_COLUMNS, collect_runs, trained_runs
from vqa_lab.config import Paths, atomic_path
from vqa_lab.vlm import vlm_tables

TITLES = {"benchmark": "Classical benchmark", "modern": "Modern encoder benchmark"}   # run-table folder -> section title
NUMBERS = ["val","yes_no","number","other"]
SHOWN = ["backbone","family","text","fusion","status","epochs","holdout",*NUMBERS,"secs","restarted"]
LEGEND = ("`val` is the official VQA accuracy % on the whole val2014 (the reported number); `holdout` is the accuracy on the image-disjoint holdout "
          "of train2014, which every funnel decision used. `epochs` are those of the reported attempt, `secs` the wall-clock time of the cell with "
          "every attempt, `retried` means the first attempt stalled and the better attempt is reported. Stopped, retried and failed runs are listed at the end.")


def pipe_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub pipe table.

    Args:
        df: Table to render; missing values become empty cells.

    Returns:
        str: The table, one line per row; a `|` inside a value is escaped, or it would split the cell.
    """
    cols = [str(c) for c in df.columns]
    lines = [f"| {' | '.join(cols)} |", f"| {' | '.join(['---'] * len(cols))} |"]
    for row in df.itertuples(index=False, name=None):
        lines.append(f"| {' | '.join('' if pd.isna(v) else str(v).replace('|', chr(92) + '|') for v in row)} |")
    return "\n".join(lines)


def _lr(v) -> str:   # the learning rate of an attempt, written only when it was changed (a retry at half the rate)
    return "" if pd.isna(v) or v == "" else f", lr {v:g}" if isinstance(v, float) else f", lr {v}"


def discarded_runs(runs: pd.DataFrame, failed: pd.DataFrame) -> pd.DataFrame:
    """List the runs to look at twice: stopped, retried, never learned and failed.

    Args:
        runs: Run table of one benchmark.
        failed: Its `failed.csv` (columns key, mode, error).

    Returns:
        pd.DataFrame: One row per run with backbone, text, fusion, reason and a one-line detail; a failed encoder has no text or fusion.
    """
    r = runs.reindex(columns=RUN_COLUMNS)
    never = r["status"] == "untrained"
    kinds = [
        (r["status"] == "stopped", "stopped", lambda x: f"holdout {x.holdout:.2f} < {x.stop_ref:.2f}, val {x.val:.2f}"),
        (r["restarted"].eq(True) & ~never, "retried", lambda x: f"kept seed {x.seed_used:.0f}{_lr(x.lr_used)} holdout {x.holdout:.2f}; "
                                                                               f"first tries seed {x.discarded_seed}{_lr(x.discarded_lr)} holdout {x.discarded_holdout}"),
        (never, "never learned", lambda x: f"best try {x.holdout:.2f}, other tries {x.discarded_holdout}"),
    ]
    rows = [(x.backbone, x.text, x.fusion, reason, detail(x)) for mask, reason, detail in kinds for x in r[mask].itertuples()]
    rows += [(f.key, None, None, "failed", (str(f.error).splitlines() or [""])[0][:100]) for f in failed.fillna("").itertuples()]
    return pd.DataFrame(rows, columns=["backbone", "text", "fusion", "reason", "detail"])


def write_report(paths: Paths) -> Path:
    """Merge the finished benchmarks into `notebooks/benchmark_results.md`, replacing the file.

    The file holds the headline, one run table per finished benchmark (classical, modern), the VLM runs and every
    discarded, retried or failed run. A benchmark that has not run is left out; one whose runs all failed still lists them.

    Args:
        paths: Project paths configuration.

    Returns:
        Path: The written file.

    Raises:
        ValueError: If nothing under `paths.runs_dir` has finished or failed.
    """
    vlm_runs, headline = vlm_tables(paths)
    sections, discarded = [], []
    for name, title in TITLES.items():
        runs = collect_runs(paths, name)
        failed_file = paths.runs_dir / name / "failed.csv"
        failed = pd.read_csv(failed_file) if failed_file.is_file() else pd.DataFrame(columns=["key", "mode", "error"])
        if runs.empty and failed.empty:
            continue
        if not runs.empty:   # a benchmark whose every cell failed has no run table, only failed.csv
            table = runs.reindex(columns=SHOWN)
            numeric = ["holdout", *NUMBERS, "secs"]
            table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
            table["epochs"] = pd.to_numeric(table["epochs"], errors="coerce").astype("Int64")
            table["restarted"] = table["restarted"].map({True: "yes"}).fillna("")
            sections.append(f"## {title}\n\n{len(runs)} runs, {len(trained_runs(runs))} trained.\n\n"
                            + pipe_table(table.sort_values("val", ascending=False).round(2).rename(columns={"restarted": "retried"})))
        discarded.append(discarded_runs(runs, failed).assign(benchmark=name)[["benchmark", "backbone", "text", "fusion", "reason", "detail"]])
    if not (sections or discarded) and vlm_runs.empty:
        raise ValueError(f"No finished runs under {paths.runs_dir}")
    parts = ["# Benchmark results", LEGEND]
    if not headline.empty:
        parts.append("## Headline\n\n" + pipe_table(headline))
    parts += sections
    if not vlm_runs.empty:
        parts.append("## VLM zero-shot baselines\n\n" + pipe_table(vlm_runs.round(2)))
    if discarded and not (rest := pd.concat(discarded, ignore_index=True)).empty:
        parts.append("## Discarded, retried and failed runs\n\n" + pipe_table(rest))
    target = paths.root / "notebooks" / "benchmark_results.md"   # the notebooks write it, and unlike artifacts/ it is published
    with atomic_path(target) as tmp:
        tmp.write_text("\n\n".join(parts) + "\n")
    return target
