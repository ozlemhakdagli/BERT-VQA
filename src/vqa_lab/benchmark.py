"""The benchmark funnel: the resumable run table (RunTable), screening, survivor selection, attention planning, report tables."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from vqa_lab import experiment
from vqa_lab.config import Paths, atomic_path, load_encoders
from vqa_lab.experiment import ATTEMPTS, STALL
from vqa_lab.metrics import cluster_bootstrap_diff, cluster_bootstrap_mean

RUN_COLUMNS = [
    "text", "backbone", "fusion", "family", "status", "seed", "seed_used", "lr", "lr_used",
    "epochs", "holdout", "val", "yes_no", "number", "other", "secs", "restarted",
    "discarded_seed", "discarded_lr", "discarded_holdout", "stop_ref", "curve", "discarded_curve",
]
# columns without any run are left out
CLASSICAL_TEXTS = ["glove", "fasttext", "word2vec", "bert", "bert_large"]   # the classical question embeddings, in board order
TOKEN_COLUMNS = ["bert", "qwen3_emb_0.6b", "qwen3_emb_0.6b_tokens", "qwen3_emb_4b", "qwen3_emb_4b_tokens"]
BENCHMARKS = {"benchmark": "", "modern": "modern_"}   # run-table folder -> prefix of its encoder registry in the config
ATTEMPT_COLUMNS = ["curve", "discarded_seed", "discarded_lr", "discarded_holdout", "discarded_curve"]


# =============================================================================
# Run table and funnel (uses experiment)
# =============================================================================

def _text(v) -> str:
    """Convert a value to string, mapping NaN and None to empty string.

    Args:
        v: Input value of any type.

    Returns:
        str: String representation, or empty string if null.
    """
    return str(v) if pd.notna(v) else ""


def _attempts(discarded_seed) -> int:
    """Count total attempts behind a run row (1 plus number of discarded attempts).

    Args:
        discarded_seed: Pipe-separated string of discarded seeds (e.g. "123|224") or NaN.

    Returns:
        int: Total number of attempts.
    """
    return 1 + (len(s.split("|")) if (s := _text(discarded_seed)) else 0)


def _curve_epochs(curve) -> int:
    """Count number of epochs recorded in a learning curve string.

    Args:
        curve: Semicolon-separated string of epoch scores (e.g. "51.2;53.0;...") or NaN.

    Returns:
        int: Number of epochs recorded, or 0 if missing.
    """
    return len(s.split(";")) if (s := _text(curve)) else 0


class RunTable:
    """The benchmark's resumable run table `run_dir/results.parquet`, one row per (encoder, embedding, head), saved
    after every change, so nothing is lost when a run stops."""

    def __init__(self, paths: Paths, run_dir: Path):
        """Initialize resumable benchmark run table.

        Args:
            paths: Project paths configuration.
            run_dir: Directory where run table `results.parquet`, `scores/`, and logs reside.
        """
        self.run_dir = Path(run_dir)
        resolved, name = self.run_dir.resolve(), self.run_dir.name
        # "benchmark" or "modern" for the two real tables; None for smoke tables
        self.name = name if name in BENCHMARKS and resolved == (paths.runs_dir / name).resolve() else None
        (self.run_dir / "scores").mkdir(parents=True, exist_ok=True)
        file = self.run_dir / "results.parquet"
        # empty on a fresh clone (artifacts/ is not in git): run the funnel
        self.results = pd.read_parquet(file) if file.is_file() else pd.DataFrame(columns=RUN_COLUMNS)
        # each benchmark labels its runs from its own registry (a smoke table, e.g. modern_smoke, from its benchmark's)
        registry = BENCHMARKS[self.name] if self.name else ("modern_" if self.run_dir.name.startswith("modern") else "")
        self.families = {e["key"]: e.get("family", "") for e in load_encoders(paths, registry + "image")}

    def _cell(self, key: str, text: str, head: str) -> pd.Series:
        """Return boolean mask indexing the row matching (backbone, text, fusion).

        Args:
            key: Image encoder backbone key.
            text: Question encoder arm name.
            head: Fusion head architecture.

        Returns:
            pd.Series: Boolean series matching the cell in `self.results`.
        """
        r = self.results
        return (r["backbone"] == key) & (r["text"] == text) & (r["fusion"] == head)

    def finished(self, key: str, text: str, head: str) -> bool:
        """Check if a benchmark cell already has a finished result to keep.

        A cell is finished if it learned (holdout >= STALL), it stalled in every rescue attempt,
        or it completed in fewer than 3 epochs.

        Args:
            key: Image encoder key.
            text: Question encoder arm.
            head: Fusion head.

        Returns:
            bool: True if cell has completed and should not be re-run.
        """
        rows = self.results[self._cell(key, text, head)]
        return any(h >= STALL or _attempts(d) >= len(ATTEMPTS) or 0 < _curve_epochs(c) < 3
                   for h, d, c in zip(rows["holdout"], rows["discarded_seed"], rows["curve"]))

    def run(self, ctx: experiment.Context, key: str, text: str, head: str, **run_kw) -> dict:
        """Return existing finished row for a cell, or train and record it now.

        Args:
            ctx: Experiment context.
            key: Image encoder key.
            text: Question encoder arm.
            head: Fusion head.
            **run_kw: Additional keyword arguments passed to `experiment.run`.

        Returns:
            dict: Completed run row record dictionary.
        """
        if self.finished(key, text, head):
            return self.results[self._cell(key, text, head)].iloc[-1].to_dict()
        row = experiment.run(ctx, key, text, head, scores_dir=self.run_dir / "scores", **run_kw)
        row["family"] = self.families.get(key, "")
        row["status"] = "untrained" if row["holdout"] < STALL else "done"
        frames = [f for f in (self.results[~self._cell(key, text, head)], pd.DataFrame([row])) if len(f)]
        self.results = pd.concat(frames, ignore_index=True).reindex(columns=RUN_COLUMNS)
        self.save()
        print(f"{key} | {head} | {text} | {row['holdout']:.2f} | {row['val']:.2f} | {row['secs']:.0f}s")
        return row

    def fail(self, key: str, mode: str, err: Exception) -> None:
        """Log a run failure to console and append failure details to `failed.csv`.

        Args:
            key: Image encoder key.
            mode: Stage or mode in which failure occurred.
            err: Caught exception instance.
        """
        print(f"FAILED {key}: {err}")
        file = self.run_dir / "failed.csv"
        pd.DataFrame([{"key": key, "mode": mode, "error": str(err)}]).to_csv(file, mode="a", header=not file.exists(), index=False)

    def visit(self, ctx: experiment.Context, key: str, cells: list[tuple[str, str]], mode: str, **run_kw) -> None:
        """Train the (text, head) `cells` of one image encoder (a finished cell is reused).

        A failure is logged and the benchmark goes on; the encoder's features are always freed
        (the combination-first rule).

        Args:
            ctx: Experiment context.
            key: Image encoder key.
            cells: List of (text_arm, head) tuples to run.
            mode: Benchmark mode name for error logging.
            **run_kw: Additional keyword arguments passed to `run`.
        """
        try:
            for text, head in cells:
                self.run(ctx, key, text, head, **run_kw)
        except Exception as e:   # logged in failed.csv
            self.fail(key, mode, e)
        finally:
            experiment.forget(ctx, key)

    def save(self) -> None:
        """Write the run table."""
        write_runs(self.results, self.run_dir / "results.parquet")


def screen_scores(results: pd.DataFrame, screen: list[str]) -> tuple[dict[str, float], list[str]]:
    """Extract best holdout scores for encoders whose screen embeddings are complete.

    Product head, a row for every screen embedding; also identifies encoders whose screen
    never trained at all. An embedding that never learned does not hide the other one.

    Args:
        results: Benchmark run table DataFrame.
        screen: List of question arm names comprising the screening set.

    Returns:
        tuple[dict[str, float], list[str]]: Mapping from encoder key to max holdout score,
            and list of encoders that failed to train.
    """
    scores, untrained = {}, []
    rows = results[(results["fusion"] == "product") & results["text"].isin(screen)]
    for key, grp in rows.groupby("backbone"):
        if not set(screen) <= set(grp["text"]):
            continue   # the screen of this encoder is still running
        trained = grp[grp["status"] != "untrained"]
        if trained.empty:
            untrained.append(key)
        else:
            scores[key] = float(trained["holdout"].max())
    return scores, untrained


def select_survivors(results: pd.DataFrame, screen: list[str], rest: list[str], keep_frac: float,
                     always: list[str]) -> tuple[list[str], list[str], list[str], float]:
    """Select surviving encoders to advance past the initial screening phase.

    Survivors: the best `keep_frac` of the screened encoders by best screen holdout, plus `always`.
    Below the cut an encoder is `kept` if it is in `always`, else `stopped`, unless it already has
    every `rest` embedding.

    Args:
        results: Benchmark run table DataFrame.
        screen: List of screening question arms.
        rest: Remaining question arms evaluated only on survivors.
        keep_frac: Fraction of top screened encoders to retain.
        always: Encoders always kept regardless of screen score.

    Returns:
        tuple[list[str], list[str], list[str], float]: Tuple of (survivors, stopped, kept, cutoff_score).
    """
    scores, _ = screen_scores(results, screen)
    if not scores:
        return [], [], [], math.nan
    n_keep = min(math.ceil(keep_frac * len(scores)), len(scores))
    cut = sorted(scores.values(), reverse=True)[n_keep - 1] if n_keep > 0 else math.inf
    survivors = [k for k in sorted(scores, key=scores.get, reverse=True) if scores[k] >= cut]
    survivors += [k for k in always if k not in survivors]
    product = results[results["fusion"] == "product"]

    def has_rest(k: str) -> bool:
        """Check if encoder `k` already has completed runs for all remaining arms.

        Args:
            k: Image encoder backbone key.

        Returns:
            bool: True if all `rest` embeddings have been run for encoder `k`.
        """
        return set(rest) <= set(product.loc[product["backbone"] == k, "text"])

    below = [k for k in scores if scores[k] < cut]
    return survivors, [k for k in below if k not in always and not has_rest(k)], [k for k in below if k in always], cut


def mark_screen(results: pd.DataFrame, screen: list[str], stopped: list[str], kept: list[str], cut: float) -> pd.DataFrame:
    """Update status and stop_ref columns on screen rows of the run table.

    Args:
        results: Benchmark run table DataFrame.
        screen: List of screening question arms.
        stopped: Encoders eliminated at the screen cutoff.
        kept: Encoders retained by virtue of `always` list despite scoring below cutoff.
        cut: Cutoff holdout accuracy score.

    Returns:
        pd.DataFrame: Updated run table copy with updated status and stop_ref.
    """
    df = results.copy()
    screen_rows = (df["fusion"] == "product") & df["text"].isin(screen) & (df["status"] != "untrained")
    df.loc[screen_rows, "status"] = "done"
    df.loc[screen_rows, "stop_ref"] = np.nan
    for label, keys in (("stopped", stopped), ("kept", kept)):
        rows = screen_rows & df["backbone"].isin(keys)
        df.loc[rows, "status"] = label
        df.loc[rows, "stop_ref"] = cut
    return df


def attention_plan(results: pd.DataFrame, screen: list[str], top_n: int, always: list[str], berts: list[str],
                   wildcards: int) -> list[tuple[str, str]]:
    """Construct the execution plan of (encoder, embedding) pairs for attention fusion.

    Pairs the `top_n` encoders by best product holdout, plus `always`, each with its own best
    embedding and `berts`; then the `wildcards` best stopped encoders with BERT-base.

    Args:
        results: Benchmark run table DataFrame.
        screen: Screening embeddings list.
        top_n: Number of top encoders by product holdout to include.
        always: Encoders always included.
        berts: BERT question arms to pair with top encoders.
        wildcards: Number of top stopped encoders to test with BERT-base attention.

    Returns:
        list[tuple[str, str]]: List of (encoder, text_arm) pairs in execution order.
    """
    product = results[results["fusion"] == "product"]
    stopped = set(product.loc[(product["status"] == "stopped") & ~product["backbone"].isin(always), "backbone"])
    usable = product[~product["backbone"].isin(stopped) & (product["status"] != "untrained") & product["holdout"].notna()
                     & (product["text"] != "random")]   # the random control is never an encoder's best embedding
    best = usable.sort_values("holdout", ascending=False).drop_duplicates("backbone").set_index("backbone")
    top = list(dict.fromkeys([*best.index[:top_n], *(k for k in always if k in best.index)]))
    top.sort(key=lambda k: best.loc[k, "holdout"], reverse=True)
    plan = [(k, t) for k in top for t in dict.fromkeys([best.loc[k, "text"], *berts])]
    stopped_screen = product[(product["status"] == "stopped") & product["text"].isin(screen) & product["holdout"].notna()]
    plan += [(k, "bert") for k in stopped_screen.groupby("backbone")["holdout"].max().nlargest(wildcards).index]
    return plan


# =============================================================================
# Tables of the report (pandas only)
# =============================================================================

def trained_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Filter run table to rows that successfully trained and have numeric validation scores.

    Args:
        runs: Run table DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with `val` cast to float.
    """
    runs_val = runs.assign(val=pd.to_numeric(runs["val"], errors="coerce"))
    return runs_val[(runs_val["status"] != "untrained") & runs_val["val"].notna()]


def screen_board(results: pd.DataFrame, screen: list[str], families: dict[str, str]) -> pd.DataFrame:
    """Generate screen leaderboard pivoting validation and holdout scores across screen arms.

    Args:
        results: Benchmark run table DataFrame.
        screen: Screening embeddings list.
        families: Mapping from encoder backbone key to family name string.

    Returns:
        pd.DataFrame: Leaderboard sorted descending by best validation score.
    """
    screen_all = results[(results["fusion"] == "product") & results["text"].isin(screen)]
    product = screen_all[screen_all["status"] != "untrained"]
    board = product.pivot(index="backbone", columns="text", values=["val", "holdout"]).reindex(screen_all["backbone"].unique())
    board.columns = [f"{t}_{v}" for v, t in board.columns]
    board = board.reindex(columns=[f"{t}_{v}" for t in screen for v in ("val", "holdout")])
    board.insert(0, "family", board.index.map(families))
    board["better_val"] = board[[f"{t}_val" for t in screen]].max(axis=1)
    label = lambda s: next((x for x in ("stopped", "kept") if (s == x).any()), "untrained" if (s == "untrained").all() else "done")
    board["status"] = screen_all.groupby("backbone")["status"].agg(label)
    return board.sort_values("better_val", ascending=False)


def write_runs(df: pd.DataFrame, path: Path) -> None:
    """Write run table DataFrame to parquet file atomically, normalizing attempt columns to text.

    Args:
        df: Run table DataFrame.
        path: Destination parquet file path.
    """
    df = df.copy()
    for c in ATTEMPT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v))
    with atomic_path(path) as tmp:
        df.to_parquet(tmp)


def collect_runs(paths: Paths, name: str = "benchmark") -> pd.DataFrame:
    """Read the benchmark run table at `runs_dir/<name>/results.parquet`.

    Args:
        paths: Project paths configuration.
        name: Benchmark run subfolder name under `runs_dir`. Defaults to 'benchmark'.

    Returns:
        pd.DataFrame: Run table ordered by backbone, question arm and fusion head.
    """
    file = paths.runs_dir / name / "results.parquet"
    df = pd.read_parquet(file) if file.is_file() else pd.DataFrame(columns=RUN_COLUMNS)
    return df.reindex(columns=RUN_COLUMNS).sort_values(["backbone", "text", "fusion"]).reset_index(drop=True)


def confidence_table(runs: pd.DataFrame, scores_dir: Path, image_ids: np.ndarray, *, top: int = 5, n_boot: int = 1000,
                     seed: int = 0) -> pd.DataFrame:
    """Compute bootstrap confidence intervals and paired differences for top runs against the best.

    Args:
        runs: Run table DataFrame.
        scores_dir: Directory containing the saved per-question validation scores (one `.npz` per run).
        image_ids: Array of cluster image IDs matching question ordering in score files.
        top: Number of top models to compare. Defaults to 5.
        n_boot: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for bootstrap resampling. Defaults to 0.

    Returns:
        pd.DataFrame: Table with validation accuracy, 95% CI, and paired difference against best run.
    """
    found = []
    for _, r in trained_runs(runs.reindex(columns=RUN_COLUMNS)).sort_values("val", ascending=False).iterrows():
        file = experiment.score_path(scores_dir, r["text"], r["backbone"], r["fusion"], float(r["lr"]), int(r["seed"]))
        if file.is_file():
            with np.load(file) as z:
                found.append((r, z["scores"]))
        if len(found) == top:
            break
    rows = []
    for r, scores in found:
        mean = cluster_bootstrap_mean(scores, image_ids, n_boot=n_boot, seed=seed)
        diff = cluster_bootstrap_diff(scores, found[0][1], image_ids, n_boot=n_boot, seed=seed)
        rows.append([r["text"], r["backbone"], r["fusion"], *(round(100 * x, 2) for x in (*mean, *diff))])
    return pd.DataFrame(rows, columns=["text", "backbone", "fusion", "val", "lo", "hi", "diff", "diff_lo", "diff_hi"])


def token_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Generate comparison table of token arms against sentence vectors and BERT-base.

    Args:
        runs: Run table DataFrame.

    Returns:
        pd.DataFrame: Formatted comparison table with paired difference columns.
    """
    diff_cols = ["d_0.6b_tokens_vs_vector", "d_4b_tokens_vs_vector", "d_4b_tokens_vs_bert"]
    out_cols = ["backbone", "fusion", *TOKEN_COLUMNS, *diff_cols]
    trained = trained_runs(runs.reindex(columns=RUN_COLUMNS))
    pairs = trained.loc[trained["text"].astype(str).str.endswith("_tokens"), ["backbone", "fusion"]].dropna().drop_duplicates()
    if pairs.empty:
        return pd.DataFrame(columns=out_cols)
    pairs = pairs.sort_values(["fusion", "backbone"]).reset_index(drop=True)
    piv = trained.pivot_table(index=["backbone", "fusion"], columns="text", values="val", aggfunc="last")
    t = piv.reindex(index=pd.MultiIndex.from_frame(pairs), columns=TOKEN_COLUMNS).reset_index()
    t.columns.name = None
    t["d_0.6b_tokens_vs_vector"] = t["qwen3_emb_0.6b_tokens"] - t["qwen3_emb_0.6b"]
    t["d_4b_tokens_vs_vector"] = t["qwen3_emb_4b_tokens"] - t["qwen3_emb_4b"]
    t["d_4b_tokens_vs_bert"] = t["qwen3_emb_4b_tokens"] - t["bert"]
    t[[*TOKEN_COLUMNS, *diff_cols]] = t[[*TOKEN_COLUMNS, *diff_cols]].round(2)
    return t[out_cols]
