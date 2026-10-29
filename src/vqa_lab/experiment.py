"""High-level experiment API: context, run, forget.

What `run(ctx, image, text, head)` does:
1. `_image_source` reads the image encoder's pooled vectors (product head) or 7x7 grids (attention head) from
   features/image/, extracting them first if they are missing (`_extract`), and standardizes them per feature.
2. `_question_source` loads the question inputs of the text arm (`text.question_inputs`), keeping only one large arm in memory.
3. `_fit` trains an H1 model (`train.train`) on the fit rows, early-stopping on the holdout rows; a run that stalls below STALL
   in its first epochs is retried with the next of ATTEMPTS, and the best attempt is kept.
4. `_score` scores the val2014 questions with the official metric; the row and the per-question scores are returned / saved.
Everything loaded stays in `ctx` (images, questions, targets) for the next run, until `forget` frees an image encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from transformers import set_seed   # seeds random, numpy, torch and every CUDA device

from vqa_lab.config import SEED, Paths, load_encoders, load_paths
from vqa_lab.data import Data, prepare_data
from vqa_lab.image import cnn_spec, decoded_dir, extract_batch, extract_features, grid_split, load_cnn, pooled_split
from vqa_lab.metrics import EvalResult, evaluate
from vqa_lab.text import BERT_ARMS, LARGE_ARMS, TOKEN_ARMS, VALID_ARMS, question_inputs
from vqa_lab.train import Grid, H1, Pooled, Rows, batches, feature_stats, predict, train


# recipe per head; the learning rates are fixed (not tuned): 8e-4 sits between the two rates that worked for the product head,
# 3e-4 is the rate attention trained at
HEADS = {"product": dict(batch=2048, norm=False, lr=8e-4), "attention": dict(batch=512, norm=True, lr=3e-4)}
STALL = 26.0   # holdout %: the "always yes" prior scores 24.4, so a run still below this after 3 epochs had a bad start, not a bad encoder
ATTEMPTS = [(0, 1.0), (101, 1.0), (0, 0.5)]   # (seed offset, learning-rate factor): as asked; seed + 101; the same seed at half the rate


@dataclass
class Context:
    """What `run` needs, and what it keeps loaded between runs (`forget` frees an image encoder's entries)."""
    paths: Paths
    device: str
    data: Data
    limit: int | None
    images: dict = field(default_factory=dict)   # (image key, head) -> the image sources and row positions (_image_source)
    questions: dict = field(default_factory=dict)   # question arm -> its inputs (text.question_inputs)
    targets: torch.Tensor | None = None   # the fit rows' soft targets, on the device
    image_names: dict | None = None   # image registry key -> loader name (both registries)


def context(limit: int | None = None, device: str | None = None, paths: Paths | None = None, answers: int = 1000) -> Context:
    """Build experiment context with dataset tables, device selection, and paths.

    Args:
        limit: Optional question sampling limit for testing/debugging.
        device: Target compute device ('cuda' or 'cpu'). Defaults to CUDA if available.
        paths: Optional Paths object, resolved via `load_paths()` if None.
        answers: Vocabulary size for answer classes. Defaults to 1000.

    Returns:
        Context: Initialized experiment execution context.
    """
    paths = paths if paths is not None else load_paths()
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return Context(paths=paths, device=device, data=prepare_data(paths, limit, answers=answers), limit=limit)


def _resolve_name(ctx: Context, image: str) -> str:
    """Resolve an image encoder key to its model loader name (timm, torchvision, or hf:).

    Args:
        ctx: Experiment context.
        image: Image encoder registry key.

    Returns:
        str: Model loader string specification.
    """
    if ctx.image_names is None:
        raw = load_encoders(ctx.paths, "image") + load_encoders(ctx.paths, "modern_image")
        ctx.image_names = {c["key"]: c["name"] for c in raw}
    return ctx.image_names.get(image, image)


def _extract(ctx: Context, image: str, grid: int | None):
    """Extract image features for train and val splits using the specified backbone.

    Args:
        ctx: Experiment context.
        image: Encoder registry key or loader name.
        grid: Grid size for spatial token extraction (e.g. 7), or None for pooled-only.
    """
    name = _resolve_name(ctx, image)
    net, spec = load_cnn(name, ctx.paths, ctx.device), cnn_spec(name)
    kind = "grid" if grid else "pooled"   # a grid extraction also writes the pooled vectors
    for split in ("train2014", "val2014"):
        extract_features(net, decoded_dir(ctx.paths, split), ctx.paths.image_features_dir / kind / image / split, size=spec["size"],
                         mean=spec["mean"], std=spec["std"], mode=spec["mode"], grid=grid, batch_size=extract_batch(spec["size"]),
                         device=ctx.device)
    del net
    torch.cuda.empty_cache()   # a no-op without CUDA


def _image_source(ctx: Context, image: str, head: str):
    """Load and standardize visual features for the given backbone and fusion head.

    Extracts features if missing and caches the standardized sources in `ctx.images`.

    Args:
        ctx: Experiment context.
        image: Image encoder key.
        head: Fusion head type ('product' or 'attention').

    Returns:
        tuple: (src_tr, src_va, rows_fit, rows_hold, rows_val, feature_dim).
    """
    if (image, head) in ctx.images:
        return ctx.images[(image, head)]
    getter = pooled_split if head == "product" else grid_split
    tr, va = getter(ctx.paths, image, "train2014"), getter(ctx.paths, image, "val2014")
    if tr is None or va is None:
        _extract(ctx, image, grid=7 if head == "attention" else None)
        tr, va = getter(ctx.paths, image, "train2014"), getter(ctx.paths, image, "val2014")
    ids_tr, arr_tr = tr
    ids_va, arr_va = va
    pos_tr, pos_va = pd.Series(np.arange(len(ids_tr)), index=ids_tr), pd.Series(np.arange(len(ids_va)), index=ids_va)   # image id -> row
    rows_fit, rows_hold = pos_tr.loc[ctx.data.fit.image_id].to_numpy(), pos_tr.loc[ctx.data.hold.image_id].to_numpy()
    rows_val = pos_va.loc[ctx.data.val.image_id].to_numpy()
    mu, sd = feature_stats(arr_tr, rows=np.unique(rows_fit))
    is_cuda = torch.device(ctx.device).type == "cuda"
    if head == "product":
        v_tr, v_va = (torch.as_tensor(arr_tr).float() - mu) / sd, (torch.as_tensor(arr_va).float() - mu) / sd
        if is_cuda:
            v_tr, v_va = v_tr.half(), v_va.half()
        src_tr, src_va = Pooled(v_tr.to(ctx.device)), Pooled(v_va.to(ctx.device))
    else:
        src_tr, src_va = Grid(arr_tr, mu, sd, ctx.device), Grid(arr_va, mu, sd, ctx.device)
    res = (src_tr, src_va, rows_fit, rows_hold, rows_val, arr_tr.shape[-1])
    ctx.images[(image, head)] = res
    return res


def _question_source(ctx: Context, text: str, gpu_states: bool = True) -> dict:
    """Load question inputs for the specified text arm into context memory.

    Manages memory by evicting other large arms from GPU cache when switching arms.

    Args:
        ctx: Experiment context.
        text: Question encoder arm name.
        gpu_states: Whether to keep BERT hidden states on the GPU. Defaults to True.

    Returns:
        dict: Mapping containing 'fit', 'hold', 'val' Rows and model kwargs.
    """
    if text in LARGE_ARMS:
        for other in [k for k in LARGE_ARMS if k != text and k in ctx.questions]:
            del ctx.questions[other]   # one large arm on the GPU at a time (BERT-base states 12 GB, BERT-large 16 GB)
        torch.cuda.empty_cache()   # a no-op without CUDA
    if text not in ctx.questions:
        ctx.questions[text] = question_inputs(text, ctx.data, ctx.paths, ctx.device, tag=f"_limit{ctx.limit}" if ctx.limit else "")
    inp = ctx.questions[text]
    # token arms: placement fixed per arm (the 4B states do not fit)
    on_gpu = TOKEN_ARMS[text][1] if text in TOKEN_ARMS else gpu_states and text in BERT_ARMS
    if on_gpu:
        for k in ("fit", "hold"):
            if not isinstance(inp[k].data, torch.Tensor):
                inp[k] = Rows(torch.as_tensor(np.asarray(inp[k].data)).to(ctx.device), device=ctx.device)
    return inp


def _targets(ctx: Context) -> torch.Tensor:
    """Retrieve and cache soft target tensor on the target device.

    Args:
        ctx: Experiment context.

    Returns:
        torch.Tensor: Precomputed soft target tensor on compute device.
    """
    if ctx.targets is None:
        ctx.targets = ctx.data.targets.to(ctx.device)
    return ctx.targets


def _score(ctx: Context, model: torch.nn.Module, src, rows, questions, df: pd.DataFrame) -> EvalResult:
    """Evaluate model predictions on a question split using official metrics.

    Args:
        ctx: Experiment context.
        model: Trained H1 model in evaluation mode.
        src: Image feature source container.
        rows: Image row indices for the questions.
        questions: Question input Rows.
        df: Ground-truth DataFrame for the evaluation split.

    Returns:
        EvalResult: Evaluation metric results object.
    """
    preds = predict(model, batches(src, rows, questions, size=1024, device=ctx.device), ctx.device)
    return evaluate({q: ctx.data.vocab[i] for q, i in zip(df["question_id"], preds)}, df)


def _fit(ctx: Context, image: str, text: str, head: str, *, lr: float, epochs: int, patience: int, batch: int | None,
         warmup_samples: int, gpu_states: bool):   # the defaults live in `run`
    """Fit an H1 model across retry attempts with early stopping on holdout accuracy.

    Tries training with seeds and learning rate schedules from `ATTEMPTS` if training
    stalls below `STALL` threshold in early epochs.

    Args:
        ctx: Experiment context.
        image: Image encoder key.
        text: Question encoder arm.
        head: Fusion head ('product' or 'attention').
        lr: Initial learning rate.
        epochs: Maximum training epochs.
        patience: Early stopping patience epochs.
        batch: Batch size override, or None to use default for head.
        warmup_samples: Number of warmup samples.
        gpu_states: Whether question states reside in GPU memory.

    Returns:
        tuple: (best_model, best_holdout, best_epochs, src_va, inputs, rows_val, info_dict).
    """
    src_tr, src_va, rows_fit, rows_hold, rows_val, dim = _image_source(ctx, image, head)
    inputs = _question_source(ctx, text, gpu_states)
    targets = _targets(ctx)
    eff_batch = batch or HEADS[head]["batch"]
    warmup = max(1, warmup_samples // eff_batch)

    records, best_model, best_i = [], None, 0
    for seed_off, lr_fac in ATTEMPTS:
        cur_seed, cur_lr = SEED + seed_off, lr * lr_fac
        set_seed(cur_seed)
        rng = np.random.default_rng(cur_seed)
        model = H1(dim, len(ctx.data.vocab), fusion=head, norm=HEADS[head]["norm"], **inputs["kwargs"]).to(ctx.device)
        curve = []

        def epoch_batches():
            """Yield training batches for the current epoch with shuffled image grouping.

            Returns:
                Iterator: Batches of (image, question, target).
            """
            return batches(src_tr, rows_fit, inputs["fit"], targets, size=eff_batch, rng=rng, device=ctx.device)

        def score_hold(m):
            """Evaluate model on the holdout split and record score in learning curve.

            Args:
                m: Model being evaluated.

            Returns:
                float: Overall holdout accuracy percentage.
            """
            curve.append(_score(ctx, m, src_tr, rows_hold, inputs["hold"], ctx.data.hold).overall)
            return curve[-1]

        best_hold, ep_run = train(model, epoch_batches, score_hold, lr=cur_lr, warmup=warmup, epochs=epochs, patience=patience,
                                  device=ctx.device)
        records.append({"seed": cur_seed, "lr": cur_lr, "holdout": best_hold, "epochs": ep_run,
                        "curve": ";".join(f"{x:.2f}" for x in curve)})
        if best_model is None or best_hold > records[best_i]["holdout"]:   # a tie keeps the earlier attempt
            best_model, best_i = model, len(records) - 1
        del model   # a losing model is freed here (the best one stays referenced by best_model)
        torch.cuda.empty_cache()   # a no-op without CUDA
        if len(curve) < 3 or max(curve[:3]) >= STALL:   # a healthy start: no further attempt
            break

    reported = records[best_i]
    discarded = [r for i, r in enumerate(records) if i != best_i]   # in attempt order, as in the published rows
    info = {   # several attempts are joined with "|", a learning curve with ";" (read back by benchmark._attempts / _curve_epochs)
        "restarted": len(discarded) > 0,
        "seed_used": reported["seed"],
        "lr_used": reported["lr"],
        "curve": reported["curve"],
        "discarded_seed": "|".join(str(h["seed"]) for h in discarded),
        "discarded_lr": "|".join(str(h["lr"]) for h in discarded),
        "discarded_holdout": "|".join(f"{h['holdout']:.2f}" for h in discarded),
        "discarded_curve": "|".join(h["curve"] for h in discarded),
    }
    return best_model, reported["holdout"], reported["epochs"], src_va, inputs, rows_val, info


def score_path(scores_dir: Path, text: str, image: str, head: str, lr: float, seed: int) -> Path:
    """Return file path for saving per-question validation scores of a benchmark run.

    Args:
        scores_dir: Directory where score arrays are stored.
        text: Question encoder arm name.
        image: Image backbone key.
        head: Fusion head name ('product' or 'attention').
        lr: Learning rate the run was asked for (a retry at half the rate keeps this file name).
        seed: The benchmark seed (a retry with another seed keeps this file name).

    Returns:
        Path: Formatted `.npz` score file path.
    """
    return Path(scores_dir) / f"{text}_{image}_{head}_{lr:g}_{seed}.npz"


def run(ctx: Context, image: str, text: str, head: str = "product", *, lr: float | None = None, epochs: int = 50,
        patience: int = 4, batch: int | None = None, warmup_samples: int = 358_400, gpu_states: bool = True,
        scores_dir: Path | None = None) -> dict:
    """Train one H1 model and evaluate it on the full val2014 dataset.

    Args:
        ctx: Experiment context.
        image: Image encoder backbone identifier.
        text: Question encoder arm identifier.
        head: Fusion architecture ('product' or 'attention'). Defaults to 'product'.
        lr: Learning rate override, or None to use default from `HEADS`.
        epochs: Maximum number of training epochs. Defaults to 50.
        patience: Early stopping patience epochs. Defaults to 4.
        batch: Batch size override, or None to use head default.
        warmup_samples: Number of training samples for learning rate warmup.
        gpu_states: Whether to store question states in GPU VRAM. Defaults to True.
        scores_dir: Optional directory to save per-question score array.

    Returns:
        dict: Benchmark run record dictionary with configuration, accuracies, and timing.

    Raises:
        ValueError: If `head` or `text` encoder is unrecognized.
    """
    if head not in HEADS:   # checked before any feature extraction, which can take minutes
        raise ValueError(f"unknown head {head!r}; choose one of: {', '.join(HEADS)}")
    if text not in VALID_ARMS:
        raise ValueError(f"unknown question encoder {text!r}; choose one of: {', '.join(VALID_ARMS)}")
    t0, lr = time.time(), HEADS[head]["lr"] if lr is None else lr
    model, best_hold, epochs_run, src_va, inputs, rows_val, info = _fit(
        ctx, image, text, head, lr=lr, epochs=epochs, patience=patience, batch=batch, warmup_samples=warmup_samples,
        gpu_states=gpu_states,
    )
    val_res = _score(ctx, model, src_va, rows_val, inputs["val"], ctx.data.val)
    if scores_dir is not None:
        Path(scores_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path(scores_dir, text, image, head, lr, SEED), scores=val_res.per_question.to_numpy().astype(np.float32))
    row = {
        "text": text, "backbone": image, "fusion": head, "seed": SEED, "lr": lr, "epochs": epochs_run,
        "holdout": best_hold, "val": val_res.overall, "yes_no": val_res.per_answer_type.get("yes/no", 0.0),
        "number": val_res.per_answer_type.get("number", 0.0), "other": val_res.per_answer_type.get("other", 0.0),
        "secs": round(time.time() - t0, 1), **info,
    }
    del model
    torch.cuda.empty_cache()   # a no-op without CUDA
    return row


def forget(ctx: Context, image: str, *, disk: bool = True) -> None:
    """Free cached image features from memory and optionally remove files from disk.

    Follows the combination-first rule to minimize disk storage requirements by
    keeping only one encoder's features at a time.

    Args:
        ctx: Experiment context.
        image: Image encoder identifier key.
        disk: Whether to delete extracted feature directories from disk. Defaults to True.
    """
    for head in HEADS:
        ctx.images.pop((image, head), None)
    if disk:
        for sub in ("pooled", "grid"):
            shutil.rmtree(ctx.paths.image_features_dir / sub / image, ignore_errors=True)
