"""VLM zero-shot answers through vLLM. Runs as its own process, as a file (`vlm.answer_command`), so the GPU is freed when it
exits and it imports nothing from vqa_lab: standard library, pandas and vLLM (torch and transformers only to record their versions)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import pandas as pd

SEED: int = 123
PROMPT: str = "{question}\nAnswer the question using a single word or phrase."


def conversation(question: str, image: str | Path) -> list[dict]:
    """Format a single question and image into a one-turn chat conversation structure.

    Args:
        question: Text question string.
        image: Path of the image file; a relative path resolves against the working directory.

    Returns:
        list[dict]: Single-turn chat messages list formatted for multimodal vLLM inference.
    """
    url = f"file://{image if os.path.isabs(image) else Path(image).resolve()}"
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": PROMPT.format(question=question)},
        ],
    }]


def clean_answer(text: str) -> str:
    """Extract and return the first non-empty stripped line of model output.

    Args:
        text: Raw generation text from the model.

    Returns:
        str: Cleaned single-line answer string.
    """
    return next((s for line in text.splitlines() if (s := line.strip())), "")


def plan_chunks(n: int, size: int) -> list[tuple[int, int]]:
    """Partition a total count into (start, end) index intervals for batch chunking.

    Args:
        n: Total number of items to process.
        size: Maximum chunk size.

    Returns:
        list[tuple[int, int]]: List of (start_idx, end_idx) index tuples.
    """
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def pending(out_dir: Path, n_chunks: int) -> list[int]:
    """Identify chunk indices whose output parquet files have not yet been completed.

    Args:
        out_dir: Directory where answer parquet chunks are saved.
        n_chunks: Total number of expected chunks.

    Returns:
        list[int]: List of chunk indices that still need to be processed.
    """
    out = Path(out_dir)
    return [i for i in range(n_chunks) if not (out / f"answers_{i:03d}.parquet").exists()]


def _write_environment_json(out_dir: Path) -> None:
    """Write environment versions and GPU hardware metadata to `environment.json`.

    Records library versions (vllm, torch, transformers) and the GPU device name
    in an atomic write.

    Args:
        out_dir: Destination directory where `environment.json` is stored.
    """
    env_file = out_dir / "environment.json"
    if env_file.exists():
        return
    import torch
    import transformers
    import vllm

    data = {
        "vllm": str(vllm.__version__),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    partial_file = out_dir / "environment.json.partial"
    partial_file.write_text(json.dumps(data, indent=2))
    os.replace(partial_file, env_file)


def main(argv: list[str] | None = None) -> None:
    """Answer questions in chunks with a vision-language model through vLLM.

    Parses command-line arguments, sets up the vLLM engine, iterates through
    pending question chunks, records timing statistics, and saves output parquet files.

    Args:
        argv: Optional list of command-line argument strings. Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(description="Answer VQA questions with a VLM via vLLM.")
    parser.add_argument("--model", type=str, required=True, help="Model directory")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions parquet file")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save answer chunks")
    parser.add_argument("--chunk", type=int, default=20000, help="Number of questions per chunk")
    parser.add_argument("--batch", type=int, default=512, help="Inference batch size")
    parser.add_argument("--max-tokens", type=int, default=10, help="Maximum generated tokens")
    parser.add_argument("--mm-processor-kwargs", type=str, default="{}", help="JSON processor kwargs")
    parser.add_argument("--chat-template-kwargs", type=str, default='{"enable_thinking": false}',
                        help="JSON chat template kwargs")
    parser.add_argument("--no-prefix-caching", action="store_true", default=False, help="Disable prefix caching")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory fraction for vLLM")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.questions)
    chunks = plan_chunks(len(df), args.chunk)
    pending_chunks = pending(out_dir, len(chunks))
    _write_environment_json(out_dir)
    if not pending_chunks:
        return

    from vllm import LLM, SamplingParams

    raw_parents = {os.path.dirname(img) for img in df["image"].unique()}
    parents = [str(Path(p).resolve()) for p in raw_parents]
    allowed_media_path = os.path.commonpath(parents)   # main returned above when there is nothing to answer

    llm_kwargs: dict[str, object] = {
        "model": args.model,
        "seed": SEED,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": not args.no_prefix_caching,
        "limit_mm_per_prompt": {"image": 1},
        "mm_processor_kwargs": json.loads(args.mm_processor_kwargs),
        "allowed_local_media_path": allowed_media_path,
    }
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    timing_file = out_dir / "timing.csv"

    for chunk_idx in pending_chunks:
        start, end = chunks[chunk_idx]
        chunk_df = df.iloc[start:end]
        t0 = time.perf_counter()
        rows: list[tuple[int, str, str, int]] = []

        for b_start in range(0, len(chunk_df), args.batch):
            batch_df = chunk_df.iloc[b_start : b_start + args.batch]
            conversations = [conversation(q, img) for q, img in zip(batch_df["question"], batch_df["image"])]
            outputs = llm.chat(conversations, sampling_params, chat_template_kwargs=chat_template_kwargs, use_tqdm=False)
            for qid, out in zip(batch_df["question_id"], outputs):
                raw_text = out.outputs[0].text if out.outputs else ""
                ptids = getattr(out, "prompt_token_ids", None)
                rows.append((qid, clean_answer(raw_text), raw_text, len(ptids) if ptids is not None else 0))

        duration = time.perf_counter() - t0
        chunk_parquet = pd.DataFrame(rows, columns=["question_id", "answer", "raw_output", "prompt_tokens"])
        partial_file = out_dir / f"answers_{chunk_idx:03d}.parquet.partial"
        chunk_parquet.to_parquet(partial_file, index=False)
        os.replace(partial_file, out_dir / f"answers_{chunk_idx:03d}.parquet")

        mean_tokens = float(chunk_parquet["prompt_tokens"].mean())   # a chunk is never empty
        timing_row = pd.DataFrame([{
            "chunk": chunk_idx,
            "questions": len(chunk_df),
            "seconds": round(duration, 4),
            "prompt_tokens_mean": round(mean_tokens, 2),
        }])
        timing_row.to_csv(timing_file, mode="a", index=False, header=not timing_file.exists())


if __name__ == "__main__":
    main()
