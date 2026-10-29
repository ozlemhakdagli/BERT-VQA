"""Question encoders ("arms"): how the questions become the question input of the H1 model.

| family                | arms                                         | what H1 gets                                    |
|-----------------------|----------------------------------------------|-------------------------------------------------|
| static word vectors   | random, glove, fasttext, word2vec            | word ids -> frozen embedding -> BiLSTM           |
| BERT token states     | bert, bert_large                             | cached last-layer states -> BiLSTM               |
| Qwen token states     | qwen3_emb_0.6b_tokens, qwen3_emb_4b_tokens   | cached, standardised last-layer states -> BiLSTM |
| sentence vectors      | qwen3_emb_0.6b, qwen3_emb_4b, qwen3vl_emb_2b_text | one L2-normalised vector -> fully connected  |

`question_inputs(arm, ...)` returns {"fit", "hold", "val": Rows, "kwargs": H1 constructor arguments}.
"""
from __future__ import annotations

import os
import time
import zipfile
import zlib
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vqa_lab.config import SEED, Paths, atomic_path
from vqa_lab.data import Data
from vqa_lab.metrics import normalize_answer
from vqa_lab.train import Rows


MAX_LEN, N_OOV, DIM = 14, 50, 300

BERT_ARMS = ("bert", "bert_large")
SENTENCE_ARMS = {
    "qwen3_emb_0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3_emb_4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3vl_emb_2b_text": "Qwen/Qwen3-VL-Embedding-2B",
}
TOKEN_ARMS: dict[str, tuple[str, bool]] = {   # arm -> (hub repository, fit and holdout states on the GPU)
    "qwen3_emb_0.6b_tokens": ("Qwen/Qwen3-Embedding-0.6B", True),
    "qwen3_emb_4b_tokens": ("Qwen/Qwen3-Embedding-4B", False),
}
STATS_SAMPLE = 4096   # fit questions sampled (seed 123) for the token-state statistics
VALID_ARMS = ("random", "glove", "fasttext", "word2vec", *BERT_ARMS, *SENTENCE_ARMS, *TOKEN_ARMS)
STATIC_FILES = {   # gensim KeyedVectors files under raw_dir (GloVe and fastText converted by notebook 00: text_vectors_to_kv)
    "glove": "embeddings/glove/glove.840B.300d.kv",
    "fasttext": "embeddings/fasttext/wiki-news-300d-1M.kv",
    "word2vec": "embeddings/word2vec/word2vec-google-news-300.model",
}
LARGE_ARMS = (*BERT_ARMS, *SENTENCE_ARMS, *TOKEN_ARMS)   # question arms too big to keep two of on the GPU


def hub_repo(arm: str) -> str | None:
    """Return the Hugging Face repository identifier for a sentence or token arm.

    Args:
        arm: Text encoder arm identifier.

    Returns:
        str | None: Hugging Face repository string, or None if arm is not hosted on Hub.
    """
    return SENTENCE_ARMS.get(arm) or TOKEN_ARMS.get(arm, (None,))[0]


def tokenize_question(question: str, max_len: int = MAX_LEN) -> list[str]:
    """Tokenize question text using classical normalization rules up to `max_len` words.

    Applies lowercase conversion, strips punctuation and articles, and truncates to `max_len`.

    Args:
        question: Raw question string.
        max_len: Maximum number of tokens to keep. Defaults to MAX_LEN (14).

    Returns:
        list[str]: List of normalized word tokens.
    """
    return normalize_answer(question).split()[:max_len]


def collect_words(questions: Iterable[str]) -> set[str]:
    """Collect unique normalized vocabulary words across an iterable of question strings.

    Args:
        questions: Iterable of question text strings.

    Returns:
        set[str]: Set of unique token words found in the questions.
    """
    return {w for q in questions for w in tokenize_question(q, MAX_LEN)}


def text_vectors_to_kv(zip_path: Path, kv_file: Path) -> Path:
    """Convert downloaded GloVe or fastText archive to gensim KeyedVectors format once.

    Extracts text vectors, converts to KeyedVectors with an external memory-mapped array,
    and removes temporary files. Subsequent loads will memory-map `kv_file` rapidly.

    Args:
        zip_path: Path to downloaded zip archive containing text vector file.
        kv_file: Destination path for the `.kv` output file.

    Returns:
        Path: Path to the generated KeyedVectors file.
    """
    kv_file = Path(kv_file)
    if kv_file.is_file():
        return kv_file
    from gensim.models import KeyedVectors

    with zipfile.ZipFile(zip_path) as z:
        (member,) = [m for m in z.namelist() if m.endswith((".txt", ".vec"))]
        text_file = Path(z.extract(member, kv_file.parent))
    with open(text_file, "rb") as f:
        first = f.readline().split()
    has_header = len(first) == 2 and all(x.isdigit() for x in first)   # fastText starts with "<words> <dim>", GloVe does not
    kv = KeyedVectors.load_word2vec_format(str(text_file), binary=False, no_header=not has_header, unicode_errors="ignore")
    tmp = kv_file.with_name(".partial-" + kv_file.name)
    kv.save(str(tmp), separately=["vectors"])   # the vectors go to a .npy file next to it, which load_vectors memory-maps
    os.replace(f"{tmp}.vectors.npy", f"{kv_file}.vectors.npy")
    os.replace(tmp, kv_file)   # last: an existing .kv file means a finished conversion
    text_file.unlink()
    return kv_file


def load_vectors(path: Path, words: Collection[str]) -> dict[str, np.ndarray]:
    """Load pretrained word vectors for a specific set of words from a KeyedVectors file.

    Uses memory-mapping to read only the requested vectors into memory.

    Args:
        path: Path to the gensim KeyedVectors file.
        words: Collection of vocabulary words to load.

    Returns:
        dict[str, np.ndarray]: Mapping from word string to float32 embedding vector.
    """
    from gensim.models import KeyedVectors

    kv = KeyedVectors.load(str(path), mmap="r")
    return {w: np.array(kv[w], dtype=np.float32) for w in set(words) if w in kv.key_to_index}


def build_embedding_matrix(vectors: Mapping[str, np.ndarray], dim: int = DIM, n_oov: int = N_OOV,
                           seed: int = 0) -> tuple[np.ndarray, dict[str, int]]:
    """Build L2-normalized embedding matrix with padding, fixed OOV buckets, and known words.

    Row 0 represents padding (zeros), rows 1..n_oov are fixed random vectors for OOV words,
    and subsequent rows contain sorted known word vectors.

    Args:
        vectors: Mapping from word to embedding vector.
        dim: Embedding vector dimensionality. Defaults to DIM (300).
        n_oov: Number of out-of-vocabulary hash buckets. Defaults to N_OOV (50).
        seed: Random seed for initializing OOV bucket vectors. Defaults to 0.

    Returns:
        tuple[np.ndarray, dict[str, int]]: Tuple of (embedding_matrix [V, dim], word_to_id dict).
    """
    words = sorted(vectors)
    oov = np.random.default_rng(seed).standard_normal((n_oov, dim))
    known = np.stack([vectors[w] for w in words]) if words else np.zeros((0, dim))
    rows = np.concatenate([np.zeros((1, dim)), oov, known]).astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, norms, out=rows, where=norms > 0)
    return rows, {w: 1 + n_oov + i for i, w in enumerate(words)}


def encode_question(tokens: Sequence[str], word_to_id: Mapping[str, int], n_oov: int = N_OOV,
                    max_len: int = MAX_LEN) -> np.ndarray:
    """Map tokenized question words to embedding matrix row IDs with right-padding.

    Unknown words are hashed to one of `n_oov` deterministic random buckets via CRC32.

    Args:
        tokens: Sequence of word tokens.
        word_to_id: Mapping from known word to matrix row index.
        n_oov: Number of OOV buckets. Defaults to N_OOV (50).
        max_len: Maximum length of padded ID sequence. Defaults to MAX_LEN (14).

    Returns:
        np.ndarray: Padded int64 array of word IDs of length `max_len`.
    """
    ids = [word_to_id.get(w) or 1 + zlib.crc32(w.encode()) % n_oov for w in tokens[:max_len]]
    return np.array(ids + [0] * (max_len - len(ids)), dtype=np.int64)


def load_bert(model_dir: Path, device="cuda"):
    """Load local BERT tokenizer and frozen model in evaluation mode.

    Args:
        model_dir: Directory containing BERT model and tokenizer files.
        device: Device to place the model on. Defaults to 'cuda'.

    Returns:
        tuple[AutoTokenizer, AutoModel]: Loaded tokenizer and frozen model.
    """
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(str(model_dir)).eval().to(device).requires_grad_(False)
    return AutoTokenizer.from_pretrained(str(model_dir)), model


def token_lengths(questions, tokenizer) -> np.ndarray:
    """Calculate exact word-piece token lengths for each question without truncation.

    Args:
        questions: Iterable of question strings.
        tokenizer: Hugging Face tokenizer instance.

    Returns:
        np.ndarray: 1-D int64 array of token counts (including [CLS] and [SEP]).
    """
    qs = list(questions)
    chunks = (tokenizer(qs[i:i + 10_000], truncation=False)["input_ids"] for i in range(0, len(qs), 10_000))
    return np.array([len(ids) for chunk in chunks for ids in chunk], dtype=np.int64)


def _hidden_states(questions: Sequence[str], tokenizer: Any, model: Any, *, max_len: int, batch_size: int,
                   device: torch.device | str, fp16_autocast: bool):
    """Yield batched last-layer hidden states and attention masks right-padded to `max_len`.

    Args:
        questions: Sequence of question strings.
        tokenizer: Tokenizer instance.
        model: Transformer model instance.
        max_len: Fixed maximum sequence length.
        batch_size: Number of questions per batch.
        device: Compute device.
        fp16_autocast: Whether to enable fp16 autocast during inference.

    Yields:
        tuple[int, torch.Tensor, torch.Tensor]: (start_index, hidden_states [B, max_len, d], mask [B, max_len]).
    """
    dev = torch.device(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16_autocast and dev.type == "cuda"):
        for start in range(0, len(questions), batch_size):
            enc = tokenizer(list(questions[start:start + batch_size]), padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(dev)
            yield start, model(**enc).last_hidden_state.float(), enc["attention_mask"].bool()


def encode_to_memmap(questions, out_path: Path, tokenizer, model, *, max_len: int, batch_size: int = 512, device="cuda",
                     fp16_autocast: bool = True, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
    """Stream last-layer float16 token states [N, max_len, d] into a memory-mapped .npy file.

    Zero-fills padding positions and optionally applies feature standardization.

    Args:
        questions: Sequence of question strings.
        out_path: Target `.npy` file path.
        tokenizer: Tokenizer instance.
        model: Transformer model instance.
        max_len: Fixed sequence length.
        batch_size: Batch size for inference. Defaults to 512.
        device: Compute device. Defaults to 'cuda'.
        fp16_autocast: Whether to run under fp16 autocast. Defaults to True.
        mean: Optional per-dimension mean array for standardization.
        std: Optional per-dimension std array for standardization.
    """
    dev = torch.device(device)
    scale = None if mean is None else tuple(torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (mean, std))
    with atomic_path(out_path) as tmp:
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(questions), max_len, model.config.hidden_size))
        for start, hidden, mask in _hidden_states(questions, tokenizer, model, max_len=max_len, batch_size=batch_size,
                                                  device=device, fp16_autocast=fp16_autocast):
            if scale is not None:
                hidden = (hidden - scale[0]) / scale[1]
            out[start:start + len(hidden)] = hidden.masked_fill(~mask.unsqueeze(-1), 0.0).half().cpu().numpy()
        out.flush()
        del out


def encode_sentences(arm: str, questions: list[str], model_dir: Path, *, batch_size: int = 256,
                     device="cuda") -> np.ndarray:
    """Encode question texts into L2-normalized float16 sentence vectors [N, d].

    Args:
        arm: Sentence arm identifier in `SENTENCE_ARMS`.
        questions: List of question strings.
        model_dir: Directory containing model weights.
        batch_size: Batch size for sentence encoding. Defaults to 256.
        device: Compute device. Defaults to 'cuda'.

    Returns:
        np.ndarray: float16 array of normalized sentence vectors [N, d].

    Raises:
        ValueError: If `arm` is not in `SENTENCE_ARMS`.
    """
    if arm not in SENTENCE_ARMS:
        raise ValueError(f"unknown sentence arm: {arm!r}")
    dev = torch.device(device)
    if arm == "qwen3vl_emb_2b_text":   # the model's chat template with its default instruction (right padding)
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(str(model_dir), dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).eval().to(dev)
        proc, system = AutoProcessor.from_pretrained(str(model_dir)), [{"type": "text", "text": "Represent the user's input."}]

        def encode(qs):
            """Apply chat template and tokenize a batch of questions for Qwen3-VL embedding.

            Args:
                qs: List of question strings.

            Returns:
                BatchEncoding: Tokenized and padded model inputs.
            """
            chats = [[{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": q}]}]
                     for q in qs]
            text = proc.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
            return proc(text=text, padding=True, return_tensors="pt")

        out = []
        with torch.inference_mode():
            for start in range(0, len(questions), batch_size):
                enc = encode(questions[start:start + batch_size]).to(dev)
                hidden, mask = model(**enc).last_hidden_state, enc["attention_mask"]
                last = (mask * torch.arange(mask.shape[1], device=dev)).argmax(1)   # the last real token, whatever the padding side
                out.append(F.normalize(hidden[torch.arange(len(last), device=dev), last].float(), dim=-1).half().cpu().numpy())
        del model
        torch.cuda.empty_cache()   # a no-op without CUDA
        return np.concatenate(out)
    else:   # Qwen3-Embedding: sentence-transformers
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": torch.bfloat16} if dev.type == "cuda" else {}
        model = SentenceTransformer(str(model_dir), device=str(dev), model_kwargs=model_kwargs)
        res = model.encode(questions, batch_size=batch_size, normalize_embeddings=True)
        del model
        torch.cuda.empty_cache()
        return res.astype(np.float16)


def token_state_stats(questions: list[str], tokenizer: Any, model: Any, *, max_len: int,
                      device: torch.device | str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-dimension mean and standard deviation over real (unpadded) token states.

    Args:
        questions: List of question strings.
        tokenizer: Tokenizer instance.
        model: Transformer model instance.
        max_len: Sequence padding length.
        device: Compute device. Defaults to 'cuda'.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (mean [d], std [d]) float32 arrays clamped at 1e-3.
    """
    states = _hidden_states(questions, tokenizer, model, max_len=max_len, batch_size=256, device=device, fp16_autocast=False)
    real = np.concatenate([hidden[mask].cpu().numpy() for _, hidden, mask in states])
    return real.mean(axis=0).astype(np.float32), np.clip(real.std(axis=0, ddof=int(len(real) > 1)), 1e-3, None).astype(np.float32)


def bert_source(arm: str, paths: Paths) -> Path:
    """Return the local raw model directory for a BERT arm.

    Args:
        arm: Arm name ('bert' or 'bert_large').
        paths: Project paths configuration.

    Returns:
        Path: Directory path to local BERT weights.

    Raises:
        ValueError: If `arm` is not in `BERT_ARMS`.
    """
    if arm not in BERT_ARMS:
        raise ValueError(f"unknown bert arm: {arm!r}")
    return paths.raw_dir / "models" / ("bert-base-uncased" if arm == "bert" else "bert-large-uncased")


def _state_rows(train_file: Path, val_file: Path, data: Data, device: torch.device | str) -> dict[str, Any]:
    """Construct fit, holdout, and validation Rows from token state cache files.

    Fit and holdout subsets are copied into RAM, while validation is read from memory map.

    Args:
        train_file: Path to train token states `.npy` file.
        val_file: Path to val token states `.npy` file.
        data: Data split containers.
        device: Target device for Rows tensors.

    Returns:
        dict[str, Any]: Dictionary containing 'fit', 'hold', 'val' Rows and H1 'kwargs'.
    """
    train_states, val_states = np.load(train_file, mmap_mode="r"), np.load(val_file, mmap_mode="r")
    return {
        "fit": Rows(train_states[data.fit.index.to_numpy()], device=device),
        "hold": Rows(train_states[data.hold.index.to_numpy()], device=device),
        "val": Rows(val_states, device=device),
        "kwargs": {"input_dim": int(train_states.shape[2])},
    }


def _cache_valid(path: Path, expected_rows: int) -> bool:
    """Check if a cached numpy states file exists and contains the expected row count.

    Args:
        path: Path to `.npy` file.
        expected_rows: Expected number of rows in the array.

    Returns:
        bool: True if file exists and has `expected_rows`, else False.
    """
    try:
        return path.is_file() and len(np.load(path, mmap_mode="r")) == expected_rows
    except Exception:
        return False


def _cached_states(tokenizer: Any, load: Any, cache: Path, data: Data, device: torch.device | str, tag: str, *,
                   fp16_autocast: bool, batch_size: int, standardise: bool) -> dict[str, Any]:
    """Encode or load cached token states for train and val question splits.

    Determines max sequence length from the 99.9th percentile of fit question token counts,
    computes standardization statistics if requested, and caches states to disk.

    Args:
        tokenizer: Tokenizer instance.
        load: Callable returning the model; it is called only when a state file or the statistics must be computed.
        cache: Cache directory path for storing state files.
        data: Data container.
        device: Target compute device.
        tag: String tag for naming cache files.
        fp16_autocast: Whether to enable fp16 autocast during inference.
        batch_size: Batch size for encoding.
        standardise: Whether to compute and apply per-dimension mean/std normalization.

    Returns:
        dict[str, Any]: Dictionary containing question input Rows and constructor kwargs.
    """
    max_len = int(np.percentile(token_lengths(data.fit["question"], tokenizer), 99.9))
    files = {split: cache / f"{split}{tag}_L{max_len}.npy" for split in ("train", "val")}
    stats_file = cache / f"stats{tag}_L{max_len}.npz"
    new_stats = standardise and not stats_file.is_file()
    todo = [(df, files[split]) for split, df in (("train", data.train), ("val", data.val))
            if new_stats or not _cache_valid(files[split], len(df))]
    if todo:
        model, mean, std = load(), None, None
        if new_stats:
            sample_pos = np.sort(np.random.default_rng(SEED).choice(len(data.fit), size=min(STATS_SAMPLE, len(data.fit)), replace=False))
            mean, std = token_state_stats(data.fit["question"].iloc[sample_pos].tolist(), tokenizer, model, max_len=max_len, device=device)
            with atomic_path(stats_file) as tmp:
                np.savez(tmp, mean=mean, std=std)
        elif standardise:
            with np.load(stats_file) as st:
                mean, std = st["mean"], st["std"]
        for df, file in todo:
            encode_to_memmap(df["question"].tolist(), file, tokenizer, model, max_len=max_len, batch_size=batch_size, device=device,
                             fp16_autocast=fp16_autocast, mean=mean, std=std)
        del model
        torch.cuda.empty_cache()   # a no-op without CUDA
    return _state_rows(files["train"], files["val"], data, device)


def question_inputs(arm: str, data: Data, paths: Paths, device: torch.device | str, tag: str = "") -> dict[str, Any]:
    """Prepare question inputs dictionary for training and evaluation for a given arm.

    Returns dictionary with 'fit', 'hold', 'val' (each a Rows container) and 'kwargs'
    containing model initialization arguments.

    Args:
        arm: Arm name from `VALID_ARMS`.
        data: Dataset container with train, val, fit, holdout splits.
        paths: Project paths configuration.
        device: Compute device.
        tag: Optional identifier tag for cached file naming. Defaults to "".

    Returns:
        dict[str, Any]: Mapping with keys 'fit', 'hold', 'val', and 'kwargs'.

    Raises:
        ValueError: If `arm` is not in `VALID_ARMS`.
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    if arm in SENTENCE_ARMS:
        return _sentence_inputs(arm, data, paths, device, tag)
    if arm in BERT_ARMS or arm in TOKEN_ARMS:
        return _state_inputs(arm, data, paths, device, tag)
    return _static_inputs(arm, data, paths, device)


def _sentence_inputs(arm: str, data: Data, paths: Paths, device: torch.device | str, tag: str) -> dict[str, Any]:
    """Load or compute cached sentence vector inputs for sentence embedding arms.

    Args:
        arm: Sentence arm identifier in `SENTENCE_ARMS`.
        data: Dataset container.
        paths: Project paths configuration.
        device: Compute device.
        tag: File naming tag.

    Returns:
        dict[str, Any]: Input dictionary with 'fit', 'hold', 'val' Rows and 'kwargs' (H1 argument 'vector_dim').
    """
    model_dir = paths.raw_dir / "models" / SENTENCE_ARMS[arm]
    cache = paths.text_features_dir / arm
    train_file = cache / f"train{tag}.npy"
    val_file = cache / f"val{tag}.npy"
    for df, file in ((data.train, train_file), (data.val, val_file)):
        if not _cache_valid(file, len(df)):
            vecs = encode_sentences(arm, df["question"].tolist(), model_dir, device=device)
            with atomic_path(file) as tmp:
                np.save(tmp, vecs)
    train_vecs, val_vecs = np.load(train_file), np.load(val_file)

    def rows(v):   # float16 as cached; Rows.take returns float32
        """Wrap array of sentence vectors in a Rows container on the target device.

        Args:
            v: Numpy array of sentence vectors.

        Returns:
            Rows: Question Rows container.
        """
        return Rows(torch.as_tensor(v, device=device), device=device)

    return {"fit": rows(train_vecs[data.fit.index.to_numpy()]), "hold": rows(train_vecs[data.hold.index.to_numpy()]),
            "val": rows(val_vecs), "kwargs": {"vector_dim": int(train_vecs.shape[1])}}


def _state_inputs(arm: str, data: Data, paths: Paths, device: torch.device | str, tag: str) -> dict[str, Any]:
    """Load or compute cached token states for BERT or Qwen token arms.

    Args:
        arm: Arm identifier from `BERT_ARMS` or `TOKEN_ARMS`.
        data: Dataset container.
        paths: Project paths configuration.
        device: Compute device.
        tag: File naming tag.

    Returns:
        dict[str, Any]: Input dictionary with 'fit', 'hold', 'val' Rows and 'kwargs' (H1 argument 'input_dim').
    """
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device(device)
    if arm in BERT_ARMS:   # fp32 weights under fp16 autocast, raw states (how the classical caches were made)
        model_dir = bert_source(arm, paths)

        def load():
            """Load frozen BERT model in eval mode.

            Returns:
                AutoModel: Evaluation mode BERT model on target device.
            """
            return load_bert(model_dir, device)[1]

        return _cached_states(AutoTokenizer.from_pretrained(str(model_dir)), load, paths.text_features_dir / arm, data, device, tag,
                              fp16_autocast=True, batch_size=512, standardise=False)
    model_dir = paths.raw_dir / "models" / TOKEN_ARMS[arm][0]   # Qwen: bf16 weights, standardised states

    def load():
        """Load frozen Qwen model with appropriate floating point precision.

        Returns:
            AutoModel: Evaluation mode Qwen model on target device.
        """
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        return AutoModel.from_pretrained(str(model_dir), dtype=dtype).eval().to(dev).requires_grad_(False)

    return _cached_states(AutoTokenizer.from_pretrained(str(model_dir), padding_side="right"), load,
                          paths.text_features_dir / arm, data, device, tag,
                          fp16_autocast=False, batch_size=256, standardise=True)


def _static_inputs(arm: str, data: Data, paths: Paths, device: torch.device | str) -> dict[str, Any]:
    """Prepare word ID sequence inputs and frozen embedding matrix for static word vector arms.

    Args:
        arm: Static word vector arm ('random', 'glove', 'fasttext', 'word2vec').
        data: Dataset container.
        paths: Project paths configuration.
        device: Compute device.

    Returns:
        dict[str, Any]: Input dictionary with 'fit', 'hold', 'val' Rows and 'kwargs' (H1 argument 'embedding_matrix').
    """
    t0 = time.time()
    all_words = collect_words(data.questions)
    if arm == "random":
        rng = np.random.default_rng(0)
        vecs = {w: rng.standard_normal(DIM).astype(np.float32) for w in sorted(all_words)}
    else:
        vecs = load_vectors(paths.raw_dir / STATIC_FILES[arm], all_words)

    matrix, w2i = build_embedding_matrix(vecs)

    def enc(df) -> Rows:   # word ids of each question, padded to MAX_LEN
        """Encode question text column of a DataFrame into padded word ID tensor rows.

        Args:
            df: DataFrame containing a 'question' column.

        Returns:
            Rows: Padded word ID rows container on target device.
        """
        ids = [encode_question(tokenize_question(q), w2i) for q in df["question"]]
        arr = np.stack(ids) if ids else np.zeros((0, MAX_LEN), dtype=np.int64)
        return Rows(torch.as_tensor(arr, dtype=torch.long, device=device), device=device)

    cov = (len(vecs) / len(all_words)) if all_words else 0.0
    print(f"{arm:9s} {len(vecs):,} of {len(all_words):,} question words have a vector ({cov:.1%}); {time.time() - t0:.0f}s")

    return {"fit": enc(data.fit), "hold": enc(data.hold), "val": enc(data.val),
            "kwargs": {"embedding_matrix": torch.as_tensor(matrix, dtype=torch.float32)}}
