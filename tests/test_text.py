"""Tests for text representation arms, word vectors, BERT embeddings, and sentence encoders."""
from pathlib import Path
import zipfile
import zlib

import numpy as np
import pandas as pd
import pytest
import torch
import transformers
from gensim.models import KeyedVectors
from transformers import BertConfig, BertModel, BertTokenizerFast

from vqa_lab.data import Data
from vqa_lab.metrics import normalize_answer
from vqa_lab.text import (
    BERT_ARMS, SENTENCE_ARMS, STATIC_FILES, TOKEN_ARMS, bert_source, build_embedding_matrix, collect_words,
    encode_question, encode_sentences, encode_to_memmap, hub_repo, load_bert, load_vectors, question_inputs, text_vectors_to_kv,
    token_lengths, tokenize_question,
)


def _save_tiny_bert(model_dir: Path) -> Path:
    """Save a tiny BERT model and tokenizer for fast testing.

    Args:
        model_dir: Target directory for model files.

    Returns:
        Path: Path to saved model directory.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "color", "is", "the", "dog", "?", "man"]
    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True).save_pretrained(model_dir)
    BertModel(BertConfig(vocab_size=12, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=16, max_position_embeddings=64)).save_pretrained(model_dir)
    return model_dir


def _make_data(train_qs: list[str], val_qs: list[str], fit_idx=slice(None, 2), hold_idx=slice(2, None), vocab=("yes", "no")) -> Data:
    """Construct a mock Data object with specified questions and splits.

    Args:
        train_qs: Training question string list.
        val_qs: Validation question string list.
        fit_idx: Index selector for fit subset.
        hold_idx: Index selector for holdout subset.
        vocab: Vocabulary answer sequence.

    Returns:
        Data: Synthetic data container object.
    """
    train, val = pd.DataFrame({"question": train_qs}), pd.DataFrame({"question": val_qs})
    fit, hold = train.iloc[fit_idx], train.iloc[hold_idx]
    return Data(train=train, val=val, fit=fit, hold=hold, questions=pd.concat([train["question"], val["question"]], ignore_index=True),
                vocab=list(vocab), targets=torch.zeros((len(fit), len(vocab)), dtype=torch.float16))


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    """Module-scoped fixture providing a loaded tiny BERT tokenizer and model."""
    return load_bert(_save_tiny_bert(tmp_path_factory.mktemp("bert") / "model"), device="cpu")


@pytest.fixture
def dummy_data():
    """Fixture providing synthetic questions and splits for text representation tests."""
    return _make_data(["what color is the dog ?", "what is the dog", "is the dog", "dog"], ["what color ?"], fit_idx=[0, 2], hold_idx=[1, 3])


def test_tokenize_and_encode_question():
    """Verify question tokenization, vocabulary indexing, hashing OOV tokens, and padding."""
    q = "What color is THE dog's collar?"
    assert tokenize_question(q) == normalize_answer(q).split()
    assert tokenize_question(q, max_len=2) == normalize_answer(q).split()[:2]
    assert collect_words([q, "What is this?"]) >= {"what", "color", "this"}
    ids = {"what": 51, "is": 52}
    out = encode_question(["what", "zzz", "is"], ids, n_oov=50, max_len=5)
    assert out.dtype == np.int64 and out.tolist() == [51, 1 + zlib.crc32(b"zzz") % 50, 52, 0, 0]
    assert encode_question(["what"] * 9, ids, max_len=4).tolist() == [51] * 4


def test_text_vectors_to_kv_converts_once_and_removes_the_text(tmp_path):
    """Verify conversion of vector text files into KeyedVectors format and source cleanup.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    for name, text in (("v.vec", "3 3\nthe 0.1 0.2 0.3\ndog 1 0 0\ndog 9 9 9\n"),   # fastText: a header line
                       ("v.txt", "the 0.1 0.2 0.3\ndog 1 0 0\ndog 9 9 9\n")):        # GloVe: no header
        zip_path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr(name, text)
        kv_file = tmp_path / name.replace(".", "_") / "v.kv"
        assert text_vectors_to_kv(zip_path, kv_file) == kv_file
        assert not (kv_file.parent / name).exists() and not list(kv_file.parent.glob(".partial-*"))   # text removed, nothing half-written
        got = load_vectors(kv_file, {"the", "dog", "missing"})
        assert set(got) == {"the", "dog"} and got["dog"].dtype == np.float32
        assert got["dog"].tolist() == [1, 0, 0]   # the first of two "dog" lines wins
        assert got["the"].tolist() == np.array([0.1, 0.2, 0.3], dtype=np.float32).tolist()
        zip_path.unlink()
        assert text_vectors_to_kv(zip_path, kv_file) == kv_file   # converted once: the zip is not needed again


def test_load_vectors_is_case_sensitive_and_exact(tmp_path):
    """Verify exact case-sensitive vector lookup from KeyedVectors format.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["the", "dog", "Dog"], np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32))
    kv.save(str(tmp_path / "w2v.model"))
    got = load_vectors(tmp_path / "w2v.model", {"dog", "Dog", "DOG", "cat"})
    assert set(got) == {"dog", "Dog"} and got["dog"].dtype == np.float32
    assert got["dog"].tolist() == [0, 2, 0] and got["Dog"].tolist() == [0, 0, 3]
    with pytest.raises(FileNotFoundError):
        load_vectors(tmp_path / "missing.model", {"dog"})


def test_embedding_matrix_layout_and_determinism():
    """Verify embedding matrix initialization, OOV slot positioning, and seed determinism."""
    vecs = {"b": np.array([3.0, 4.0]), "a": np.array([0.0, 2.0]), "z": np.zeros(2)}
    m, ids = build_embedding_matrix(vecs, dim=2, n_oov=4, seed=1)
    assert m.shape == (1 + 4 + 3, 2) and m.dtype == np.float32
    assert ids == {"a": 5, "b": 6, "z": 7}
    assert not m[0].any() and not m[7].any()
    assert np.allclose(np.linalg.norm(m[1:7], axis=1), 1) and m[6].tolist() == pytest.approx([0.6, 0.8])
    assert np.array_equal(m, build_embedding_matrix(vecs, dim=2, n_oov=4, seed=1)[0])
    assert not np.array_equal(m, build_embedding_matrix(vecs, dim=2, n_oov=4, seed=2)[0])


def _encode(questions, tok, model, tmp_path, **kw):
    """States of `questions` as the cache file stores them (float16, padding exactly 0)."""
    out = tmp_path / f"states_{len(list(tmp_path.glob('states_*')))}.npy"
    encode_to_memmap(questions, out, tok, model, device="cpu", **kw)
    return np.load(out)


def test_bert_encoding_and_batch_invariance(tiny_bert, tmp_path):
    """Verify BERT token encoding produces identical representations regardless of batch size.

    Args:
        tiny_bert: Fixture providing tiny BERT tokenizer and model.
        tmp_path: Temporary directory fixture provided by pytest.
    """
    tok, model = tiny_bert
    questions = ["what color is the dog ?", "dog"]
    assert token_lengths(questions, tok).tolist() == [8, 3]
    out = _encode(questions, tok, model, tmp_path, max_len=10, batch_size=2)
    assert out.shape == (2, 10, 8) and out.dtype == np.float16
    assert np.all(out[1, 3:] == 0.0) and not np.all(out[1, :3] == 0.0)
    assert np.all(out[0, 8:] == 0.0) and not np.all(out[0, :8] == 0.0)
    assert _encode(["what color is the dog ?"], tok, model, tmp_path, max_len=4, batch_size=1).shape == (1, 4, 8)
    q3 = questions + ["what color is the man ?"]
    b1 = _encode(q3, tok, model, tmp_path, max_len=10, batch_size=1)
    b3 = _encode(q3, tok, model, tmp_path, max_len=10, batch_size=3)
    assert b1.shape == b3.shape == (3, 10, 8) and np.allclose(b1, b3, atol=1e-2)


def test_random_and_static_arms_and_unknown_arm(dummy_data, paths, monkeypatch):
    """Verify the random arm layout, that static arms read their KeyedVectors files, and the error on an unknown arm.

    Args:
        dummy_data: Synthetic data container fixture.
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for intercepting load_vectors.
    """
    res = question_inputs("random", dummy_data, paths, device="cpu")
    assert len(res["fit"]) == len(dummy_data.fit) and len(res["hold"]) == len(dummy_data.hold) and len(res["val"]) == len(dummy_data.val)
    for split in ("fit", "hold", "val"):
        assert res[split].take(torch.tensor([0])).dtype == torch.long
    fit_tokens = res["fit"].take(torch.tensor([0]))
    assert (fit_tokens[0, 4:] == 0).all()
    mat = res["kwargs"]["embedding_matrix"]
    assert mat.shape == (1 + 50 + len(collect_words(dummy_data.questions)), 300)
    assert torch.equal(mat[0], torch.zeros(300))
    words = mat[1 + 50:]
    assert len(words) > 1 and len({tuple(row.tolist()) for row in words}) == len(words)

    loaded = []
    monkeypatch.setattr("vqa_lab.text.load_vectors", lambda p, w: loaded.append(p) or {x: np.zeros(300, dtype=np.float32) for x in w})
    for arm in ("glove", "fasttext", "word2vec"):   # all three read gensim KeyedVectors files
        res = question_inputs(arm, dummy_data, paths, device="cpu")
        assert loaded[-1] == paths.raw_dir / STATIC_FILES[arm] and len(res["fit"]) == len(dummy_data.fit)
    assert STATIC_FILES["glove"].endswith(".kv") and STATIC_FILES["fasttext"].endswith(".kv")

    with pytest.raises(ValueError):
        question_inputs("unknown", dummy_data, paths, device="cpu")
    with pytest.raises(ValueError):
        bert_source("unknown", paths)


def test_bert_large_reads_cache_and_bert_source(dummy_data, paths, monkeypatch):
    """Verify bert_large path resolution, hub repositories, and caching behavior.

    Args:
        dummy_data: Synthetic data container fixture.
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking tokenizer loading.
    """
    assert BERT_ARMS == ("bert", "bert_large")
    assert bert_source("bert", paths) == paths.raw_dir / "models/bert-base-uncased"
    assert bert_source("bert_large", paths) == paths.raw_dir / "models/bert-large-uncased"
    assert hub_repo("qwen3_emb_4b") == "Qwen/Qwen3-Embedding-4B" and hub_repo("qwen3_emb_0.6b_tokens") == "Qwen/Qwen3-Embedding-0.6B"
    assert hub_repo("glove") is None and hub_repo("bert") is None
    cache_dir = paths.processed_dir / "features/text/bert_large"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train_L20.npy", np.zeros((len(dummy_data.train), 20, 8), dtype=np.float16))
    np.save(cache_dir / "val_L20.npy", np.zeros((len(dummy_data.val), 20, 8), dtype=np.float16))

    class DummyTokenizer:
        def __call__(self, texts, truncation=False):
            """Return mock token encoding dictionary with zero-filled IDs.

            Args:
                texts: Input question text list.
                truncation: Whether truncation is enabled.

            Returns:
                dict[str, list[list[int]]]: Input IDs dictionary.
            """
            return {"input_ids": [[0] * 20 for _ in texts]}

    tok_args = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: tok_args.append(args) or DummyTokenizer())
    res = question_inputs("bert_large", dummy_data, paths, "cpu")
    assert tok_args == [(str(paths.raw_dir / "models/bert-large-uncased"),)]
    assert res["kwargs"] == {"input_dim": 8} and len(res["fit"]) == len(dummy_data.fit) and len(res["val"]) == len(dummy_data.val)


def test_bert_states_follow_fit_and_hold_rows(dummy_data, paths, tmp_path):
    """Verify BERT hidden state sequence representations match fit/hold subsets and cache on disk.

    Args:
        dummy_data: Synthetic data container fixture.
        paths: Project paths fixture.
        tmp_path: Temporary directory fixture provided by pytest.
    """
    model_dir = _save_tiny_bert(paths.raw_dir / "models/bert-base-uncased")
    res = question_inputs("bert", dummy_data, paths, "cpu")
    tok, bert = load_bert(model_dir, device="cpu")
    max_len = res["fit"].data.shape[1]
    exp_fit = _encode(dummy_data.fit["question"].tolist(), tok, bert, tmp_path, max_len=max_len)
    exp_hold = _encode(dummy_data.hold["question"].tolist(), tok, bert, tmp_path, max_len=max_len)
    assert np.array_equal(res["fit"].data, exp_fit) and np.array_equal(res["hold"].data, exp_hold)

    train_cache = paths.processed_dir / f"features/text/bert/train_L{max_len}.npy"
    assert train_cache.is_file()
    assert len({row.tobytes() for row in np.load(train_cache, mmap_mode="r")}) == len(dummy_data.train)
    np.save(train_cache, np.zeros((2, max_len, 8), dtype=np.float16))
    res2 = question_inputs("bert", dummy_data, paths, "cpu")
    assert len(np.load(train_cache, mmap_mode="r")) == len(dummy_data.train)
    assert np.array_equal(res2["fit"].data, exp_fit) and np.array_equal(res2["hold"].data, exp_hold)


def test_sentence_arm_caching_and_rows(dummy_data, paths, monkeypatch):
    """Verify sentence embedding extraction, float16 disk caching, and subset row slicing.

    Args:
        dummy_data: Synthetic data container fixture.
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking sentence encoding.
    """
    d, calls = 16, []

    def fake_encode(arm, questions, model_dir, batch_size=256, device="cuda"):
        """Simulate sentence transformer encoding producing normalized vectors.

        Args:
            arm: Sentence arm identifier.
            questions: Question text list.
            model_dir: Model repository directory path.
            batch_size: Encoding batch size.
            device: Target execution device.

        Returns:
            np.ndarray: Normalized sentence embedding array.
        """
        assert arm == "qwen3_emb_0.6b" and model_dir == paths.raw_dir / "models" / SENTENCE_ARMS[arm] and device == "cpu"
        calls.append(len(questions))
        vecs = [np.random.default_rng(zlib.crc32(q.encode())).standard_normal(d).astype(np.float32) for q in questions]
        return np.array([v / np.linalg.norm(v) for v in vecs], dtype=np.float16)

    monkeypatch.setattr("vqa_lab.text.encode_sentences", fake_encode)

    res = question_inputs("qwen3_emb_0.6b", dummy_data, paths, "cpu")
    assert res["kwargs"] == {"vector_dim": d} and calls == [len(dummy_data.train), len(dummy_data.val)]
    cache_dir = paths.processed_dir / "features/text/qwen3_emb_0.6b"
    train_cache, val_cache = cache_dir / "train.npy", cache_dir / "val.npy"
    assert train_cache.is_file() and val_cache.is_file()

    exp_train = fake_encode("qwen3_emb_0.6b", dummy_data.train["question"].tolist(), paths.raw_dir / "models" / SENTENCE_ARMS["qwen3_emb_0.6b"], device="cpu")
    exp_val = fake_encode("qwen3_emb_0.6b", dummy_data.val["question"].tolist(), paths.raw_dir / "models" / SENTENCE_ARMS["qwen3_emb_0.6b"], device="cpu")
    assert np.array_equal(np.load(train_cache), exp_train)

    assert isinstance(res["fit"].data, torch.Tensor) and res["fit"].data.device.type == "cpu"
    assert res["fit"].data.dtype == torch.float16   # half the GPU memory; batches come out float32 with the exact cached values
    batch = res["fit"].take(torch.tensor([1, 0]))
    assert batch.dtype == torch.float32 and torch.equal(batch, torch.as_tensor(exp_train[[2, 0]], dtype=torch.float32))
    assert torch.equal(res["hold"].data, torch.as_tensor(exp_train[[1, 3]]))
    assert torch.equal(res["val"].data, torch.as_tensor(exp_val))

    # Re-encodes only train when train row count is wrong; valid val cache is kept
    calls.clear()
    np.save(train_cache, np.zeros((2, d), dtype=np.float16))
    res2 = question_inputs("qwen3_emb_0.6b", dummy_data, paths, "cpu")
    assert calls == [len(dummy_data.train)]
    assert len(np.load(train_cache)) == len(dummy_data.train) and torch.equal(res2["fit"].data, torch.as_tensor(exp_train[[0, 2]]))


def test_encode_sentences_qwen_and_vl(tmp_path, monkeypatch):
    """Verify Qwen and Qwen-VL sentence embedding pooling and L2 normalization.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        monkeypatch: Monkeypatch fixture for mocking processors and models.
    """
    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            """Simulate chat template application returning formatted prompt strings.

            Args:
                messages: Conversation message list.
                add_generation_prompt: Whether to append generation prompt.
                tokenize: Whether to tokenize the output.

            Returns:
                list[str]: Formatted text list.
            """
            return ["formatted prompt" for _ in messages]

        def __call__(self, text=None, padding=True, return_tensors="pt"):
            """Simulate tokenizer batch encoding with input IDs and attention masks.

            Args:
                text: Input text strings.
                padding: Padding strategy flag.
                return_tensors: Return tensor format.

            Returns:
                transformers.BatchEncoding: Encoded batch dictionary.
            """
            mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
            return transformers.BatchEncoding({"input_ids": torch.ones(len(text), 4, dtype=torch.long), "attention_mask": mask})

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kw):
            """Simulate model forward pass returning sequential hidden states.

            Args:
                input_ids: Input token tensor.
                attention_mask: Attention mask tensor.
                **kw: Additional keyword arguments.

            Returns:
                Any: Mock output object with last_hidden_state.
            """
            b, s = input_ids.shape
            return type("Out", (), {"last_hidden_state": torch.arange(1, b * s * 8 + 1, dtype=torch.float32).reshape(b, s, 8)})()

    class FakeST:
        def __init__(self, model_name_or_path, device=None, model_kwargs=None):
            """Initialise mock SentenceTransformer instance.

            Args:
                model_name_or_path: Model identifier or local directory.
                device: Target computation device.
                model_kwargs: Additional configuration parameters.
            """
            pass

        def encode(self, sentences, batch_size=256, normalize_embeddings=True):
            """Simulate sentence transformer batch encoding.

            Args:
                sentences: List of sentence strings.
                batch_size: Batch size for encoding.
                normalize_embeddings: Whether to normalize output embeddings.

            Returns:
                np.ndarray: Normalized embedding array.
            """
            h_st = torch.arange(1, len(sentences) * 4 * 8 + 1, dtype=torch.float32).reshape(len(sentences), 4, 8)
            return torch.nn.functional.normalize(h_st[:, -1], dim=-1).numpy()

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: FakeProcessor())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **kw: FakeModel().eval())

    h = torch.arange(1, 2 * 4 * 8 + 1, dtype=torch.float32).reshape(2, 4, 8)
    out = encode_sentences("qwen3_emb_0.6b", ["hello", "world"], tmp_path / "m", batch_size=2, device="cpu")
    assert out.shape == (2, 8) and out.dtype == np.float16
    assert np.allclose(np.linalg.norm(out.astype(np.float32), axis=1), 1.0, atol=1e-3)
    assert np.allclose(out.astype(np.float32), torch.nn.functional.normalize(h[:, -1], dim=-1).numpy(), atol=1e-3)

    out_vl = encode_sentences("qwen3vl_emb_2b_text", ["hello", "world"], tmp_path / "m", batch_size=2, device="cpu")
    assert out_vl.shape == (2, 8) and out_vl.dtype == np.float16
    assert np.allclose(np.linalg.norm(out_vl.astype(np.float32), axis=1), 1.0, atol=1e-3)
    assert np.allclose(out_vl.astype(np.float32), torch.nn.functional.normalize(h[[0, 1], [2, 1]], dim=-1).numpy(), atol=1e-3)

    with pytest.raises(ValueError):
        encode_sentences("unknown_arm", ["hello"], tmp_path / "m", device="cpu")


WORDS, END, PAD = {"what": 1, "color": 2, "is": 3, "it": 4, "zzz": 9}, 7, 0


class _TokTokenizer:   # word ids + end token, like Qwen3-Embedding; pads on the side it was loaded with
    def __init__(self, padding_side="left"):
        """Initialise mock tokenizer with specified padding side.

        Args:
            padding_side: Direction for sequence padding ('left' or 'right').
        """
        self.padding_side = padding_side

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        """Tokenize text into integer IDs with padding and attention masks.

        Args:
            texts: Sequence of question strings.
            padding: Whether to apply sequence padding.
            truncation: Whether to truncate sequence to max_length.
            max_length: Maximum allowed sequence length.
            return_tensors: Tensor format for return values.

        Returns:
            transformers.BatchEncoding or dict: Tokenization encoding container.
        """
        ids = [[WORDS[w] for w in t.split()] + [END] for t in texts]
        if not padding:
            return {"input_ids": ids}
        ids = [i[:max_length] for i in ids]
        right = self.padding_side == "right"
        pad = lambda i, v: (i + [v] * (max_length - len(i))) if right else ([v] * (max_length - len(i)) + i)
        return transformers.BatchEncoding({"input_ids": torch.tensor([pad(i, PAD) for i in ids]),
                                           "attention_mask": torch.tensor([pad([1] * len(i), 0) for i in ids])})


class _TokModel(torch.nn.Module):   # state = token id * (dim + 1) + position, so padding states are not 0
    config = type("Config", (), {"hidden_size": 3})()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        """Compute synthetic hidden states from token IDs and positions.

        Args:
            input_ids: Token ID tensor.
            attention_mask: Attention mask tensor.
            **kw: Additional keyword arguments.

        Returns:
            Any: Mock output object with last_hidden_state.
        """
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1)
        return type("Out", (), {"last_hidden_state": input_ids.unsqueeze(-1).float() * torch.tensor([1.0, 2.0, 3.0]) + pos})()


def _expected(q, L=4):   # the fake model's states of one question, right-padded, and its real-token count
    """Compute expected token states and real token count for synthetic test question.

    Args:
        q: Question text string.
        L: Sequence length limit.

    Returns:
        tuple[np.ndarray, int]: Simulated token feature matrix and count of real tokens.
    """
    ids = ([WORDS[w] for w in q.split()] + [END])[:L]
    h = np.zeros((L, 3))
    h[:len(ids)] = [[i * s + p for s in (1, 2, 3)] for p, i in enumerate(ids)]
    return h, len(ids)


def test_token_states_arm(paths, monkeypatch):
    """Verify token states feature extraction, statistics caching, and standardization.

    Args:
        paths: Project paths fixture.
        monkeypatch: Monkeypatch fixture for mocking tokenizers and models.
    """
    exp_dir = str(paths.raw_dir / "models" / TOKEN_ARMS["qwen3_emb_0.6b_tokens"][0])
    loaded_models = []

    def fake_tok(*a, **kw):
        """Mock AutoTokenizer.from_pretrained returning _TokTokenizer.

        Args:
            *a: Positional arguments.
            **kw: Keyword arguments including padding_side.

        Returns:
            _TokTokenizer: Configured tokenizer mock.
        """
        assert a == (exp_dir,)
        return _TokTokenizer(kw.get("padding_side", "left"))

    def fake_model(*a, **kw):
        """Mock AutoModel.from_pretrained returning _TokModel.

        Args:
            *a: Positional arguments.
            **kw: Keyword arguments including dtype.

        Returns:
            _TokModel: Configured model mock.
        """
        assert a == (exp_dir,) and kw.get("dtype") == torch.float32
        m = _TokModel()
        m.p = torch.nn.Parameter(torch.ones(1))
        loaded_models.append(m)
        return m

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tok)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    assert TOKEN_ARMS["qwen3_emb_4b_tokens"][1] is False
    data = _make_data(["what color is it", "color it", "what is it"], ["zzz zzz zzz"], vocab=("yes",))   # fit = rows 0-1, hold = row 2; val word must not move stats
    res = question_inputs("qwen3_emb_0.6b_tokens", data, paths, "cpu")
    assert loaded_models and not loaded_models[0].training and not loaded_models[0].p.requires_grad
    d = paths.processed_dir / "features/text/qwen3_emb_0.6b_tokens"
    tr, va, st = np.load(d / "train_L4.npy"), np.load(d / "val_L4.npy"), np.load(d / "stats_L4.npz")   # L = int(p99.9 of [5, 3]) = 4
    assert res["kwargs"] == {"input_dim": 3} and tr.dtype == np.float16 and tr.shape == (3, 4, 3) and va.shape == (1, 4, 3)
    fit_real = np.concatenate([_expected(q)[0][:_expected(q)[1]] for q in ["what color is it", "color it"]])   # fit only, no padding, no val
    mean, std = fit_real.mean(0), np.clip(fit_real.std(0, ddof=1), 1e-3, None)
    assert np.allclose(st["mean"], mean, atol=1e-4) and np.allclose(st["std"], std, atol=1e-4)
    for arr, i, q in ((tr, 0, "what color is it"), (tr, 1, "color it"), (tr, 2, "what is it"), (va, 0, "zzz zzz zzz")):
        h, n = _expected(q)
        assert np.allclose(arr[i, :n].astype(np.float32), (h[:n] - mean) / std, atol=2e-3)   # real tokens first (right padding), standardised
        assert (arr[i, n:] == 0).all()                                                        # padding exactly 0
    assert np.array_equal(res["fit"].data, tr[[0, 1]]) and np.array_equal(res["hold"].data, tr[[2]])

    st = dict(st)
    np.savez(d / "stats_L4.npz", **{**st, "mean": st["mean"] + 1.0})   # an interrupted run: saved statistics, train cache lost
    (d / "train_L4.npy").unlink()
    val_before = (d / "val_L4.npy").read_bytes()
    question_inputs("qwen3_emb_0.6b_tokens", data, paths, "cpu")
    h, n = _expected("color it")
    assert np.allclose(np.load(d / "train_L4.npy")[1, :n].astype(np.float32), (h[:n] - (st["mean"] + 1.0)) / st["std"], atol=2e-3)   # saved stats reused
    assert (d / "val_L4.npy").read_bytes() == val_before   # a valid cache is not encoded again
    monkeypatch.setattr("vqa_lab.text.STATS_SAMPLE", 3)   # three of 30 fit questions: the seed decides which
    many = _make_data([["what is it", "color is it", "what color it"][i % 3] for i in range(31)], ["zzz zzz zzz"],
                      fit_idx=slice(None, 30), hold_idx=slice(30, None), vocab=("yes",))
    question_inputs("qwen3_emb_0.6b_tokens", many, paths, "cpu", tag="_limit500")
    assert {p.name for p in d.iterdir()} >= {"train_limit500_L4.npy", "val_limit500_L4.npy", "stats_limit500_L4.npz"}
    rows = [many.fit["question"].iloc[i] for i in (0, 17, 19)]   # the seed-123 draw (seed 0 draws 15, 18, 23)
    real = np.concatenate([_expected(q)[0][:_expected(q)[1]] for q in rows])
    stats = np.load(d / "stats_limit500_L4.npz")
    assert np.allclose(stats["mean"], real.mean(0), atol=1e-4) and np.allclose(stats["std"], np.clip(real.std(0, ddof=1), 1e-3, None), atol=1e-4)
