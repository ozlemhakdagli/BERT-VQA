"""Tests for VLM answer generation, answer collection, scoring, and subprocess CLI commands."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import transformers

from vqa_lab import vlm_answer
from vqa_lab.vlm import VLM_MODELS, answer_command, collect_answers, out_dir, question_file, score_answers
from vqa_lab.vlm_answer import PROMPT, SEED, clean_answer, conversation, pending, plan_chunks


def _val():   # three questions on two images, ten human answers each
    """Create a synthetic validation DataFrame with three questions across two images.

    Returns:
        pd.DataFrame: Synthetic question annotations with raw human answers.
    """
    return pd.DataFrame({"question_id": [21, 12, 11], "image_id": [2, 1, 1], "question": ["What color?", "How many?", "Is it?"],
                         "question_type": ["what color", "how many", "is"], "answer_type": ["other", "number", "yes/no"],
                         "answers_raw": [np.array(["red"] * 3 + ["blue"] * 7, dtype=object), np.array(["2"] * 10, dtype=object),
                                         np.array(["yes"] * 10, dtype=object)]})


def _answers(out, chunk, qids, answers):
    """Write a synthetic answers parquet chunk file.

    Args:
        out: Directory to store the output parquet file.
        chunk: Chunk index used in the filename.
        qids: Sequence of question identifiers.
        answers: Sequence of answer strings corresponding to the questions.
    """
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"question_id": qids, "answer": answers, "raw_output": answers, "prompt_tokens": [300] * len(qids)})
    df.to_parquet(out / f"answers_{chunk:03d}.parquet")


def test_question_file_and_command(paths):
    """Verify question parquet export sorting, sampling, and VLM CLI command generation.

    Args:
        paths: Project paths fixture.
    """
    f = question_file(_val(), paths)
    q = pd.read_parquet(f)
    assert f.name == "questions_full.parquet" and q["question_id"].tolist() == [11, 12, 21]   # sorted by image, then question
    assert q["image"].iloc[2] == str(paths.raw_dir / "val2014" / "COCO_val2014_000000000002.jpg")
    assert list(q.columns) == ["question_id", "question", "image"]   # the image id only sorts the rows and names the file
    assert question_file(_val(), paths, limit=2).name == "questions_n2.parquet"
    assert f.parent == paths.processed_dir / "vlm" and not (paths.runs_dir / "vlm").exists()   # a rebuildable input is data, not a result
    sample = pd.read_parquet(paths.processed_dir / "vlm/questions_n2.parquet")
    assert sample["question_id"].tolist() == [12, 21]   # the seed-123 sample (seed 0: [11, 12])
    cmd = answer_command(paths, "gemma4_e4b", f, prefix_caching=False)
    script = str(paths.root / "src/vqa_lab/vlm_answer.py")
    assert cmd[:8] == ["conda", "run", "--no-capture-output", "-n", "vqa-lab", "python", "-P", script]
    assert cmd[cmd.index("--model") + 1] == str(paths.raw_dir / "models/google/gemma-4-E4B-it") and "--tokenizer" not in cmd
    assert json.loads(cmd[cmd.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert all(m["chat_template_kwargs"] == {"enable_thinking": False} for m in VLM_MODELS.values())   # Qwen3.5 thinks by default
    assert json.loads(cmd[cmd.index("--mm-processor-kwargs") + 1]) == {"max_soft_tokens": 280}
    assert cmd[-1] == "--no-prefix-caching"
    assert cmd[cmd.index("--out-dir") + 1] == str(paths.artifacts_dir / "runs/vlm/gemma4_e4b/full_nocache")
    assert out_dir(paths, "qwen35_9b_fp8", f) == paths.artifacts_dir / "runs/vlm/qwen35_9b_fp8/full"


def test_collect_and_score(paths):
    """Verify answer collection across chunk files, error detection, and accuracy scoring.

    Args:
        paths: Project paths fixture.
    """
    f = question_file(_val(), paths)
    out = out_dir(paths, "gemma4_e4b", f)
    out.mkdir(parents=True)
    (out / "answers_000.parquet.partial").write_bytes(b"half")   # an unfinished chunk is not an answer file
    with pytest.raises(ValueError, match="No answer files"):
        collect_answers(f, out)
    _answers(out, 0, [11, 12], ["Yes.", "Two"])
    with pytest.raises(ValueError, match="Missing 1 questions"):
        collect_answers(f, out)   # question 21 is missing
    _answers(out, 1, [21], ["red"])
    r = score_answers(collect_answers(f, out), _val())
    assert r.per_question.loc[[11, 12, 21]].tolist() == pytest.approx([1.0, 1.0, 0.9])   # "Yes." and "Two" count only after normalisation
    assert r.overall == pytest.approx(290 / 3)
    _answers(out, 2, [99], ["blue"])
    with pytest.raises(ValueError, match=r"Extra 1 questions found in answers: \[99\]"):
        collect_answers(f, out)   # a question that is not in the question file
    _answers(out, 2, [21], ["blue"])
    with pytest.raises(ValueError, match="Duplicate question_id"):
        collect_answers(f, out)   # question 21 twice


def test_answer_script(tmp_path):   # vqa_lab/vlm_answer.py, run as a file in its own process (the GPU is freed when it exits)
    """Verify standalone dependencies, conversation formatting, and chunking in vlm_answer.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    tree = ast.parse(Path(vlm_answer.__file__).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    # only vLLM's own dependencies, never vqa_lab (mutmut: the import its rewritten copy of this file adds)
    assert imported - set(sys.stdlib_module_names) - {"mutmut"} <= {"pandas", "torch", "transformers", "vllm"}
    parts = conversation("What color is the dog?", "/d/COCO_val2014_000000000042.jpg")[0]["content"]
    assert parts[0] == {"type": "image_url", "image_url": {"url": "file:///d/COCO_val2014_000000000042.jpg"}}   # the image first
    assert parts[1] == {"type": "text", "text": "What color is the dog?\nAnswer the question using a single word or phrase."}
    assert SEED == 123 and PROMPT.format(question="Q") == "Q\nAnswer the question using a single word or phrase."
    assert clean_answer("  Yes.\nBecause the man is ...") == "Yes."
    assert clean_answer("\n\n 2 \n") == "2" and clean_answer("   ") == ""
    assert plan_chunks(45, 20) == [(0, 20), (20, 40), (40, 45)]
    (tmp_path / "answers_001.parquet").write_bytes(b"done")
    (tmp_path / "answers_002.parquet.partial").write_bytes(b"half")   # an interrupted write does not count
    assert pending(tmp_path, 3) == [0, 2]


class _FakeLLM:
    """Stands in for vllm.LLM: answers with the question's first word and counts one prompt token per question character."""

    instances: list["_FakeLLM"] = []

    def __init__(self, **kwargs):
        """Initialise the fake LLM instance and record instantiation arguments.

        Args:
            **kwargs: Configuration keyword arguments passed to the LLM constructor.
        """
        self.kwargs, self.asked, self.urls, self.settings = kwargs, [], [], set()
        _FakeLLM.instances.append(self)

    def chat(self, conversations, sampling_params, *, chat_template_kwargs, use_tqdm):
        """Simulate LLM chat completion by echoing the first word of each question.

        Args:
            conversations: Iterable of conversation turn sequences.
            sampling_params: Generation sampling parameters dictionary.
            chat_template_kwargs: Template formatting keyword arguments.
            use_tqdm: Whether progress bars are enabled during generation.

        Returns:
            list[SimpleNamespace]: Mock completion outputs with token IDs.
        """
        self.settings.add(json.dumps([sampling_params, chat_template_kwargs, use_tqdm]))
        outputs = []
        for (turn,) in conversations:
            image, text = turn["content"]
            question = text["text"].removesuffix("\nAnswer the question using a single word or phrase.")
            self.asked.append(question)
            self.urls.append(image["image_url"]["url"])
            if question == "Empty?":   # no completion and no prompt ids
                outputs.append(SimpleNamespace(outputs=[], prompt_token_ids=None))
            else:
                completion = SimpleNamespace(text=f"\n {question.split()[0]} \nmore")
                outputs.append(SimpleNamespace(outputs=[completion], prompt_token_ids=list(range(len(question)))))
        return outputs


@pytest.fixture
def fake_vllm(monkeypatch):
    """A stand-in `vllm` module on a machine without a GPU."""
    module = ModuleType("vllm")
    module.__version__ = "0.0-test"
    module.LLM = _FakeLLM
    module.SamplingParams = dict
    _FakeLLM.instances = []
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.usefixtures("fake_vllm")
def test_answer_script_main_and_resume(tmp_path, monkeypatch):
    """Verify end-to-end execution of the answer script CLI and resumption of partial runs.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        monkeypatch: Monkeypatch fixture for modifying directory and modules.
    """
    monkeypatch.chdir(tmp_path)
    images = (tmp_path / "imgs").resolve()
    images.mkdir()
    questions = pd.DataFrame({"question_id": [5, 6, 7, 8, 9], "image_id": [1, 1, 2, 2, 3],
                              "question": ["Red car?", "Two dogs?", "Empty?", "Blue sky?", "Yes no?"],
                              "image": [f"imgs/COCO_val2014_{i:012d}.jpg" for i in (1, 1, 2, 2, 3)]})   # relative paths
    questions.to_parquet(tmp_path / "q.parquet")
    out = tmp_path / "out"
    argv = ["--model", "models/m", "--questions", "q.parquet", "--out-dir", str(out), "--chunk", "2", "--batch", "1",
            "--mm-processor-kwargs", '{"max_soft_tokens": 280}']

    vlm_answer.main(argv)
    (llm,) = _FakeLLM.instances
    assert llm.kwargs == {"model": "models/m", "seed": 123, "max_model_len": 4096, "gpu_memory_utilization": 0.9,
                          "enable_prefix_caching": True, "limit_mm_per_prompt": {"image": 1},
                          "mm_processor_kwargs": {"max_soft_tokens": 280}, "allowed_local_media_path": str(images)}
    assert llm.settings == {json.dumps([{"temperature": 0.0, "max_tokens": 10}, {"enable_thinking": False}, False])}
    assert llm.asked == questions["question"].tolist()
    assert llm.urls[0] == f"file://{images / 'COCO_val2014_000000000001.jpg'}"   # resolved against the working directory
    assert sorted(p.name for p in out.glob("answers_*")) == ["answers_000.parquet", "answers_001.parquet", "answers_002.parquet"]
    answers = pd.concat([pd.read_parquet(out / f"answers_{i:03d}.parquet") for i in range(3)], ignore_index=True)
    assert answers["question_id"].tolist() == [5, 6, 7, 8, 9]
    assert answers["answer"].tolist() == ["Red", "Two", "", "Blue", "Yes"]   # the first non-empty line, stripped
    assert answers["raw_output"].tolist() == ["\n Red \nmore", "\n Two \nmore", "", "\n Blue \nmore", "\n Yes \nmore"]
    assert answers["prompt_tokens"].tolist() == [8, 9, 0, 9, 7]
    timing = pd.read_csv(out / "timing.csv")
    assert timing["chunk"].tolist() == [0, 1, 2] and timing["questions"].tolist() == [2, 2, 1]
    assert timing["prompt_tokens_mean"].tolist() == [8.5, 4.5, 7.0] and (timing["seconds"] >= 0).all()
    env = out / "environment.json"
    assert json.loads(env.read_text()) == {"vllm": "0.0-test", "torch": str(torch.__version__), "transformers": transformers.__version__, "gpu": "CPU"}
    assert not (out / "environment.json.partial").exists()
    env.write_text("{}")   # written once: a resumed run keeps the first environment

    (out / "answers_001.parquet").unlink()   # an interrupted run: chunk 1 is answered again, the others are kept
    (out / "answers_001.parquet.partial").write_bytes(b"half")
    vlm_answer.main(argv)
    assert len(_FakeLLM.instances) == 2 and _FakeLLM.instances[1].asked == ["Empty?", "Blue sky?"]
    assert pd.read_parquet(out / "answers_001.parquet")["question_id"].tolist() == [7, 8]
    assert not (out / "answers_001.parquet.partial").exists()
    assert pd.read_csv(out / "timing.csv")["chunk"].tolist() == [0, 1, 2, 1]   # appended; vlm_tables keeps the last row per chunk
    assert env.read_text() == "{}"

    vlm_answer.main(argv)   # nothing pending: the model is never loaded
    assert len(_FakeLLM.instances) == 2
