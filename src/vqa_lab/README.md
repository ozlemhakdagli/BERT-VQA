# `vqa_lab`

A modular, reproducible research library for controlled Visual Question Answering (VQA) benchmarking on the VQA v2 dataset.

---

## Overview

`vqa_lab` is the core Python engine powering the comparative analysis in this repository. It evaluates and compares visual encoders, question embeddings, and multimodal fusion architectures under standardized, strictly controlled experimental conditions:

- **Combinatorial Benchmarks:** Crosses classical CNN backbones (ResNet, ConvNeXt, EfficientNet, CLIP, etc.) and modern vision foundation models (SigLIP 2, C-RADIOv4, Qwen3-VL) against diverse question representations (static vectors: GloVe, fastText, word2vec; transformers: BERT-base, BERT-large; LLM states: Qwen3-Embedding).
- **Dual Fusion Mechanisms:** Supports both elementwise **Product** fusion (pooled visual vectors × question representations) and multi-glimpse spatial **Attention** fusion (questions attend over 7×7 feature map grids).
- **Rigorous Standardization:**
  - Fixed random seed (`123`) across all runs, initialization, and splits.
  - Image-disjoint train/holdout splits (early stopping images never overlap with training images).
  - Standard top-K answer classification (1,000 classical / 3,129 modern) using soft target cross-entropy.
  - Official VQA evaluation protocol on the entire unpruned `val2014` split (214,354 questions) with 10-annotator soft consensus scoring and official answer string normalization.
- **Zero-Shot VLM Baselines:** Evaluates modern Vision-Language Models (Gemma 4, Qwen3.5-9B) using vLLM in an isolated process without retraining.

---

## Script & Module Reference

The `src/vqa_lab/` package consists of 12 focused modules:

| Script / Module | Role | Key Responsibilities & Highlights |
|---|---|---|
| [`__init__.py`](__init__.py) | Package Entry Point | Declares `__version__ = "0.1.0"` and exports the high-level API functions: `context()`, `run()`, and `forget()`. |
| [`config.py`](config.py) | Configuration & Paths | Resolves repository root (`find_repo_root`), loads directory layouts from `configs/vqa_lab.yaml` into the `Paths` dataclass, loads encoder registries (`load_encoders`), manages atomic disk writes (`atomic_path`), and defines the global benchmark random seed (`SEED = 123`). |
| [`data.py`](data.py) | Dataset Pipeline | Handles remote downloading (`fetch_archive`, `fetch_hub`) and extraction (`extract_archive`); compiles raw JSON questions/annotations into consolidated parquet tables (`build_split_table`, `load_tables`); generates image-disjoint train/holdout splits (`holdout_mask`, `assert_image_disjoint`); creates answer vocabularies (`build_vocab`); computes 10-annotator soft target vectors (`soft_target`); and constructs the master `Data` container (`prepare_data`). |
| [`image.py`](image.py) | Vision Backbones & Caching | Builds and reads a 448×448 decoded image cache (`build_decoded_cache`, `decoded_dir`) to accelerate downstream processing; dynamically instantiates vision backbones across `timm`, `torchvision`, and Hugging Face (`load_cnn`); and extracts / reads pooled feature vectors and 7×7 feature map grids with memory-mapped disk caching (`extract_features`, `pooled_split`, `grid_split`). |
| [`text.py`](text.py) | Question Representations | Implements question representation pipelines ("arms"): tokenization (`tokenize_question`), loading static word vectors (GloVe, fastText, word2vec via `gensim`), caching sequential transformer token states (BERT-base, BERT-large, Qwen3-Embedding), generating pooled sentence vectors (`sentence-transformers`), and packing inputs for model consumption (`question_inputs`). |
| [`train.py`](train.py) | Model Architecture & Training | Defines the `H1` neural network architecture (BiLSTM / linear question projections + visual projection + Product or Multi-glimpse Attention fusion + 2-layer MLP classifier); manages streaming batch data structures (`Rows`, `Pooled`, `Grid`, `batches`); standardizes visual features (`feature_stats`); and executes AdamW optimization with binary cross-entropy on soft targets, linear warmup, and holdout early stopping (`train`, `predict`). |
| [`metrics.py`](metrics.py) | Official VQA Metrics | Implements the official VQA evaluation protocol: answer string normalization (`normalize_answer` handling contractions, punctuation, articles, digits), 10-annotator leave-one-out soft scoring (`soft_score`), full evaluation on predictions against annotations (`evaluate`, `EvalResult` breakdown by overall, yes/no, number, other), and cluster bootstrap confidence intervals (`cluster_bootstrap_diff`, `cluster_bootstrap_mean`). |
| [`experiment.py`](experiment.py) | High-Level Experiment Runner | Coordinates end-to-end runs: initializes execution state (`Context`, `context()`); orchestrates feature loading, on-the-fly extraction, and model training (`run()`); implements automated stall detection and multi-attempt retry policies (`ATTEMPTS`); saves per-question evaluation score arrays (`score_path`); and frees visual encoder memory and feature caches when done (`forget()`). |
| [`benchmark.py`](benchmark.py) | Benchmark Funnel & Leaderboards | Drives multi-stage benchmark funnels (screening -> survivors -> attention) using resumable parquet run tables (`RunTable`); records run parameters, durations, and learning curves; and builds the leaderboard tables as DataFrames (`screen_board`, `confidence_table`, `token_table`). |
| [`report.py`](report.py) | Report Aggregator | Aggregates all finished benchmark run tables (classical, modern, VLM baselines), along with retried, stopped, and failed runs, into a single unified Markdown report at `notebooks/benchmark_results.md` (`write_report`). |
| [`vlm.py`](vlm.py) | VLM Baselines Orchestration | Prepares validation question parquets containing absolute image file paths (`question_file`); constructs CLI execution commands for isolated VLM inference (`answer_command`); stitches together chunked answer parquets; evaluates predictions against ground truth using `metrics.evaluate`; and compiles VLM benchmark run tables. |
| [`vlm_answer.py`](vlm_answer.py) | Isolated VLM Answering Worker | A standalone CLI script executed in its own process with vLLM (`python -P src/vqa_lab/vlm_answer.py`). Does not import `vqa_lab` internals to avoid CUDA context pollution and memory leaks; generates chat completions in chunked batches with restartable disk checkpointing. |

---

## Core Design Principles

1. **Strict Reproducibility:** Every run, split, and weight initialization uses fixed `SEED = 123`. Hyperparameters are kept consistent across model comparisons rather than tuned per model.
2. **Image-Disjoint Holdout Validation:** Images in the `holdout` split (used for early stopping) never appear in the `fit` split. This prevents visual overfitting and leakage across questions referencing the same scene.
3. **Combination-First Resource Management:** Features for an image encoder are extracted and cached only when needed for its experimental arms. Once all text combinations and fusion heads for that image backbone are completed, `vqa_lab.forget(ctx, image_key)` removes heavy image feature caches from disk to preserve storage.
4. **Soft-Target Cross-Entropy:** Rather than treating VQA as single-label classification, models are trained with binary cross-entropy against continuous soft targets computed from 10 human annotations per question:
   $$\text{score} = \min\left(1.0, \frac{\text{agreement count}}{3}\right)$$
5. **Atomic Disk Operations:** All table writes, parquet saves, and feature checkpoints use `atomic_path`, ensuring incomplete writes from interrupted processes never corrupt state.

---

## Quickstart Usage

### Running an Experiment via Python API

```python
import vqa_lab

# 1. Initialize experiment context (loads tables, splits, vocab, device)
ctx = vqa_lab.context()

# 2. Run a training & evaluation job (extracts features if missing, trains H1, evaluates on val2014)
# Parameters: ctx, image_encoder_key, text_encoder_arm, fusion_head ("product" or "attention")
row = vqa_lab.run(ctx, "resnet50.tv_in1k", "glove", "product")

print(f"Val Accuracy: {row['val']:.2f}% (Yes/No: {row['yes_no']:.2f}%, Number: {row['number']:.2f}%, Other: {row['other']:.2f}%)")

# 3. Clean up visual feature cache when finished with this image encoder
vqa_lab.forget(ctx, "resnet50.tv_in1k")
```

### Inspecting Results & Generating Reports

```python
from vqa_lab.config import load_paths
from vqa_lab.report import write_report

paths = load_paths()
report_file = write_report(paths)
print(f"Report written to: {report_file}")
```

