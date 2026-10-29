# vqa-lab

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)
![Dataset: VQA v2](https://img.shields.io/badge/Dataset-VQA%20v2-orange.svg)

A controlled benchmark of Visual Question Answering (VQA) models on the VQA v2 dataset, comparing classical CNN + word-embedding architectures, modern vision-language encoders, and zero-shot foundation models—all evaluated with the official VQA metric on the entire `val2014` split.

---

## Benchmark Highlights & Results

The benchmark evaluates **54 pretrained CNN image encoders** crossed with **5 question embeddings** (GloVe, fastText, word2vec, BERT-base, and BERT-large) across two fusion heads:
- **Product head**: elementwise multiplication of pooled image vectors and question representations.
- **Attention head**: multi-glimpse question-guided spatial attention over 7×7 feature map grids.

All models predict from a fixed answer vocabulary, use an image-disjoint holdout set for early stopping, and are evaluated with the official VQA metric on the entire `val2014` split (214,354 questions) using fixed seed `123`.

Three follow-up studies extend the benchmark using the exact same evaluation protocol:
- **[Modern encoder benchmark](notebooks/README.md#modern-encoder-benchmark):** 5 modern vision foundation backbones (SigLIP 2, Perception Encoder, C-RADIOv4, CLIP ViT-L/14, Qwen3-VL-Embedding) crossed with Qwen3 sentence-embedding models and BERT-base across both heads with a 3,129-answer vocabulary.
- **[Token-state study](notebooks/README.md#token-state-study):** Qwen3-Embedding token states routed through the same BiLSTM as BERT for a like-for-like "LLM vs BERT" comparison.
- **[VLM zero-shot baselines](notebooks/README.md#vlm-zero-shot-baselines):** Gemma 4 E4B and Qwen3.5-9B answering questions directly through vLLM without any VQA fine-tuning.

### Headline Results

| Study | Best Combination / Model | Output Space | val2014 Accuracy | Yes/No | Number | Other |
|---|---|---|---|---|---|---|
| [**Classical Benchmark**](notebooks/README.md#results) (195 runs) | CLIP ConvNeXt-Base + BERT-base (Attention) | Top-1,000 answers | **60.27 %** | 77.94 % | 42.46 % | 51.55 % |
| [**Modern Encoders**](notebooks/README.md#modern-encoder-benchmark) (49 runs) | Qwen3-VL-Embedding + Qwen3-Embedding-4B (Attention) | Top-3,129 answers | **67.72 %** | 84.27 % | 53.52 % | 58.89 % |
| [**VLM Zero-Shot Baselines**](notebooks/README.md#vlm-zero-shot-baselines) | Qwen3.5-9B (RedHatAI FP8-dynamic) | Open-ended free text | **83.01 %** | 95.50 % | 75.51 % | 75.49 % |

- **Protocol, findings and caveats:** [`notebooks/README.md`](notebooks/README.md), starting with the [protocol](notebooks/README.md#protocol) and [how the recipe was chosen](notebooks/README.md#how-the-recipe-was-chosen)
- **Every run, including the stopped, retried and failed ones:** [`notebooks/benchmark_results.md`](notebooks/benchmark_results.md)
- **Detailed package documentation:** [`src/vqa_lab/README.md`](src/vqa_lab/README.md)

---

## Datasets & Raw Archives (`data/zip_file`)

All experiments are conducted on the official [VQA v2](https://visualqa.org/) benchmark balanced dataset paired with Microsoft COCO 2014 images. The raw archive files downloaded by [`notebooks/00_data.ipynb`](notebooks/00_data.ipynb) into `paths.zip_dir` (`data/zip_file`) are listed below:

| Archive Filename | Size | Source / URL | Extracted Contents & Role |
|---|---|---|---|
| `v2_Questions_Train_mscoco.zip` | 7.2 MB | [VQA v2 Questions Train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip) | `v2_OpenEnded_mscoco_train2014_questions.json` (443,757 training questions) |
| `v2_Questions_Val_mscoco.zip` | 3.5 MB | [VQA v2 Questions Val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip) | `v2_OpenEnded_mscoco_val2014_questions.json` (214,354 validation questions) |
| `v2_Questions_Test_mscoco.zip` | 9.0 MB | [VQA v2 Questions Test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip) | `test-dev2015` (107,394) and `test2015` (447,793 questions) |
| `v2_Annotations_Train_mscoco.zip` | 21.7 MB | [VQA v2 Annotations Train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip) | `v2_mscoco_train2014_annotations.json` (10 human answers per training question) |
| `v2_Annotations_Val_mscoco.zip` | 10.5 MB | [VQA v2 Annotations Val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip) | `v2_mscoco_val2014_annotations.json` (10 human answers per validation question) |
| `train2014.zip` | 13.5 GB | [COCO train2014 Images](http://images.cocodataset.org/zips/train2014.zip) | 82,783 MS COCO training images (`COCO_train2014_*.jpg`) |
| `val2014.zip` | 6.6 GB | [COCO val2014 Images](http://images.cocodataset.org/zips/val2014.zip) | 40,504 MS COCO validation images (`COCO_val2014_*.jpg`) |
| `glove.840B.300d.zip` | 2.2 GB | [Stanford GloVe 840B](https://nlp.stanford.edu/data/glove.840B.300d.zip) | 840B token 300-d word vectors, converted to `gensim` KeyedVectors format |
| `wiki-news-300d-1M.vec.zip` | 682 MB | [fastText English Vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) | 1M 300-d subword-informed fastText vectors, converted to `gensim` format |

> [!TIP]
> **Disk Cleanup:** Once the archives are extracted to `data/raw_files/` and compiled into Parquet tables under `data/processed/tables/`, the entire `data/zip_file/` folder can be safely deleted to free **~23 GB** of storage.

---

## Quickstart API

Train and evaluate any model-embedding combination with three lines of Python:

```python
import vqa_lab

# 1. Load context (splits, paths, vocabulary)
ctx = vqa_lab.context()

# 2. Run training and evaluation (extracts features on-demand, trains H1, scores on val2014)
row = vqa_lab.run(ctx, "resnet50.tv_in1k", "glove", "product")

# 3. Clean up visual feature cache after completing this image backbone
vqa_lab.forget(ctx, "resnet50.tv_in1k")
```

---

## Setup & Installation

### Conda Environment

```bash
conda env create -f environment.yml
conda activate vqa-lab
pytest
```

Optional verification checks:
- Code coverage: `pytest --cov=vqa_lab`
- Mutation testing: `mutmut run` followed by `mutmut results`

### Hardware Requirements

- **GPU:** NVIDIA GPU with $\ge$ 24 GB VRAM recommended. For cards with smaller memory, set `GPU_STATES = False` in `notebooks/03_encoder_benchmark.ipynb`.
- **Disk Space:**
  - **Download phase:** ~23 GB for downloaded zip archives in `data/zip_file/`.
  - **Extracted assets:** ~27 GB in `data/raw_files/` (COCO images, JSON annotations, word vectors).
  - **Model weights:** ~65 GB in `data/raw_files/models/` (the CNN encoders, BERT, the Qwen embedding models and, for notebook 05, ~29 GB of VLM weights).
  - **Decoded image cache & processed tables:** ~70 GB for 448×448 decoded image tensors in `data/processed/features/image/decoded448/`, plus ~7 GB of cached question states in `data/processed/features/text/`.
  - *Note:* Image feature representations are extracted dynamically and deleted via `vqa_lab.forget()` to prevent disk bloat.

---

## Notebooks

The research narrative and stage-by-stage workflows are structured across six Jupyter notebooks:

| Notebook | Focus | Key Actions |
|---|---|---|
| [`notebooks/00_data.ipynb`](notebooks/00_data.ipynb) | Data Pipeline | Downloads archives, unpacks COCO images and JSON files, converts static word vectors to KeyedVectors, compiles Parquet split tables, and generates the 448×448 decoded image cache. |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory Data Analysis | Inspects question types, answer distributions, annotator agreement, answer vocabulary coverage, and computes trivial baseline ceilings (always yes, question-type prior). |
| [`notebooks/02_first_model.ipynb`](notebooks/02_first_model.ipynb) | Baseline Pipeline | End-to-end training and evaluation of a classic baseline (ResNet-50 + GloVe, product fusion) in ~10 minutes. |
| [`notebooks/03_encoder_benchmark.ipynb`](notebooks/03_encoder_benchmark.ipynb) | Classical Benchmark Funnel | Evaluates 54 CNN encoders across funnel modes: initial screen (GloVe & BERT-base) $\to$ survivor evaluation (all 5 embeddings) $\to$ spatial attention head $\to$ reporting. |
| [`notebooks/04_modern_benchmark.ipynb`](notebooks/04_modern_benchmark.ipynb) | Modern Encoders & Token States | Evaluates 5 modern vision backbones against Qwen3 sentence vectors and token hidden states with a 3,129-answer vocabulary. |
| [`notebooks/05_vlm_baselines.ipynb`](notebooks/05_vlm_baselines.ipynb) | Zero-Shot VLM Baselines | Benchmarks Gemma 4 E4B and Qwen3.5-9B directly on unpruned `val2014` using vLLM in an isolated process. |

---

## Package Architecture (`src/vqa_lab`)

Twelve small modules in [`src/vqa_lab/`](src/vqa_lab/): the data pipeline (`data`), image and question encoders (`image`, `text`), the model and its training loop (`train`), the official metric (`metrics`), the experiment runner and the benchmark funnel (`experiment`, `benchmark`), the merged report (`report`) and the VLM baselines (`vlm`, `vlm_answer`). Module by module: [`src/vqa_lab/README.md`](src/vqa_lab/README.md).

---

## License

The code is released under the [Apache License 2.0](LICENSE). The answer normalisation and the accuracy in [`src/vqa_lab/metrics.py`](src/vqa_lab/metrics.py) are ported from the official VQA evaluation tools (BSD licence); their notice stays in that file's header. The VQA v2 data, the COCO images, the word vectors and the pretrained model weights are not part of this repository and keep their own licences; check each one before reuse.
