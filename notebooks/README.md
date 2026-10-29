# Notebooks

> What was tried and discarded before the benchmark: [How the recipe was chosen](#how-the-recipe-was-chosen).

Each notebook is one stage of the comparison; the process and the story live here, reusable code lives in `src/vqa_lab/`.
All numbers are VQA accuracy (%) on the **entire official val2014** (214,354 questions, nothing filtered) with the official
metric, using the image-disjoint holdout for early stopping. The protocol that produced them is the next section.

| Notebook | What it does |
|---|---|
| `00_data.ipynb` | Download and extract the archives, build the question tables and the decoded-image cache |
| `01_eda.ipynb` | Dataset sizes, answer/question types, vocabulary coverage, annotator agreement, trivial baselines and ceilings |
| `02_first_model.ipynb` | Train one classical H1 model (frozen CNN image vector x question vector, product fusion, 1,000 answers) and score it on the whole val2014 |
| `03_encoder_benchmark.ipynb` | The benchmark, one image encoder at a time: GloVe and BERT-base screen with the pooled product head, the other question embeddings for the survivors, attention for the top six; modes `funnel`, `survivors`, `attention`, `report` |
| `04_modern_benchmark.ipynb` | Modern encoder benchmark and token-state study: 5 modern image encoders x 4 question encoders (3 Qwen sentence-vector models + BERT-base as the control) x both heads, Qwen3-Embedding token states on 2 image encoders, 3,129-answer list, plus the bridge run; modes `run`, `report` |
| `05_vlm_baselines.ipynb` | Zero-shot vision-language models (Gemma 4 E4B, Qwen3.5-9B FP8) answering val2014 directly through vLLM in its own process, scored with the official metric; modes `smoke`, `pilot`, `full`, `report` |

The report cells of notebooks 03-05 merge every finished run, including the stopped, retried and failed ones, into [`benchmark_results.md`](benchmark_results.md) in this folder.

Reference points (official metric, val2014): always "yes" 24.04 · question-type prior 32.36 · best possible with a top-1000 answer
vocabulary 88.87 (the single-answer oracle is 96.38).

## Protocol

What produced every number below; the reasons for these choices are in the next section.

- **Data and metric.** VQA v2.0: train on train2014, report on all of val2014. The metric is the official VQA accuracy: `min(1, matches / 3)` against the ten human answers, averaged over the ten leave-one-out subsets, after the official answer normalisation (lower case, punctuation, number words, articles); it is also reported per answer type (yes/no, number, other).
- **Answer lists.** Classical benchmark: the 1,000 most frequent answers; only training questions whose majority answer is in the list are used, and evaluation keeps every question. Modern benchmark: the 3,129 most frequent normalised majority answers of the training rows. val2014 never builds a list.
- **Holdout.** Early stopping and every funnel decision use an image-disjoint holdout of the training images, never val2014. The published runs picked it by a sha1 hash of the image id (4.94 % of the images); see the reproducibility note under Results.
- **Model.** Frozen image encoders: features are extracted once, from images resized to 448x448 and then to each encoder's own input size. Question encoders: GloVe 840B, fastText wiki-news and word2vec GoogleNews word vectors, and BERT-base / BERT-large last-layer states (20 word pieces), through a 2-layer BiLSTM. Head: the question vector and the image vector are fused by an element-wise **product** (pooled image vector) or by question-guided **attention** over the 7x7 patch grid (2 glimpses, LayerNorm), then an MLP over the answer list. Soft-score binary cross-entropy, AdamW, gradient clipping at 1.0, bf16.
- **Fixed settings, no tuning.** Seed 123 for every cell, one run per (image encoder, question encoder, head). Product head: learning rate 8e-4, batch 2048. Attention head: learning rate 3e-4, batch 512. Warm-up 358,400 samples, at most 50 epochs, early stopping with patience 4 on the holdout.
- **Stalled starts.** A run whose holdout is still below 26 % after 3 epochs (always "yes" scores 24.4 on the holdout) is retrained with seed 123 + 101 and, if it stalls again, with the same seed at half the learning rate. The better attempt is reported; every attempt is logged.
- **Funnel.** (1) Screen: every encoder gets GloVe and BERT-base (product head), no stopping. (2) Survivors: the best third of the screened encoders by their better screen holdout (an encoder whose GloVe or BERT-base cell never learned is ranked by the one that did), plus Xception (the original backbone, labelled `kept` if it falls below the cut) and any encoder that already has all embeddings, get fastText, word2vec and BERT-large; the others are `stopped`. (3) Attention: the top six by best product holdout, plus Xception, with their best embedding, BERT-base and BERT-large; the three best stopped encoders get BERT-base as a safety net. A random-vector control runs once.
- **Modern benchmark.** The same protocol with the 3,129-answer list. The image side is the last-layer patch tokens (class and register tokens dropped) as a 2D grid: their mean for the product head, their adaptive average pool to 7x7 for the attention head, standardised per dimension on the fit images. A sentence-vector question encoder (each model's own last-token, L2-normalised vector) enters the head through Linear + tanh instead of the BiLSTM.
- **VLMs.** vLLM 0.32, greedy decoding, at most 10 new tokens, thinking off, seed 123, prefix caching on; the original COCO images go through each model's own processor (about 300 prompt tokens per question). The answer is the first non-empty line of the output, normalised with the official answer normalisation (needed because the official metric normalises a prediction only when the ten human answers differ, so a raw "Yes." would score 0 against ten "yes") and scored with the official metric.

**Where the numbers come from.** Every per-run result is a row of [`benchmark_results.md`](benchmark_results.md); the differences and means quoted in the text are arithmetic on those rows. The 95 % image-bootstrap intervals of the top runs (classical and modern) come from the report cells of notebooks 03 and 04. A few numbers are records of one-off computations that no notebook repeats: the paired token-state intervals, the VLM intervals and answer-format statistics (library bootstrap and normalised word counts), the seed check, the pilot results behind the recipe and the 4-bit weight check. All intervals use the library's default bootstrap seed 0; training used seed 123.

## How the recipe was chosen

Every cell of the benchmark uses one recipe. It was fixed in a pilot (17-18 October 2026: Xception, ResNet-152 and CLIP ResNet-50x4 with GloVe and BERT, seeds 0-2) before the benchmark ran. The pilot runs are not in the tables below; their numbers are quoted here only to explain the choices.

- **Standardised features, clipped gradients.** The first port of the classic head collapsed to "always yes" (24.04) because the product of raw CNN features and a question vector is tiny. Precision, learning rate, dropout and initialisation were ruled out first. Per-dimension standardisation of the image features (`train.feature_stats`) and gradient clipping at 1.0 fixed it.
- **Product and attention heads; concatenation dropped.** The product head beat concatenation in all four cells of the first fusion ladder. Product is the cheap screening head; attention over a 7x7 patch grid is kept for the best encoders.
- **LayerNorm per head.** Attention needs a LayerNorm at fusion to train at all; the product head loses about 2 points with it. So product runs without and attention with (`experiment.HEADS`).
- **Warm-up and epoch cap.** A 700-step warm-up (358,400 samples) came with the LayerNorm fix that made the first fusion ladder train; it was neutral for the product head. The 50-epoch cap replaced a 30-epoch cap that cost about 0.9 points (a 60-epoch run converged at epoch 47).
- **Batch sizes.** Product 2048 with the question states kept on the GPU (2.3x faster per epoch), attention 512 (batch 2048 at learning rate 6e-4 did not train).
- **Grid size.** A 14x14 grid gave attention no gain over 7x7 (55.11 against 55.17) at four times the storage, so the attention head reads 7x7.
- **Sanity controls** (GloVe + Xception pilot). Shuffling the val images drops 51.2 to 35.0, shuffling the val questions drops it to 11.2, a language-only bag-of-words model scores 42.1 and an image-only model about 24.4: the pipeline is sound, and the image is worth about 9 points over language alone.
- **Two bugs found on the way, each fixed with a regression test.** The random-vector control gave every word the same vector (about 25 %). The first "question only through the classic head" control could not train (the question vector varies by 0.0006 between questions at initialisation), so the bag-of-words model replaced it.

## Results

**Classical benchmark, seed 123.** 54 pretrained CNN image encoders x GloVe / fastText / word2vec / BERT-base / BERT-large question embeddings, classic head with a pooled **product** fusion or **attention** over a 7x7 patch grid, 1,000-answer list. All numbers are overall VQA accuracy % on the **entire official val2014** (official metric), one run per cell (seed 123, fixed learning rates). 195 runs were logged (69 needed the automatic restart after a stalled start). Full record, including stopped, retried and failed runs: [`benchmark_results.md`](benchmark_results.md).

*Reproducibility note:* The current code uses `scikit-learn` for the holdout split (`GroupShuffleSplit`) and the feature statistics (`StandardScaler`), and `sentence-transformers` for the Qwen3-Embedding sentence vectors, so a new training run lands within seed noise of the published runs instead of reproducing them bit for bit. The published results come from the original run tables and are unchanged. (`gensim` now also reads the GloVe and fastText files; their vectors are identical.)

### What the benchmark shows

1. **Pretraining data matters more than architecture.** The top of the table is filled by models pretrained on very large or noisy-label data (CLIP, ImageNet-21k / 12k). Plain ImageNet-1k classification encoders top out around 54-55 (best: NFNet-F0 54.75 with BERT-large, RegNetY-040 54.43 and SE-ResNeXt-101 54.05 with BERT-base), whatever their size or age.
2. **Best product-head result: 58.33** (CLIP ConvNeXt-Base with word2vec). **Best overall: 60.27** (CLIP ConvNeXt-Base, BERT-base, attention).
3. **BERT is not a clear winner over classic embeddings with the product head.** The best BERT beats the best static embedding on 12 of 20 fully evaluated encoders, by +0.63 on average (median +0.41); the winner depends on the encoder. BERT wins by 2.8-4.0 points on the CLIP ResNets (RN50, RN50x16, RN50x64); static vectors win on ConvNeXt, CLIP ConvNeXt and BiT-152x2. BERT-large is not reliably better than BERT-base.
4. **Attention needs BERT and a rich image encoder.** With BERT, attention improved the product result in 11 of 18 runs (+0.34 on average; +0.65 excluding the two Xception runs), with gains up to +2.9 on CLIP ConvNeXt-Base. With word2vec it lost in all 4 runs (-1.4 to -4.0). On older ImageNet-1k encoders (Xception, SE-ResNeXt-101, NFNet-F0, RegNetY-040, DenseNet-201) attention lost 0.7 to 2.6 even with BERT.
5. **Four GloVe cells never learned** (ConvNeXt-Tiny, ConvNeXt-Small, ResNeXt-101 WSL, AlexNet) and one BERT-large cell (ConvNeXt-V2-Base) even after the three-attempt rescue; they are listed in the *Discarded, retried and failed runs* section of [`benchmark_results.md`](benchmark_results.md) and excluded from every ranking.

### Screen: every encoder with GloVe and BERT-base (product head)

Label: `done` = survivor (best third of the screen, completed with all five embeddings); `stopped` = below the cut (only these two embeddings); `kept` = Xception, exempt from stopping. 33 encoders were stopped. Rank = the better of the two screen results; an encoder whose GloVe cell never learned (ConvNeXt-Tiny, ConvNeXt-Small, ResNeXt-101 WSL, AlexNet) is ranked by its BERT-base cell alone, and all four fell below the cut.

All 54 screened encoders with their GloVe and BERT-base runs (product head) are rows of the Classical benchmark table in [`benchmark_results.md`](benchmark_results.md) (columns `text` and `fusion`).

### Survivors: all five question embeddings (product head)

| image encoder | GloVe | fastText | word2vec | BERT-base | BERT-large | best BERT - best static |
| --- | --- | --- | --- | --- | --- | --- |
| convnext_base.clip_laion2b | 58.18 | 58.23 | 58.33 | 57.38 | 57.59 | -0.74 |
| resnet50x64_clip.openai | 53.38 | 53.14 | 53.15 | 56.89 | 57.25 | 3.87 |
| resnet50x16_clip.openai | 52.26 | 53.1 | 53.09 | 56.16 | 57.06 | 3.96 |
| tf_efficientnetv2_l.in21k_ft_in1k | 56.59 | 56.6 | 56.82 | 57.16 | 57.04 | 0.34 |
| regnety_160.sw_in12k_ft_in1k | 55.61 | 55.88 | 55.71 | 56.69 | 56.89 | 1.01 |
| efficientnet_b5.sw_in12k_ft_in1k | 55.53 | 54.86 | 55.57 | 56.02 | 56.89 | 1.31 |
| convnext_large.fb_in22k_ft_in1k | 55.82 | 56.03 | 56.46 | 55.69 | 56.15 | -0.31 |
| resnet50x4_clip.openai | 55.7 | 54.9 | 56.61 | 55.65 | 56.0 | -0.61 |
| resnet101_clip.openai | 55.53 | 53.48 | 56.0 | 55.62 | 55.56 | -0.37 |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | 54.95 | 52.2 | 55.41 | 54.43 | 55.36 | -0.05 |
| convnext_base.fb_in22k_ft_in1k | 56.67 | 56.47 | 56.88 | 55.52 | 55.23 | -1.35 |
| resnet50.fb_ssl_yfcc100m_ft_in1k | 53.64 | 54.31 | 54.38 | 54.77 | 55.11 | 0.73 |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | 54.78 | 54.79 | 55.6 | 54.68 | 54.94 | -0.66 |
| resnet50_clip.openai | 52.36 | 52.1 | 52.51 | 55.32 | 54.9 | 2.81 |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | 54.23 | 54.44 | 54.46 | 54.88 | 54.5 | 0.42 |
| tf_efficientnet_b7.ns_jft_in1k | 54.8 | 54.58 | 54.72 | 55.79 | 54.1 | 1.0 |
| dm_nfnet_f0.dm_in1k | 54.04 | 54.06 | 53.34 | 54.28 | 54.75 | 0.69 |
| resnet50.fb_swsl_ig1b_ft_in1k | 54.26 | 54.33 | 54.51 | 53.26 | 54.0 | -0.51 |
| resnet152.tv_in1k | 52.05 | 52.88 | 52.9 | 53.3 | 52.41 | 0.4 |
| legacy_xception.tf_in1k | 51.45 | 50.87 | 51.39 | 52.13 | 51.9 | 0.68 |

### Attention versus product (same encoder, same embedding)

Attention was run for the top six encoders (their best embedding plus BERT-base and BERT-large), for Xception, and as a safety net for the three best stopped encoders (BERT-base only).

| image encoder | embedding | product | attention | change |
| --- | --- | --- | --- | --- |
| convnext_base.clip_laion2b | bert | 57.38 | 60.27 | +2.89 |
| convnext_base.clip_laion2b | bert_large | 57.59 | 59.17 | +1.58 |
| convnext_base.clip_laion2b | word2vec | 58.33 | 55.70 | -2.64 |
| convnext_base.fb_in22k_ft_in1k | bert | 55.52 | 57.39 | +1.87 |
| convnext_base.fb_in22k_ft_in1k | bert_large | 55.23 | 55.01 | -0.22 |
| convnext_base.fb_in22k_ft_in1k | word2vec | 56.88 | 54.94 | -1.94 |
| convnextv2_base.fcmae_ft_in22k_in1k | bert | 55.22 | 56.99 | +1.78 |
| convnextv2_base.fcmae_ft_in22k_in1k | bert_large | n/a (never learned) | 57.51 |  |
| convnextv2_base.fcmae_ft_in22k_in1k | word2vec | 56.66 | 55.24 | -1.42 |
| densenet201.tv_in1k | bert | 53.73 | 52.42 | -1.31 |
| dm_nfnet_f0.dm_in1k | bert | 54.28 | 53.55 | -0.73 |
| legacy_xception.tf_in1k | bert | 52.13 | 49.93 | -2.21 |
| legacy_xception.tf_in1k | bert_large | 51.90 | 49.73 | -2.16 |
| regnety_040.ra3_in1k | bert | 54.43 | 53.27 | -1.16 |
| resnet50.fb_swsl_ig1b_ft_in1k | bert | 53.26 | 53.76 | +0.50 |
| resnet50x4_clip.openai | bert | 55.65 | 57.51 | +1.86 |
| resnet50x4_clip.openai | bert_large | 56.00 | 56.67 | +0.67 |
| resnet50x4_clip.openai | word2vec | 56.61 | 52.64 | -3.97 |
| resnet50x64_clip.openai | bert | 56.89 | 59.22 | +2.33 |
| resnet50x64_clip.openai | bert_large | 57.25 | 59.05 | +1.80 |
| seresnext101_32x8d.ah_in1k | bert | 54.05 | 51.45 | -2.60 |
| tf_efficientnetv2_l.in21k_ft_in1k | bert | 57.16 | 57.97 | +0.81 |
| tf_efficientnetv2_l.in21k_ft_in1k | bert_large | 57.04 | 57.44 | +0.40 |

### Overall top 10 (val %)

| rank | image encoder | question embedding | head | val % |
| --- | --- | --- | --- | --- |
| 1 | convnext_base.clip_laion2b | bert | attention | 60.27 |
| 2 | resnet50x64_clip.openai | bert | attention | 59.22 |
| 3 | convnext_base.clip_laion2b | bert_large | attention | 59.17 |
| 4 | resnet50x64_clip.openai | bert_large | attention | 59.05 |
| 5 | convnext_base.clip_laion2b | word2vec | product | 58.33 |
| 6 | convnext_base.clip_laion2b | fasttext | product | 58.23 |
| 7 | convnext_base.clip_laion2b | glove | product | 58.18 |
| 8 | tf_efficientnetv2_l.in21k_ft_in1k | bert | attention | 57.97 |
| 9 | convnext_base.clip_laion2b | bert_large | product | 57.59 |
| 10 | convnextv2_base.fcmae_ft_in22k_in1k | bert_large | attention | 57.51 |

### Caveats

- **One seed.** Differences below about 0.5 points are within run-to-run noise; there are no confidence intervals over seeds. The three best combinations were retrained with two more seeds (see the seed check below); the ranking holds. A first attempt that stalled at the "always yes" level for its first 3 epochs was retrained (another seed, then half the learning rate) and the attempt with the best holdout is reported; all attempts are logged (69 of 195 runs).
- **Fixed, untuned learning rates** (product 8e-4, attention 3e-4). The ConvNeXt family trained least reliably under them, so some of its low or missing cells reflect the recipe, not the encoder.
- **Screen decisions use the holdout, never val2014.** Encoders stopped after the screen have only GloVe and BERT-base scores, and only the three best of them were also tried with attention; an encoder that is mediocre with those two embeddings but excellent with another would have been missed.
- **Image features:** each encoder is applied at its native test resolution to images first resized (aspect ratio not kept) to 448x448; the 600-pixel models therefore see upsampled images.
- **The attention step compares each encoder with its own best static embedding**, chosen on the holdout, so its static columns are not a controlled comparison across encoders.
- **1,000-answer list**: numbers are not directly comparable with published VQA v2 results that use about 3,129 answers.

### Seed check of the three best combinations

The three best reported combinations were retrained with seeds 124 and 125 (an exception to the single-seed rule; same recipe and
rescue attempts). Overall VQA accuracy % on the whole val2014
(this table is the only record of the seed-check runs):

| image encoder | embedding | head | seed 123 | seed 124 | seed 125 | mean ± sd |
| --- | --- | --- | --- | --- | --- | --- |
| convnext_base.clip_laion2b | bert | attention | 60.27 | 60.37 | 60.20 | **60.28 ± 0.09** |
| convnext_base.clip_laion2b | bert_large | attention | 59.17 | 59.32 | 59.60 | 59.36 ± 0.22 |
| resnet50x64_clip.openai | bert | attention | 59.22 | 59.95 | 58.43 | 59.20 ± 0.76 |

**Reading.** The best result is stable: its worst seed (60.20) is above the best seed of the two others (59.95), so CLIP ConvNeXt-Base
+ BERT-base + attention is the winner whatever the seed. Run-to-run spread depends on the encoder: 0.1-0.2 points for CLIP ConvNeXt,
0.8 for CLIP RN50x64 (range 1.5) — consistent with the "about 0.5 points" caveat above, and a reason to read gaps under one point
between other cells as ties.


## Modern encoder benchmark

**Set-up.** Five modern frozen image encoders (Qwen3-VL-Embedding-2B vision tower, Perception Encoder PE-Lang-L, SigLIP 2 So400m,
C-RADIOv4-SO400M, CLIP ViT-L/14-336) x four question encoders (Qwen3-Embedding-0.6B and -4B and the text side of
Qwen3-VL-Embedding-2B as one L2-normalised sentence vector each; BERT-base in its classical token form as the **control**) x both
heads, with a **3,129-answer list**, plus one **bridge** run: the classical winner re-run with that list. Everything else is the
classical protocol: seed 123, one run per cell, fixed learning rates, early stopping on the image-disjoint holdout, official metric
on the entire val2014. 41 runs, all learned; 8 needed the stall restart (all product head); about 7 GPU hours in total, feature extraction
included. Notebook
[`04_modern_benchmark.ipynb`](04_modern_benchmark.ipynb); full record and merged tables [`benchmark_results.md`](benchmark_results.md). The 3,129-answer numbers are never mixed with the 1,000-answer tables
above; the bridge run links the two.

### What the modern benchmark shows

1. **The image encoder brings most of the gain.** With the question side fixed (BERT-base, attention), the modern encoders score
   67.17 (Qwen3-VL-Embedding), 64.10 (PE-Lang), 63.53 (SigLIP 2), 62.59 (C-RADIOv4) and 61.02 (CLIP ViT-L/14-336), against 60.69 for
   the classical winner with the same answer list: +0.33 to +6.48 points. The larger answer list alone adds only 0.42 (60.27 → 60.69).
2. **Best overall: 67.72** (Qwen3-VL-Embedding image + Qwen3-Embedding-4B + attention; yes/no 84.27, number 53.52, other 58.89). The
   Qwen3-VL-Embedding text side is a tie with it (67.63; paired image-bootstrap gap -0.09, 95 % interval -0.25 to +0.05).
3. **A better text encoder adds about one point, and size matters.** With attention, Qwen3-Embedding-4B beats BERT-base on all five
   image encoders (+0.55 to +2.15, mean +1.11). The small Qwen3-Embedding-0.6B loses to BERT on four of five (mean -0.87). The text side
   of Qwen3-VL-Embedding is level with BERT on average (-0.05), with one outlier: CLIP ViT-L/14-336 56.99 (-4.02 against BERT).
4. **With the product head, BERT's token states beat every sentence vector** on all five encoders (+0.35 to +2.91, mean +2.12): one
   vector per question loses the word-level detail that the BiLSTM over BERT tokens keeps. (See the epoch-cap caveat below.)
5. **Attention is essential for modern encoders.** Attention beats the product head by +4.5 to +12.8 points in 19 of 20 pairs (the
   exception, CLIP ViT-L/14-336 with the Qwen3-VL text side, by +1.35); mean +5.93 with BERT and +8.4 to +9.4 with the sentence vectors.
   In the classical benchmark attention added only +0.34 on average with BERT. The clearest case is Qwen3-VL-Embedding: with the product
   head (the mean of its tokens) its best result, 57.33, ranks fourth of five; with attention it is first with every question encoder.

Val % with the **attention** head (rows by best result):

| image encoder | Qwen3-Emb-0.6B | Qwen3-Emb-4B | Qwen3-VL-Emb text | BERT-base (control) |
| --- | --- | --- | --- | --- |
| qwen3vl_emb_2b | 66.29 | **67.72** | 67.63 | 67.17 |
| pe_lang_l | 63.13 | **65.16** | 65.08 | 64.10 |
| cradio_v4_so400m | 62.68 | **64.74** | 64.14 | 62.59 |
| siglip2_so400m | 62.37 | **64.58** | 64.29 | 63.53 |
| clip_l14_336 | 59.59 | **61.77** | 56.99 | 61.02 |

Val % with the **product** head:

| image encoder | Qwen3-Emb-0.6B | Qwen3-Emb-4B | Qwen3-VL-Emb text | BERT-base (control) |
| --- | --- | --- | --- | --- |
| qwen3vl_emb_2b | 54.64 | 54.90 | 54.86 | **57.33** |
| pe_lang_l | 53.27 | 56.11 | 55.64 | **58.67** |
| cradio_v4_so400m | 52.42 | 55.02 | 54.89 | **57.93** |
| siglip2_so400m | 53.33 | 55.97 | 55.28 | **58.33** |
| clip_l14_336 | 53.30 | 56.17 | 55.64 | **56.52** |

Bridge to the classical benchmark (attention head, BERT-base):

| run | answer list | val % |
| --- | --- | --- |
| classical winner, CLIP ConvNeXt-Base (seed 123; seeds 123-125: 60.28 ± 0.09) | 1,000 | 60.27 |
| the same model and recipe | 3,129 | 60.69 |
| best modern image encoder with the same BERT arm (Qwen3-VL-Embedding) | 3,129 | 67.17 |
| best modern combination (+ Qwen3-Embedding-4B) | 3,129 | 67.72 |

### Caveats (modern benchmark)

- **One seed.** The 95 % image-bootstrap intervals (about ±0.2) cover test-set sampling only, not training randomness; in the seed
  check, the spread over seeds was 0.1-0.8 points. Read gaps under about half a point as ties.
- **Epoch cap.** 18 of 20 product-head cells reached the 50-epoch cap while their holdout was still rising (+0.1 to +2.3 over the last
  10 epochs; the sentence-vector cells rose a little more than the BERT cells), so the product-head numbers are lower bounds. The
  attention cells stopped early (1 of 21 reached the cap). The cap is the classical one (88 of 172 classical product runs reached it).
- **Fixed, untuned recipe.** The learning rates and the Linear + tanh question branch for sentence vectors were not tuned; all 8
  restarts were product-head cells.
- **Frozen encoders, one vector per question.** No encoder is fine-tuned; the sentence-vector arms give the head no token sequence.
- **Images** come from the 448x448 decoded cache, resized to each encoder's input (C-RADIOv4 at 512 px sees upsampled images).
- **Gated models were not run** (DINOv3, EmbeddingGemma: licence click-through) and neither was SigLIP 2 Giant. They are not in `configs/vqa_lab.yaml`: every entry there was run. The 3,129-answer list is built from the training rows
  only, so it is close to, but not the same as, the list used in published VQA v2 work.

## Token-state study

**Question.** The modern benchmark gave each Qwen model to the head as **one sentence vector**, while BERT-base kept its **token
states + BiLSTM**; with the product head BERT beat every sentence vector. Is that because BERT's text features are better, or
because one vector loses information? Here the Qwen3-Embedding models (0.6B and 4B) get BERT's input form: the last-layer state of
every token (question + end token, 19 tokens, each dimension standardised on fit questions only) through the **unchanged BiLSTM**.
Same image features (a finished Qwen3-VL-Embedding cell re-run on the re-extracted features reproduced it to 6 decimals; PE-Lang
was re-extracted with the same code), same recipe, same
3,129-answer list, seed 123. 8 runs on Qwen3-VL-Embedding and PE-Lang, both heads; ~2.9 GPU hours plus ~20 min to encode the states.
Record: [`benchmark_results.md`](benchmark_results.md).

Val % on the whole val2014 (finished cells for comparison):

| image encoder | head | BERT-base tokens | Qwen3-Emb-0.6B vector | 0.6B tokens | Qwen3-Emb-4B vector | 4B tokens |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3vl_emb_2b | attention | 67.17 | 66.29 | 66.66 | **67.72** | 67.44 |
| pe_lang_l | attention | 64.10 | 63.13 | 64.18 | **65.16** | 64.28 |
| qwen3vl_emb_2b | product | **57.33** | 54.64 | 56.79 | 54.90 | 56.67 |
| pe_lang_l | product | **58.67** | 53.27 | 57.49 | 56.11 | 58.27 |

Paired differences in points with 95 % image-bootstrap intervals (test-set sampling only, not training noise):

| image encoder | head | 0.6B tokens - vector | 4B tokens - vector | 0.6B tokens - BERT | 4B tokens - BERT |
| --- | --- | --- | --- | --- | --- |
| qwen3vl_emb_2b | attention | +0.37 [+0.21, +0.54] | -0.28 [-0.44, -0.11] | -0.51 [-0.67, -0.36] | +0.28 [+0.12, +0.43] |
| pe_lang_l | attention | +1.05 [+0.87, +1.21] | -0.88 [-1.04, -0.71] | +0.08 [-0.08, +0.24] | +0.17 [+0.02, +0.35] |
| qwen3vl_emb_2b | product | +2.14 [+1.96, +2.32] | +1.77 [+1.60, +1.94] | -0.55 [-0.71, -0.39] | -0.66 [-0.82, -0.51] |
| pe_lang_l | product | +4.22 [+4.04, +4.40] | +2.16 [+1.99, +2.33] | -1.18 [-1.33, -1.01] | -0.40 [-0.53, -0.24] |

### What the token-state study shows

1. **With the product head, one vector does lose information.** The whole token sequence beats the same model's sentence vector
   in all four pairs, by +1.8 to +4.2 points. Most of BERT's product-head lead disappears (PE-Lang: 2.56 -> 0.40 for the 4B model).
2. **But BERT's tokens still win with the product head**, by 0.4 to 1.2 points in all four pairs: the remaining gap is the text
   model, not the input form.
3. **With attention, the 4B sentence vector stays the best question input.** The 4B tokens are below the 4B vector on both encoders
   (-0.28, -0.88); only the small 0.6B model gains from its tokens (+0.37, +1.05). The embedding model's final-token vector is what it
   was trained to produce; its other token states come from a causal model (each token sees only the words before it) and were never
   trained as outputs.
4. **LLM tokens vs BERT tokens, like for like: no clear winner.** With attention the 4B tokens are +0.28 and +0.17 ahead of BERT and
   the 0.6B tokens -0.51 and +0.08; with the product head BERT leads in every pair. All gaps are within about one point, and the
   attention gaps are within the 0.1-0.8 run-to-run spread measured in the classical seed check.
5. **The best modern result is unchanged:** Qwen3-VL-Embedding + Qwen3-Embedding-4B **sentence vector** + attention, 67.72.

### Caveats (token-state study)

- **Two image encoders, one seed.** The bootstrap intervals cover test-set sampling only; read gaps under about half a point as ties.
- **Epoch cap and restarts.** All four product-head token cells needed the stall restart and three reached the 50-epoch cap while still
  improving (the attention cells stopped early at 29-47 epochs), so the product-head token numbers are lower bounds, as for the other
  modern product cells.
- **Standardised Qwen states, raw BERT states.** Qwen's last-layer states are ~6x larger than BERT's with outlier dimensions, so they
  were standardised per dimension (fit statistics); BERT's arm is unchanged from the classical benchmark.
- **Causal vs bidirectional.** Each Qwen token sees only the tokens before it, each BERT token the whole question; the BiLSTM reads both
  directions in both arms.

## VLM zero-shot baselines

**Set-up.** Two open vision-language models answer every val2014 question directly, with no VQA training by us: **Gemma 4 E4B**
(bf16) and **Qwen3.5-9B** (Red Hat AI's FP8 version of the official model). Each sees the original COCO image and
`<question>\nAnswer the question using a single word or phrase.`, decodes greedily (thinking off) in vLLM 0.32 (in its own process), and its answer is normalised with the official answer normalisation and scored with the official metric, exactly like the
classifiers. A 2,000-question pilot came first. Record: [`benchmark_results.md`](benchmark_results.md); notebook
[`05_vlm_baselines.ipynb`](05_vlm_baselines.ipynb).

Val % on the whole val2014 (95 % image-bootstrap intervals):

| system | answers | val | yes/no | number | other |
| --- | --- | --- | --- | --- | --- |
| best classical classifier (CLIP ConvNeXt-Base + BERT-base, attention) | chosen from 1,000 | 60.27 [60.03, 60.48] | 77.94 | 42.46 | 51.55 |
| best modern classifier (Qwen3-VL-Embedding + Qwen3-Embedding-4B, attention) | chosen from 3,129 | 67.72 [67.51, 67.93] | 84.27 | 53.52 | 58.89 |
| **Gemma 4 E4B** (bf16) | free text | 57.98 [57.77, 58.19] | 70.81 | 53.52 | 49.39 |
| **Qwen3.5-9B** (FP8) | free text | **83.01** [82.85, 83.15] | 95.50 | 75.51 | 75.49 |

### What the VLM baselines show

1. **A strong current VLM is far ahead of frozen encoders + a trained head.** Qwen3.5-9B answers 83.01 % zero-shot, +15.29 points
   above the best modern classifier ([+15.10, +15.48]) and +22.7 above the best classical one, in every answer type; it scores better
   on 54,765 questions and worse on 25,025.
2. **Size and answer style matter as much as being a VLM.** Gemma 4 E4B scores 57.98, below both classifiers (-9.74 against the best
   modern one, -2.29 against the best classical one). Much of the gap is format: 20.6 % of its answers have 2-3 words and 3.0 % more
   than 3 (those score 3.8 on average), and 2.5 % are hedges ("uncertain", "cannot"), while Qwen answers in one word 93 % of the time.
   Its one-word answers score 71.0, Qwen's 84.9.
3. **Counting is where the VLM gains most:** Qwen3.5-9B's "number" accuracy is 75.51, against 53.52 for the best classifier.
4. **Speed on one RTX 4090 (vLLM):** Gemma 4 E4B 99 questions/s (36 minutes for val2014), Qwen3.5-9B 43 questions/s (83 minutes).

### Caveats (VLM baselines)

- **Not unseen data.** VQA v2 or COCO may be in the models' training mix; "zero-shot" means only that we trained nothing on VQA.
- **Free text vs a fixed answer list.** The classifiers choose from 1,000 / 3,129 answers (the 1,000-answer list caps the official
  accuracy at 88.87; the 3,129-answer list covers 92.5 % of val majority answers); the VLMs answer freely, scored after the official
  normalisation.
- **Precision.** Qwen3.5-9B runs as an FP8 quantisation (weights and activations), not the bf16 model.
- **vLLM is not bit-reproducible across batch / cache settings:** in the pilot, switching prefix caching off changed 49 of 2,000
  Gemma answers (accuracy 56.75 -> 56.71). Single run, greedy decoding, seed 123.
- **One prompt for every model, no prompt tuning.** Gemma 4 E4B would likely score higher with a prompt that forces one-word answers;
  the comparison keeps the common VQA v2 zero-shot prompt.
- **Gemma 4 26B-A4B (QAT 4-bit) was not run:** vLLM's GGUF plugin (0.0.5) cannot load the vision tower of Google's QAT GGUF, and the published vLLM repack matches the official 4-bit weights on 97.62 % of values (the rest one 4-bit step apart), so it is not the exact QAT model; two VLMs already answer the question.
