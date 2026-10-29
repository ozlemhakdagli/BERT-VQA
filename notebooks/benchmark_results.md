# Benchmark results

`val` is the official VQA accuracy % on the whole val2014 (the reported number); `holdout` is the accuracy on the image-disjoint holdout of train2014, which every funnel decision used. `epochs` are those of the reported attempt, `secs` the wall-clock time of the cell with every attempt, `retried` means the first attempt stalled and the better attempt is reported. Stopped, retried and failed runs are listed at the end.

## Headline

| system | answers | val | yes_no | number | other |
| --- | --- | --- | --- | --- | --- |
| bert + convnext_base.clip_laion2b (attention) | classical, 1,000-answer list | 60.27 | 77.94 | 42.46 | 51.55 |
| qwen3_emb_4b + qwen3vl_emb_2b (attention) | modern, 3,129-answer list | 67.72 | 84.27 | 53.52 | 58.89 |
| Gemma 4 E4B-it | free text | 57.98 | 70.81 | 53.52 | 49.39 |
| Qwen3.5-9B (RedHatAI FP8-dynamic) | free text | 83.01 | 95.5 | 75.51 | 75.49 |

## Classical benchmark

195 runs, 190 trained.

| backbone | family | text | fusion | status | epochs | holdout | val | yes_no | number | other | secs | retried |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_base.clip_laion2b | CLIP | bert | attention | done | 32 | 59.98 | 60.27 | 77.94 | 42.46 | 51.55 | 534.3 |  |
| resnet50x64_clip.openai | CLIP | bert | attention | done | 33 | 59.33 | 59.22 | 75.56 | 39.17 | 52.11 | 677.9 |  |
| convnext_base.clip_laion2b | CLIP | bert_large | attention | done | 32 | 58.93 | 59.17 | 76.34 | 41.32 | 50.83 | 749.3 |  |
| resnet50x64_clip.openai | CLIP | bert_large | attention | done | 25 | 59.37 | 59.05 | 75.73 | 39.97 | 51.42 | 1663.2 |  |
| convnext_base.clip_laion2b | CLIP | word2vec | product | done | 50 | 58.3 | 58.33 | 77.1 | 41.61 | 48.48 | 194.2 |  |
| convnext_base.clip_laion2b | CLIP | fasttext | product | done | 50 | 57.84 | 58.23 | 76.94 | 41.93 | 48.31 | 293.1 |  |
| convnext_base.clip_laion2b | CLIP | glove | product | done | 47 | 58.13 | 58.18 | 76.6 | 42.11 | 48.41 | 371.3 |  |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | bert | attention | done | 30 | 57.86 | 57.97 | 76.32 | 37.15 | 49.52 | 929.3 |  |
| convnext_base.clip_laion2b | CLIP | bert_large | product | done | 50 | 56.65 | 57.59 | 78.01 | 40.83 | 46.49 | 910.9 | yes |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | bert_large | attention | done | 33 | 57.52 | 57.51 | 75.26 | 34.67 | 50.07 | 643.0 |  |
| resnet50x4_clip.openai | CLIP | bert | attention | done | 27 | 57.44 | 57.51 | 75.16 | 37.37 | 49.42 | 460.0 |  |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | bert_large | attention | done | 32 | 57.46 | 57.44 | 75.6 | 36.78 | 49.1 | 663.2 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | bert | attention | done | 29 | 57.61 | 57.39 | 75.28 | 35.69 | 49.53 | 428.0 |  |
| convnext_base.clip_laion2b | CLIP | bert | product | done | 50 | 56.49 | 57.38 | 77.8 | 40.41 | 46.34 | 482.6 | yes |
| resnet50x64_clip.openai | CLIP | bert_large | product | done | 50 | 56.59 | 57.25 | 78.08 | 39.71 | 46.04 | 830.4 | yes |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | bert | product | done | 29 | 56.67 | 57.16 | 77.5 | 37.65 | 46.85 | 150.8 |  |
| resnet50x16_clip.openai | CLIP | bert_large | product | done | 50 | 56.34 | 57.06 | 78.14 | 39.67 | 45.62 | 815.3 | yes |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | bert_large | product | done | 33 | 56.62 | 57.04 | 77.25 | 36.45 | 47.11 | 185.6 |  |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | bert | attention | done | 29 | 56.76 | 56.99 | 73.89 | 35.79 | 49.77 | 426.9 |  |
| resnet50x64_clip.openai | CLIP | bert | product | done | 46 | 56.38 | 56.89 | 77.97 | 39.32 | 45.51 | 755.9 | yes |
| regnety_160.sw_in12k_ft_in1k | RegNet | bert_large | product | done | 40 | 56.14 | 56.89 | 77.56 | 35.78 | 46.76 | 226.6 |  |
| efficientnet_b5.sw_in12k_ft_in1k | EfficientNet | bert_large | product | done | 41 | 55.95 | 56.89 | 77.48 | 36.78 | 46.54 | 229.9 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | word2vec | product | done | 50 | 56.57 | 56.88 | 76.08 | 36.98 | 47.53 | 177.4 |  |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | word2vec | product | done | 50 | 56.39 | 56.82 | 75.58 | 37.19 | 47.75 | 177.7 |  |
| regnety_160.sw_in12k_ft_in1k | RegNet | bert | product | done | 37 | 56.32 | 56.69 | 77.81 | 34.69 | 46.44 | 193.8 |  |
| resnet50x4_clip.openai | CLIP | bert_large | attention | done | 29 | 56.85 | 56.67 | 73.82 | 37.2 | 48.79 | 658.4 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | glove | product | done | 50 | 56.39 | 56.67 | 75.8 | 37.11 | 47.29 | 396.2 |  |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | word2vec | product | done | 50 | 56.48 | 56.66 | 75.89 | 35.62 | 47.6 | 351.8 | yes |
| resnet50x4_clip.openai | CLIP | word2vec | product | done | 50 | 56.53 | 56.61 | 76.79 | 38.56 | 46.03 | 357.8 | yes |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | fasttext | product | done | 50 | 56.26 | 56.6 | 75.73 | 36.47 | 47.38 | 584.1 |  |
| tf_efficientnetv2_l.in21k_ft_in1k | EfficientNetV2 | glove | product | done | 50 | 56.44 | 56.59 | 75.53 | 36.71 | 47.45 | 704.7 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | fasttext | product | done | 50 | 56.46 | 56.47 | 74.93 | 37.65 | 47.41 | 338.4 | yes |
| convnext_large.fb_in22k_ft_in1k | ConvNeXt | word2vec | product | done | 50 | 56.15 | 56.46 | 75.25 | 37.07 | 47.3 | 178.1 |  |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | glove | product | done | 50 | 56.25 | 56.29 | 75.27 | 35.99 | 47.24 | 799.1 | yes |
| resnet50x16_clip.openai | CLIP | bert | product | done | 50 | 55.12 | 56.16 | 77.86 | 37.98 | 44.47 | 259.4 |  |
| convnext_large.fb_in22k_ft_in1k | ConvNeXt | bert_large | product | done | 50 | 55.47 | 56.15 | 76.65 | 36.22 | 45.84 | 806.4 | yes |
| convnext_large.fb_in22k_ft_in1k | ConvNeXt | fasttext | product | done | 50 | 56.07 | 56.03 | 74.54 | 35.97 | 47.26 | 391.7 |  |
| efficientnet_b5.sw_in12k_ft_in1k | EfficientNet | bert | product | done | 29 | 55.43 | 56.02 | 76.53 | 36.68 | 45.53 | 152.6 |  |
| resnet50x4_clip.openai | CLIP | bert_large | product | done | 50 | 55.2 | 56.0 | 77.8 | 37.9 | 44.21 | 567.3 | yes |
| resnet101_clip.openai | CLIP | word2vec | product | done | 50 | 55.94 | 56.0 | 75.92 | 38.53 | 45.46 | 178.8 |  |
| regnety_160.sw_in12k_ft_in1k | RegNet | fasttext | product | done | 50 | 55.85 | 55.88 | 75.44 | 35.42 | 46.42 | 327.4 |  |
| convnext_large.fb_in22k_ft_in1k | ConvNeXt | glove | product | done | 45 | 55.79 | 55.82 | 74.44 | 36.19 | 46.85 | 574.9 |  |
| tf_efficientnet_b7.ns_jft_in1k | EfficientNet | bert | product | done | 41 | 54.39 | 55.79 | 77.07 | 34.58 | 45.23 | 239.3 |  |
| regnety_160.sw_in12k_ft_in1k | RegNet | word2vec | product | done | 50 | 56.11 | 55.71 | 74.98 | 35.52 | 46.4 | 181.6 |  |
| resnet50x4_clip.openai | CLIP | glove | product | done | 50 | 55.56 | 55.7 | 75.22 | 37.86 | 45.58 | 511.6 | yes |
| convnext_base.clip_laion2b | CLIP | word2vec | attention | done | 41 | 55.91 | 55.7 | 70.1 | 39.15 | 49.12 | 460.6 |  |
| convnext_large.fb_in22k_ft_in1k | ConvNeXt | bert | product | done | 50 | 54.61 | 55.69 | 76.11 | 36.06 | 45.36 | 469.0 | yes |
| resnet50x4_clip.openai | CLIP | bert | product | done | 48 | 55.04 | 55.65 | 77.15 | 37.87 | 44.01 | 499.6 | yes |
| resnet101_clip.openai | CLIP | bert | product | done | 48 | 54.78 | 55.62 | 77.14 | 37.47 | 44.06 | 455.0 | yes |
| regnety_160.sw_in12k_ft_in1k | RegNet | glove | product | done | 50 | 55.55 | 55.61 | 74.88 | 35.92 | 46.18 | 415.9 |  |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | BiT | word2vec | product | done | 50 | 55.65 | 55.6 | 75.17 | 36.99 | 45.65 | 182.7 |  |
| efficientnet_b5.sw_in12k_ft_in1k | EfficientNet | word2vec | product | done | 50 | 55.25 | 55.57 | 74.95 | 35.71 | 46.09 | 178.6 |  |
| resnet101_clip.openai | CLIP | bert_large | product | done | 50 | 55.06 | 55.56 | 77.32 | 37.8 | 43.69 | 278.5 |  |
| resnet101_clip.openai | CLIP | glove | product | done | 50 | 55.24 | 55.53 | 75.23 | 38.0 | 45.18 | 389.8 | yes |
| efficientnet_b5.sw_in12k_ft_in1k | EfficientNet | glove | product | done | 50 | 55.3 | 55.53 | 74.75 | 35.8 | 46.13 | 446.7 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | bert | product | done | 48 | 55.06 | 55.52 | 75.49 | 35.98 | 45.51 | 306.6 | yes |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | BiT | word2vec | product | done | 50 | 55.37 | 55.41 | 74.97 | 36.3 | 45.58 | 309.5 | yes |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | BiT | bert_large | product | done | 50 | 54.46 | 55.36 | 76.25 | 36.02 | 44.58 | 787.1 | yes |
| resnet50_clip.openai | CLIP | bert | product | done | 50 | 54.33 | 55.32 | 77.58 | 36.76 | 43.3 | 495.8 | yes |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | word2vec | attention | done | 38 | 55.82 | 55.24 | 71.96 | 34.43 | 48.03 | 604.8 |  |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | bert_large | product | done | 50 | 54.7 | 55.23 | 74.83 | 35.66 | 45.5 | 453.5 | yes |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | bert | product | done | 50 | 54.19 | 55.22 | 74.99 | 34.95 | 45.54 | 496.6 | yes |
| resnet50.fb_ssl_yfcc100m_ft_in1k | WSL | bert_large | product | done | 42 | 54.22 | 55.11 | 76.17 | 35.64 | 44.24 | 505.6 | yes |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | bert_large | attention | done | 18 | 55.86 | 55.01 | 71.71 | 35.44 | 47.49 | 380.8 |  |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | BiT | glove | product | done | 50 | 54.63 | 54.95 | 74.03 | 36.18 | 45.41 | 532.6 |  |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | BiT | bert_large | product | done | 50 | 54.67 | 54.94 | 75.54 | 36.45 | 44.17 | 831.5 | yes |
| convnext_base.fb_in22k_ft_in1k | ConvNeXt | word2vec | attention | done | 29 | 55.44 | 54.94 | 71.41 | 34.31 | 47.87 | 466.7 |  |
| resnet50x4_clip.openai | CLIP | fasttext | product | done | 49 | 55.18 | 54.9 | 73.9 | 38.13 | 44.89 | 359.7 | yes |
| resnet50_clip.openai | CLIP | bert_large | product | done | 44 | 54.63 | 54.9 | 76.71 | 37.11 | 43.0 | 771.0 | yes |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | BiT | bert | product | done | 50 | 54.34 | 54.88 | 75.93 | 35.99 | 43.86 | 761.8 | yes |
| efficientnet_b5.sw_in12k_ft_in1k | EfficientNet | fasttext | product | done | 45 | 55.1 | 54.86 | 73.74 | 35.74 | 45.57 | 369.3 |  |
| tf_efficientnet_b7.ns_jft_in1k | EfficientNet | glove | product | done | 50 | 54.5 | 54.8 | 74.59 | 34.04 | 45.25 | 860.9 |  |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | BiT | fasttext | product | done | 50 | 55.03 | 54.79 | 73.65 | 35.85 | 45.46 | 1350.4 |  |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | BiT | glove | product | done | 50 | 54.83 | 54.78 | 73.57 | 36.01 | 45.45 | 1350.1 |  |
| resnet50.fb_ssl_yfcc100m_ft_in1k | WSL | bert | product | done | 33 | 54.35 | 54.77 | 75.58 | 35.35 | 44.09 | 172.0 |  |
| dm_nfnet_f0.dm_in1k | NFNet | bert_large | product | done | 37 | 54.45 | 54.75 | 76.17 | 34.57 | 43.79 | 298.6 |  |
| tf_efficientnet_b7.ns_jft_in1k | EfficientNet | word2vec | product | done | 50 | 54.27 | 54.72 | 74.08 | 34.7 | 45.29 | 180.2 |  |
| resnetv2_152x2_bit.goog_in21k_ft_in1k | BiT | bert | product | done | 50 | 54.55 | 54.68 | 75.76 | 35.81 | 43.65 | 778.3 | yes |
| tf_efficientnet_b7.ns_jft_in1k | EfficientNet | fasttext | product | done | 50 | 54.23 | 54.58 | 73.79 | 34.16 | 45.37 | 849.3 |  |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | word2vec | product | done | 50 | 54.13 | 54.51 | 74.48 | 35.43 | 44.37 | 170.5 |  |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | BiT | bert_large | product | done | 50 | 54.09 | 54.5 | 75.4 | 35.47 | 43.63 | 819.6 | yes |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | BiT | word2vec | product | done | 42 | 54.39 | 54.46 | 73.81 | 35.84 | 44.67 | 327.1 | yes |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | BiT | fasttext | product | done | 48 | 54.45 | 54.44 | 73.25 | 36.16 | 44.98 | 587.6 | yes |
| convnext_small.fb_in22k_ft_in1k | ConvNeXt | bert | product | stopped | 50 | 52.94 | 54.43 | 74.34 | 34.68 | 44.52 | 305.9 | yes |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | BiT | bert | product | done | 50 | 53.44 | 54.43 | 75.13 | 35.37 | 43.73 | 512.9 | yes |
| regnety_040.ra3_in1k | RegNet | bert | product | stopped | 35 | 53.76 | 54.43 | 75.72 | 33.86 | 43.67 | 181.1 |  |
| resnet50.fb_ssl_yfcc100m_ft_in1k | WSL | word2vec | product | done | 50 | 54.24 | 54.38 | 74.57 | 35.35 | 44.05 | 178.8 |  |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | fasttext | product | done | 50 | 54.1 | 54.33 | 73.85 | 35.83 | 44.39 | 415.3 |  |
| resnet50.fb_ssl_yfcc100m_ft_in1k | WSL | fasttext | product | done | 50 | 54.14 | 54.31 | 74.27 | 34.94 | 44.26 | 220.2 |  |
| dm_nfnet_f0.dm_in1k | NFNet | bert | product | done | 33 | 54.05 | 54.28 | 75.39 | 33.82 | 43.64 | 371.7 | yes |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | glove | product | done | 50 | 53.88 | 54.26 | 74.06 | 35.74 | 44.1 | 247.6 |  |
| resnetv2_50x1_bit.goog_in21k_ft_in1k | BiT | glove | product | done | 47 | 54.21 | 54.23 | 73.33 | 35.29 | 44.73 | 408.6 |  |
| tf_efficientnet_b7.ns_jft_in1k | EfficientNet | bert_large | product | done | 47 | 53.19 | 54.1 | 74.55 | 34.67 | 43.68 | 768.5 | yes |
| dm_nfnet_f0.dm_in1k | NFNet | fasttext | product | done | 50 | 54.36 | 54.06 | 74.11 | 34.92 | 43.88 | 278.9 |  |
| seresnext101_32x8d.ah_in1k | SENet | bert | product | stopped | 36 | 53.76 | 54.05 | 75.95 | 32.29 | 43.16 | 189.2 |  |
| dm_nfnet_f0.dm_in1k | NFNet | glove | product | done | 50 | 54.15 | 54.04 | 74.38 | 34.36 | 43.77 | 349.0 |  |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | bert_large | product | done | 50 | 53.38 | 54.0 | 74.71 | 36.11 | 42.97 | 790.4 | yes |
| densenet161.tv_in1k | DenseNet | bert | product | stopped | 29 | 53.21 | 53.95 | 75.51 | 35.28 | 42.48 | 153.0 |  |
| efficientnet_b3.ra2_in1k | EfficientNet | bert | product | stopped | 29 | 53.67 | 53.83 | 75.23 | 34.42 | 42.7 | 152.4 |  |
| densenet121.tv_in1k | DenseNet | bert | product | stopped | 36 | 53.08 | 53.79 | 75.8 | 34.48 | 42.15 | 187.4 |  |
| mobilenetv3_large_100.ra_in1k | MobileNet | bert | product | stopped | 33 | 53.34 | 53.78 | 75.64 | 34.44 | 42.27 | 171.7 |  |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | bert | attention | done | 23 | 53.71 | 53.76 | 73.06 | 33.85 | 44.35 | 545.3 |  |
| densenet201.tv_in1k | DenseNet | bert | product | stopped | 29 | 53.46 | 53.73 | 75.69 | 35.3 | 41.89 | 152.7 |  |
| resnext101_32x8d.tv_in1k | ResNeXt | bert | product | stopped | 42 | 52.69 | 53.71 | 75.62 | 34.42 | 42.15 | 222.7 |  |
| resnet50.fb_ssl_yfcc100m_ft_in1k | WSL | glove | product | done | 45 | 53.85 | 53.64 | 73.26 | 35.3 | 43.56 | 230.1 |  |
| dm_nfnet_f0.dm_in1k | NFNet | bert | attention | done | 27 | 53.7 | 53.55 | 72.44 | 31.95 | 44.89 | 771.3 |  |
| regnety_040.ra3_in1k | RegNet | glove | product | stopped | 50 | 53.49 | 53.54 | 73.67 | 33.7 | 43.49 | 268.2 |  |
| densenet201.tv_in1k | DenseNet | glove | product | stopped | 48 | 53.72 | 53.49 | 74.26 | 35.16 | 42.53 | 172.8 |  |
| resnet101_clip.openai | CLIP | fasttext | product | done | 50 | 53.15 | 53.48 | 72.71 | 38.02 | 42.94 | 448.6 | yes |
| senet154.gluon_in1k | SENet | bert | product | stopped | 29 | 53.23 | 53.46 | 75.08 | 34.1 | 42.14 | 156.7 |  |
| mobilenetv3_large_100.ra_in1k | MobileNet | glove | product | stopped | 47 | 53.55 | 53.41 | 74.36 | 34.45 | 42.5 | 186.9 |  |
| resnet50x64_clip.openai | CLIP | glove | product | done | 50 | 53.23 | 53.38 | 72.4 | 37.52 | 43.1 | 1274.1 | yes |
| densenet121.tv_in1k | DenseNet | glove | product | stopped | 46 | 52.82 | 53.35 | 74.39 | 35.25 | 42.13 | 163.7 |  |
| dm_nfnet_f0.dm_in1k | NFNet | word2vec | product | done | 50 | 53.38 | 53.34 | 73.0 | 34.0 | 43.5 | 189.7 |  |
| resnet152.tv_in1k | ResNet | bert | product | done | 36 | 52.62 | 53.3 | 75.69 | 33.11 | 41.62 | 267.4 |  |
| resnext101_32x8d.fb_wsl_ig1b_ft_in1k | WSL | bert | product | stopped | 50 | 51.76 | 53.29 | 73.72 | 33.98 | 42.86 | 508.3 | yes |
| convnext_tiny.fb_in1k | ConvNeXt | bert | product | stopped | 50 | 52.6 | 53.29 | 73.97 | 34.08 | 42.64 | 387.9 | yes |
| regnety_040.ra3_in1k | RegNet | bert | attention | done | 35 | 52.96 | 53.27 | 72.84 | 31.68 | 44.09 | 480.6 |  |
| resnet50.fb_swsl_ig1b_ft_in1k | WSL | bert | product | done | 48 | 52.53 | 53.26 | 74.06 | 34.96 | 42.27 | 500.4 | yes |
| senet154.gluon_in1k | SENet | glove | product | stopped | 50 | 53.31 | 53.18 | 73.87 | 33.68 | 42.6 | 185.4 |  |
| resnet50x64_clip.openai | CLIP | word2vec | product | done | 50 | 53.34 | 53.15 | 72.11 | 37.87 | 42.77 | 543.1 | yes |
| resnet50x64_clip.openai | CLIP | fasttext | product | done | 50 | 53.53 | 53.14 | 71.25 | 38.76 | 43.18 | 1377.4 | yes |
| resnet50x16_clip.openai | CLIP | fasttext | product | done | 50 | 53.18 | 53.1 | 72.24 | 38.43 | 42.42 | 696.0 | yes |
| resnet50x16_clip.openai | CLIP | word2vec | product | done | 50 | 53.22 | 53.09 | 71.78 | 38.42 | 42.74 | 539.3 | yes |
| seresnext101_32x8d.ah_in1k | SENet | glove | product | stopped | 48 | 53.05 | 53.06 | 73.05 | 33.52 | 43.02 | 176.8 |  |
| resnet101.tv_in1k | ResNet | bert | product | stopped | 33 | 52.58 | 52.97 | 75.41 | 34.36 | 40.82 | 176.5 |  |
| efficientnet_b0.ra_in1k | EfficientNet | bert | product | stopped | 29 | 52.75 | 52.91 | 74.53 | 34.46 | 41.35 | 152.1 |  |
| resnet152.tv_in1k | ResNet | word2vec | product | done | 50 | 52.46 | 52.9 | 73.68 | 34.76 | 41.89 | 178.6 |  |
| resnet152.tv_in1k | ResNet | fasttext | product | done | 50 | 52.64 | 52.88 | 73.65 | 34.84 | 41.85 | 178.8 |  |
| resnet50.tv_in1k | ResNet | bert | product | stopped | 35 | 52.19 | 52.66 | 75.24 | 34.3 | 40.35 | 183.9 |  |
| mobilenetv2_100.ra_in1k | MobileNet | bert | product | stopped | 29 | 52.23 | 52.65 | 74.99 | 34.75 | 40.4 | 152.1 |  |
| resnet50x4_clip.openai | CLIP | word2vec | attention | done | 30 | 52.72 | 52.64 | 68.67 | 33.88 | 45.41 | 629.6 |  |
| wide_resnet101_2.tv_in1k | ResNet | bert | product | stopped | 33 | 52.19 | 52.61 | 74.6 | 34.14 | 40.77 | 176.3 |  |
| tv_shufflenet_v2_x1_0 | ShuffleNet | bert | product | stopped | 29 | 51.82 | 52.6 | 75.07 | 34.6 | 40.27 | 150.5 |  |
| resnet101.tv_in1k | ResNet | glove | product | stopped | 50 | 52.24 | 52.59 | 73.63 | 34.56 | 41.35 | 182.7 |  |
| resnext50_32x4d.tv_in1k | ResNeXt | bert | product | stopped | 35 | 51.93 | 52.56 | 74.67 | 33.73 | 40.72 | 186.8 |  |
| wide_resnet50_2.tv_in1k | ResNet | bert | product | stopped | 33 | 52.17 | 52.52 | 74.53 | 33.71 | 40.75 | 176.4 |  |
| resnet50_clip.openai | CLIP | word2vec | product | done | 50 | 52.42 | 52.51 | 72.35 | 36.34 | 41.7 | 355.4 | yes |
| efficientnet_b3.ra2_in1k | EfficientNet | glove | product | stopped | 41 | 52.36 | 52.49 | 72.0 | 33.96 | 42.55 | 228.2 |  |
| densenet161.tv_in1k | DenseNet | glove | product | stopped | 37 | 52.8 | 52.46 | 72.8 | 33.81 | 41.92 | 251.2 |  |
| resnext101_32x16d.fb_wsl_ig1b_ft_in1k | WSL | bert | product | stopped | 50 | 51.45 | 52.44 | 72.21 | 32.91 | 42.57 | 257.6 |  |
| densenet201.tv_in1k | DenseNet | bert | attention | done | 25 | 52.5 | 52.42 | 72.14 | 33.11 | 42.54 | 376.8 |  |
| resnet152.tv_in1k | ResNet | bert_large | product | done | 33 | 52.7 | 52.41 | 75.02 | 33.51 | 40.2 | 235.9 |  |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | fasttext | product | done | 50 | 52.26 | 52.37 | 69.34 | 34.41 | 44.22 | 615.0 | yes |
| resnet50.tv_in1k | ResNet | glove | product | stopped | 50 | 52.15 | 52.37 | 73.39 | 35.09 | 40.94 | 180.2 |  |
| resnet50_clip.openai | CLIP | glove | product | done | 50 | 52.37 | 52.36 | 72.35 | 36.17 | 41.43 | 431.4 | yes |
| inception_resnet_v2.tf_in1k | Inception | bert | product | stopped | 35 | 51.3 | 52.32 | 74.07 | 33.29 | 40.81 | 183.4 |  |
| resnet50x16_clip.openai | CLIP | glove | product | done | 50 | 52.47 | 52.26 | 70.42 | 37.58 | 42.33 | 797.3 | yes |
| resnetv2_101x1_bit.goog_in21k_ft_in1k | BiT | fasttext | product | done | 50 | 51.94 | 52.2 | 70.7 | 35.73 | 42.49 | 885.7 | yes |
| efficientnet_b0.ra_in1k | EfficientNet | glove | product | stopped | 45 | 52.06 | 52.17 | 72.75 | 33.51 | 41.46 | 186.6 |  |
| legacy_xception.tf_in1k | Xception | bert | product | kept | 37 | 51.39 | 52.13 | 74.29 | 32.53 | 40.47 | 266.4 |  |
| resnet50_clip.openai | CLIP | fasttext | product | done | 50 | 52.33 | 52.1 | 71.86 | 35.51 | 41.46 | 451.4 | yes |
| resnet152.tv_in1k | ResNet | glove | product | done | 46 | 51.98 | 52.05 | 72.7 | 34.45 | 41.01 | 252.1 |  |
| wide_resnet101_2.tv_in1k | ResNet | glove | product | stopped | 48 | 51.98 | 51.95 | 72.75 | 34.23 | 40.82 | 176.7 |  |
| inception_v3.tv_in1k | Inception | bert | product | stopped | 37 | 51.0 | 51.94 | 74.33 | 32.74 | 39.99 | 193.7 |  |
| legacy_xception.tf_in1k | Xception | bert_large | product | done | 35 | 51.37 | 51.9 | 73.54 | 32.55 | 40.55 | 299.5 |  |
| resnext50_32x4d.tv_in1k | ResNeXt | glove | product | stopped | 50 | 51.98 | 51.83 | 72.64 | 33.45 | 40.87 | 183.6 |  |
| nasnetalarge.tf_in1k | NAS | bert | product | stopped | 33 | 51.17 | 51.79 | 73.28 | 32.59 | 40.52 | 174.4 |  |
| mobilenetv2_100.ra_in1k | MobileNet | glove | product | stopped | 42 | 51.42 | 51.58 | 72.33 | 34.27 | 40.37 | 173.3 |  |
| inception_v4.tf_in1k | Inception | glove | product | stopped | 50 | 51.24 | 51.56 | 72.13 | 33.45 | 40.71 | 179.1 |  |
| tv_googlenet | GoogLeNet | bert | product | stopped | 29 | 51.36 | 51.51 | 74.47 | 33.71 | 38.75 | 150.7 |  |
| inception_v3.tv_in1k | Inception | glove | product | stopped | 50 | 51.09 | 51.49 | 72.56 | 33.86 | 40.13 | 179.9 |  |
| legacy_xception.tf_in1k | Xception | glove | product | kept | 50 | 51.15 | 51.45 | 72.34 | 32.24 | 40.65 | 186.6 |  |
| seresnext101_32x8d.ah_in1k | SENet | bert | attention | done | 33 | 51.27 | 51.45 | 70.87 | 31.36 | 42.0 | 752.1 |  |
| resnet34.tv_in1k | ResNet | bert | product | stopped | 43 | 50.56 | 51.45 | 73.82 | 32.92 | 39.33 | 494.8 | yes |
| vgg19.tv_in1k | VGG | bert | product | stopped | 48 | 50.53 | 51.4 | 73.63 | 33.65 | 39.18 | 426.6 | yes |
| legacy_xception.tf_in1k | Xception | word2vec | product | done | 48 | 51.32 | 51.39 | 71.88 | 33.42 | 40.55 | 190.1 |  |
| resnext101_32x8d.tv_in1k | ResNeXt | glove | product | stopped | 40 | 51.13 | 51.3 | 71.14 | 33.72 | 40.85 | 148.0 |  |
| resnet18.tv_in1k | ResNet | bert | product | stopped | 49 | 50.15 | 51.15 | 73.55 | 32.81 | 38.97 | 466.8 | yes |
| wide_resnet50_2.tv_in1k | ResNet | glove | product | stopped | 41 | 50.64 | 51.07 | 71.55 | 33.82 | 40.05 | 151.5 |  |
| nasnetalarge.tf_in1k | NAS | glove | product | stopped | 45 | 50.7 | 51.07 | 70.95 | 33.47 | 40.59 | 434.7 |  |
| tv_shufflenet_v2_x1_0 | ShuffleNet | glove | product | stopped | 36 | 51.37 | 51.05 | 72.19 | 33.07 | 39.72 | 128.6 |  |
| inception_v4.tf_in1k | Inception | bert | product | stopped | 24 | 51.11 | 51.03 | 72.6 | 32.95 | 39.41 | 127.7 |  |
| legacy_xception.tf_in1k | Xception | fasttext | product | done | 41 | 51.34 | 50.87 | 71.42 | 33.06 | 39.96 | 152.4 |  |
| resnet152.tv_in1k | ResNet | random | product | done | 50 | 50.71 | 50.56 | 70.39 | 33.48 | 39.99 | 186.6 |  |
| pnasnet5large.tf_in1k | NAS | bert | product | stopped | 50 | 49.59 | 50.54 | 71.76 | 31.9 | 39.34 | 511.8 | yes |
| vgg16.tv_in1k | VGG | bert | product | stopped | 36 | 49.85 | 50.3 | 73.02 | 32.87 | 37.62 | 396.7 | yes |
| inception_resnet_v2.tf_in1k | Inception | glove | product | stopped | 35 | 49.95 | 50.19 | 70.08 | 32.55 | 39.72 | 127.4 |  |
| vgg16.tv_in1k | VGG | glove | product | stopped | 29 | 49.96 | 50.13 | 70.96 | 33.63 | 38.65 | 277.1 | yes |
| legacy_xception.tf_in1k | Xception | bert | attention | done | 29 | 49.57 | 49.93 | 70.56 | 31.8 | 39.03 | 661.1 |  |
| tv_googlenet | GoogLeNet | glove | product | stopped | 36 | 49.9 | 49.9 | 71.14 | 32.63 | 38.3 | 128.8 |  |
| legacy_xception.tf_in1k | Xception | bert_large | attention | done | 32 | 49.64 | 49.73 | 70.06 | 31.28 | 39.15 | 687.3 |  |
| pnasnet5large.tf_in1k | NAS | glove | product | stopped | 36 | 49.97 | 49.64 | 69.19 | 30.82 | 39.74 | 561.3 |  |
| tv_squeezenet1_1 | SqueezeNet | bert | product | stopped | 35 | 49.09 | 49.45 | 72.38 | 32.66 | 36.44 | 365.6 | yes |
| tv_alexnet | AlexNet | bert | product | stopped | 36 | 48.73 | 49.29 | 73.16 | 32.37 | 35.59 | 300.9 | yes |
| vgg19.tv_in1k | VGG | glove | product | stopped | 50 | 48.74 | 49.1 | 70.53 | 32.61 | 37.17 | 203.9 | yes |
| resnet18.tv_in1k | ResNet | glove | product | stopped | 50 | 48.78 | 48.85 | 69.23 | 32.86 | 37.58 | 239.3 | yes |
| resnet34.tv_in1k | ResNet | glove | product | stopped | 50 | 48.67 | 48.73 | 69.06 | 32.56 | 37.55 | 367.2 | yes |
| tv_squeezenet1_1 | SqueezeNet | glove | product | stopped | 50 | 48.3 | 48.51 | 69.59 | 33.38 | 36.47 | 351.4 | yes |
| resnext101_32x16d.fb_wsl_ig1b_ft_in1k | WSL | glove | product | stopped | 37 | 47.35 | 47.39 | 65.33 | 30.62 | 38.19 | 624.8 | yes |
| tv_alexnet | AlexNet | glove | product | untrained | 9 | 25.56 | 25.35 | 66.15 | 0.29 | 0.92 | 124.2 | yes |
| convnextv2_base.fcmae_ft_in22k_in1k | ConvNeXt-V2 | bert_large | product | untrained | 17 | 25.28 | 24.99 | 64.34 | 0.29 | 1.57 | 249.4 | yes |
| resnext101_32x8d.fb_wsl_ig1b_ft_in1k | WSL | glove | product | untrained | 19 | 25.16 | 24.92 | 64.16 | 0.3 | 1.56 | 234.6 | yes |
| convnext_small.fb_in22k_ft_in1k | ConvNeXt | glove | product | untrained | 13 | 25.14 | 24.83 | 63.09 | 0.26 | 2.21 | 194.1 | yes |
| convnext_tiny.fb_in1k | ConvNeXt | glove | product | untrained | 10 | 24.83 | 24.78 | 64.73 | 0.27 | 0.85 | 160.2 | yes |

## Modern encoder benchmark

49 runs, 49 trained.

| backbone | family | text | fusion | status | epochs | holdout | val | yes_no | number | other | secs | retried |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_4b | attention | done | 44 | 68.2 | 67.72 | 84.27 | 53.52 | 58.89 | 258.2 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3vl_emb_2b_text | attention | done | 38 | 67.79 | 67.63 | 84.62 | 53.02 | 58.58 | 221.5 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_4b_tokens | attention | done | 47 | 66.68 | 67.44 | 82.7 | 52.79 | 59.72 | 1376.6 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | bert | attention | done | 40 | 66.97 | 67.17 | 82.83 | 52.04 | 59.25 | 1522.3 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_0.6b_tokens | attention | done | 36 | 65.92 | 66.66 | 82.15 | 51.3 | 58.94 | 565.8 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_0.6b | attention | done | 50 | 66.65 | 66.29 | 82.92 | 51.64 | 57.51 | 1318.5 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_4b | attention | done | 35 | 65.44 | 65.16 | 82.75 | 48.11 | 56.3 | 143.6 |  |
| pe_lang_l | Perception Encoder | qwen3vl_emb_2b_text | attention | done | 33 | 65.26 | 65.08 | 82.47 | 48.19 | 56.32 | 133.6 |  |
| cradio_v4_so400m | C-RADIO | qwen3_emb_4b | attention | done | 38 | 64.95 | 64.74 | 81.45 | 49.56 | 56.05 | 163.9 |  |
| siglip2_so400m | SigLIP 2 | qwen3_emb_4b | attention | done | 35 | 64.86 | 64.58 | 81.84 | 48.93 | 55.6 | 919.5 |  |
| siglip2_so400m | SigLIP 2 | qwen3vl_emb_2b_text | attention | done | 37 | 64.78 | 64.29 | 81.21 | 48.29 | 55.65 | 909.9 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_4b_tokens | attention | done | 29 | 63.53 | 64.28 | 80.18 | 46.98 | 56.76 | 834.9 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_0.6b_tokens | attention | done | 29 | 63.55 | 64.18 | 80.38 | 46.88 | 56.45 | 1557.8 |  |
| cradio_v4_so400m | C-RADIO | qwen3vl_emb_2b_text | attention | done | 29 | 64.04 | 64.14 | 80.68 | 49.05 | 55.56 | 124.1 |  |
| pe_lang_l | Perception Encoder | bert | attention | done | 36 | 64.24 | 64.1 | 80.37 | 45.39 | 56.69 | 1568.5 |  |
| siglip2_so400m | SigLIP 2 | bert | attention | done | 42 | 63.59 | 63.53 | 78.62 | 47.48 | 56.29 | 1386.9 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_0.6b | attention | done | 37 | 63.37 | 63.13 | 80.08 | 46.51 | 54.64 | 1300.0 |  |
| cradio_v4_so400m | C-RADIO | qwen3_emb_0.6b | attention | done | 39 | 62.7 | 62.68 | 79.18 | 47.82 | 54.06 | 1274.5 |  |
| cradio_v4_so400m | C-RADIO | bert | attention | done | 31 | 62.54 | 62.59 | 77.51 | 46.26 | 55.57 | 1452.4 |  |
| siglip2_so400m | SigLIP 2 | qwen3_emb_0.6b | attention | done | 34 | 62.71 | 62.37 | 79.52 | 47.24 | 53.33 | 1014.8 |  |
| clip_l14_336 | CLIP ViT | qwen3_emb_4b | attention | done | 34 | 61.57 | 61.77 | 80.96 | 45.07 | 51.59 | 139.7 |  |
| clip_l14_336 | CLIP ViT | bert | attention | done | 38 | 60.37 | 61.02 | 78.59 | 43.16 | 52.38 | 951.4 |  |
| convnext_base.clip_laion2b | CLIP ConvNeXt (classical winner, bridge) | bert | attention | done | 35 | 60.54 | 60.69 | 78.05 | 41.69 | 52.52 | 582.6 |  |
| clip_l14_336 | CLIP ViT | qwen3_emb_0.6b | attention | done | 35 | 59.64 | 59.59 | 78.12 | 42.8 | 49.94 | 675.8 |  |
| pe_lang_l | Perception Encoder | bert | product | done | 50 | 57.6 | 58.67 | 78.87 | 40.66 | 48.06 | 808.9 | yes |
| siglip2_so400m | SigLIP 2 | bert | product | done | 47 | 58.26 | 58.33 | 78.0 | 41.96 | 47.7 | 795.2 | yes |
| pe_lang_l | Perception Encoder | qwen3_emb_4b_tokens | product | done | 50 | 56.31 | 58.27 | 78.24 | 41.0 | 47.65 | 2209.8 | yes |
| cradio_v4_so400m | C-RADIO | bert | product | done | 50 | 56.97 | 57.93 | 77.94 | 40.53 | 47.31 | 272.0 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_0.6b_tokens | product | done | 48 | 56.06 | 57.49 | 77.71 | 40.64 | 46.56 | 523.1 | yes |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | bert | product | done | 50 | 57.03 | 57.33 | 77.46 | 42.57 | 45.92 | 814.3 | yes |
| clip_l14_336 | CLIP ViT | qwen3vl_emb_2b_text | attention | done | 39 | 57.48 | 56.99 | 77.55 | 43.1 | 45.03 | 157.7 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_0.6b_tokens | product | done | 50 | 54.93 | 56.79 | 77.15 | 41.64 | 45.3 | 840.2 | yes |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_4b_tokens | product | done | 50 | 54.67 | 56.67 | 76.73 | 41.83 | 45.33 | 2364.4 | yes |
| clip_l14_336 | CLIP ViT | bert | product | done | 50 | 55.34 | 56.52 | 77.4 | 39.41 | 45.16 | 808.2 | yes |
| clip_l14_336 | CLIP ViT | qwen3_emb_4b | product | done | 50 | 56.4 | 56.17 | 77.79 | 39.02 | 44.26 | 34.5 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_4b | product | done | 50 | 56.11 | 56.11 | 76.92 | 40.0 | 44.54 | 39.1 |  |
| siglip2_so400m | SigLIP 2 | qwen3_emb_4b | product | done | 50 | 55.95 | 55.97 | 76.95 | 41.3 | 43.89 | 105.1 | yes |
| pe_lang_l | Perception Encoder | qwen3vl_emb_2b_text | product | done | 50 | 55.45 | 55.64 | 76.16 | 39.53 | 44.3 | 37.2 |  |
| clip_l14_336 | CLIP ViT | qwen3vl_emb_2b_text | product | done | 50 | 55.75 | 55.64 | 77.33 | 38.27 | 43.74 | 33.5 |  |
| siglip2_so400m | SigLIP 2 | qwen3vl_emb_2b_text | product | done | 50 | 55.28 | 55.28 | 76.0 | 39.79 | 43.62 | 37.7 |  |
| cradio_v4_so400m | C-RADIO | qwen3_emb_4b | product | done | 50 | 55.12 | 55.02 | 75.48 | 39.67 | 43.51 | 39.5 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_4b | product | done | 50 | 54.88 | 54.9 | 75.5 | 41.38 | 42.8 | 107.3 | yes |
| cradio_v4_so400m | C-RADIO | qwen3vl_emb_2b_text | product | done | 50 | 54.57 | 54.89 | 75.35 | 39.77 | 43.33 | 37.4 |  |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3vl_emb_2b_text | product | done | 50 | 54.94 | 54.86 | 75.55 | 40.53 | 42.92 | 103.1 | yes |
| qwen3vl_emb_2b | Qwen3-VL-Embedding | qwen3_emb_0.6b | product | done | 50 | 55.19 | 54.64 | 74.52 | 41.9 | 42.89 | 1090.5 | yes |
| siglip2_so400m | SigLIP 2 | qwen3_emb_0.6b | product | done | 50 | 53.52 | 53.33 | 73.12 | 39.72 | 41.87 | 902.4 |  |
| clip_l14_336 | CLIP ViT | qwen3_emb_0.6b | product | done | 39 | 53.86 | 53.3 | 74.28 | 37.44 | 41.54 | 550.8 |  |
| pe_lang_l | Perception Encoder | qwen3_emb_0.6b | product | done | 50 | 53.25 | 53.27 | 73.45 | 38.58 | 41.81 | 1181.3 |  |
| cradio_v4_so400m | C-RADIO | qwen3_emb_0.6b | product | done | 50 | 52.22 | 52.42 | 72.02 | 39.41 | 40.95 | 1135.8 |  |

## VLM zero-shot baselines

| model | name | precision | set | questions | overall | yes_no | number | other | empty | minutes | questions_per_s | prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4_e4b | Gemma 4 E4B-it | bf16 | full | 214354 | 57.98 | 70.81 | 53.52 | 49.39 | 0.0 | 36.17 | 98.77 | 292.1 |
| gemma4_e4b | Gemma 4 E4B-it | bf16 | n2000 | 2000 | 56.75 | 66.26 | 57.99 | 49.03 | 0.0 | 1.22 | 27.22 | 292.1 |
| gemma4_e4b | Gemma 4 E4B-it | bf16 | n2000_nocache | 2000 | 56.71 | 66.29 | 58.39 | 48.84 | 0.0 | 1.22 | 27.3 | 292.1 |
| gemma4_e4b | Gemma 4 E4B-it | bf16 | n50 | 50 | 44.2 | 68.67 | 40.0 | 31.85 | 0.0 | 0.05 | 16.47 | 291.9 |
| qwen35_9b_fp8 | Qwen3.5-9B (RedHatAI FP8-dynamic) | FP8 W8A8 | full | 214354 | 83.01 | 95.5 | 75.51 | 75.49 | 0.0 | 83.17 | 42.96 | 300.5 |
| qwen35_9b_fp8 | Qwen3.5-9B (RedHatAI FP8-dynamic) | FP8 W8A8 | n2000 | 2000 | 83.41 | 95.82 | 78.23 | 75.12 | 0.0 | 1.0 | 33.38 | 301.2 |
| qwen35_9b_fp8 | Qwen3.5-9B (RedHatAI FP8-dynamic) | FP8 W8A8 | n50 | 50 | 73.6 | 86.0 | 63.75 | 69.63 | 0.0 | 0.03 | 30.97 | 297.9 |

## Discarded, retried and failed runs

| benchmark | backbone | text | fusion | reason | detail |
| --- | --- | --- | --- | --- | --- |
| benchmark | convnext_small.fb_in22k_ft_in1k | bert | product | stopped | holdout 52.94 < 53.88, val 54.43 |
| benchmark | convnext_tiny.fb_in1k | bert | product | stopped | holdout 52.60 < 53.88, val 53.29 |
| benchmark | densenet121.tv_in1k | bert | product | stopped | holdout 53.08 < 53.88, val 53.79 |
| benchmark | densenet121.tv_in1k | glove | product | stopped | holdout 52.82 < 53.88, val 53.35 |
| benchmark | densenet161.tv_in1k | bert | product | stopped | holdout 53.21 < 53.88, val 53.95 |
| benchmark | densenet161.tv_in1k | glove | product | stopped | holdout 52.80 < 53.88, val 52.46 |
| benchmark | densenet201.tv_in1k | bert | product | stopped | holdout 53.46 < 53.88, val 53.73 |
| benchmark | densenet201.tv_in1k | glove | product | stopped | holdout 53.72 < 53.88, val 53.49 |
| benchmark | efficientnet_b0.ra_in1k | bert | product | stopped | holdout 52.75 < 53.88, val 52.91 |
| benchmark | efficientnet_b0.ra_in1k | glove | product | stopped | holdout 52.06 < 53.88, val 52.17 |
| benchmark | efficientnet_b3.ra2_in1k | bert | product | stopped | holdout 53.67 < 53.88, val 53.83 |
| benchmark | efficientnet_b3.ra2_in1k | glove | product | stopped | holdout 52.36 < 53.88, val 52.49 |
| benchmark | inception_resnet_v2.tf_in1k | bert | product | stopped | holdout 51.30 < 53.88, val 52.32 |
| benchmark | inception_resnet_v2.tf_in1k | glove | product | stopped | holdout 49.95 < 53.88, val 50.19 |
| benchmark | inception_v3.tv_in1k | bert | product | stopped | holdout 51.00 < 53.88, val 51.94 |
| benchmark | inception_v3.tv_in1k | glove | product | stopped | holdout 51.09 < 53.88, val 51.49 |
| benchmark | inception_v4.tf_in1k | bert | product | stopped | holdout 51.11 < 53.88, val 51.03 |
| benchmark | inception_v4.tf_in1k | glove | product | stopped | holdout 51.24 < 53.88, val 51.56 |
| benchmark | mobilenetv2_100.ra_in1k | bert | product | stopped | holdout 52.23 < 53.88, val 52.65 |
| benchmark | mobilenetv2_100.ra_in1k | glove | product | stopped | holdout 51.42 < 53.88, val 51.58 |
| benchmark | mobilenetv3_large_100.ra_in1k | bert | product | stopped | holdout 53.34 < 53.88, val 53.78 |
| benchmark | mobilenetv3_large_100.ra_in1k | glove | product | stopped | holdout 53.55 < 53.88, val 53.41 |
| benchmark | nasnetalarge.tf_in1k | bert | product | stopped | holdout 51.17 < 53.88, val 51.79 |
| benchmark | nasnetalarge.tf_in1k | glove | product | stopped | holdout 50.70 < 53.88, val 51.07 |
| benchmark | pnasnet5large.tf_in1k | bert | product | stopped | holdout 49.59 < 53.88, val 50.54 |
| benchmark | pnasnet5large.tf_in1k | glove | product | stopped | holdout 49.97 < 53.88, val 49.64 |
| benchmark | regnety_040.ra3_in1k | bert | product | stopped | holdout 53.76 < 53.88, val 54.43 |
| benchmark | regnety_040.ra3_in1k | glove | product | stopped | holdout 53.49 < 53.88, val 53.54 |
| benchmark | resnet101.tv_in1k | bert | product | stopped | holdout 52.58 < 53.88, val 52.97 |
| benchmark | resnet101.tv_in1k | glove | product | stopped | holdout 52.24 < 53.88, val 52.59 |
| benchmark | resnet18.tv_in1k | bert | product | stopped | holdout 50.15 < 53.88, val 51.15 |
| benchmark | resnet18.tv_in1k | glove | product | stopped | holdout 48.78 < 53.88, val 48.85 |
| benchmark | resnet34.tv_in1k | bert | product | stopped | holdout 50.56 < 53.88, val 51.45 |
| benchmark | resnet34.tv_in1k | glove | product | stopped | holdout 48.67 < 53.88, val 48.73 |
| benchmark | resnet50.tv_in1k | bert | product | stopped | holdout 52.19 < 53.88, val 52.66 |
| benchmark | resnet50.tv_in1k | glove | product | stopped | holdout 52.15 < 53.88, val 52.37 |
| benchmark | resnext101_32x16d.fb_wsl_ig1b_ft_in1k | bert | product | stopped | holdout 51.45 < 53.88, val 52.44 |
| benchmark | resnext101_32x16d.fb_wsl_ig1b_ft_in1k | glove | product | stopped | holdout 47.35 < 53.88, val 47.39 |
| benchmark | resnext101_32x8d.fb_wsl_ig1b_ft_in1k | bert | product | stopped | holdout 51.76 < 53.88, val 53.29 |
| benchmark | resnext101_32x8d.tv_in1k | bert | product | stopped | holdout 52.69 < 53.88, val 53.71 |
| benchmark | resnext101_32x8d.tv_in1k | glove | product | stopped | holdout 51.13 < 53.88, val 51.30 |
| benchmark | resnext50_32x4d.tv_in1k | bert | product | stopped | holdout 51.93 < 53.88, val 52.56 |
| benchmark | resnext50_32x4d.tv_in1k | glove | product | stopped | holdout 51.98 < 53.88, val 51.83 |
| benchmark | senet154.gluon_in1k | bert | product | stopped | holdout 53.23 < 53.88, val 53.46 |
| benchmark | senet154.gluon_in1k | glove | product | stopped | holdout 53.31 < 53.88, val 53.18 |
| benchmark | seresnext101_32x8d.ah_in1k | bert | product | stopped | holdout 53.76 < 53.88, val 54.05 |
| benchmark | seresnext101_32x8d.ah_in1k | glove | product | stopped | holdout 53.05 < 53.88, val 53.06 |
| benchmark | tv_alexnet | bert | product | stopped | holdout 48.73 < 53.88, val 49.29 |
| benchmark | tv_googlenet | bert | product | stopped | holdout 51.36 < 53.88, val 51.51 |
| benchmark | tv_googlenet | glove | product | stopped | holdout 49.90 < 53.88, val 49.90 |
| benchmark | tv_shufflenet_v2_x1_0 | bert | product | stopped | holdout 51.82 < 53.88, val 52.60 |
| benchmark | tv_shufflenet_v2_x1_0 | glove | product | stopped | holdout 51.37 < 53.88, val 51.05 |
| benchmark | tv_squeezenet1_1 | bert | product | stopped | holdout 49.09 < 53.88, val 49.45 |
| benchmark | tv_squeezenet1_1 | glove | product | stopped | holdout 48.30 < 53.88, val 48.51 |
| benchmark | vgg16.tv_in1k | bert | product | stopped | holdout 49.85 < 53.88, val 50.30 |
| benchmark | vgg16.tv_in1k | glove | product | stopped | holdout 49.96 < 53.88, val 50.13 |
| benchmark | vgg19.tv_in1k | bert | product | stopped | holdout 50.53 < 53.88, val 51.40 |
| benchmark | vgg19.tv_in1k | glove | product | stopped | holdout 48.74 < 53.88, val 49.10 |
| benchmark | wide_resnet101_2.tv_in1k | bert | product | stopped | holdout 52.19 < 53.88, val 52.61 |
| benchmark | wide_resnet101_2.tv_in1k | glove | product | stopped | holdout 51.98 < 53.88, val 51.95 |
| benchmark | wide_resnet50_2.tv_in1k | bert | product | stopped | holdout 52.17 < 53.88, val 52.52 |
| benchmark | wide_resnet50_2.tv_in1k | glove | product | stopped | holdout 50.64 < 53.88, val 51.07 |
| benchmark | convnext_base.clip_laion2b | bert | product | retried | kept seed 224 holdout 56.49; first tries seed 123 holdout 56.2167 |
| benchmark | convnext_base.clip_laion2b | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 56.65; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 56.62\|56.30 |
| benchmark | convnext_base.fb_in22k_ft_in1k | bert | product | retried | kept seed 224 holdout 55.06; first tries seed 123 holdout 40.7561 |
| benchmark | convnext_base.fb_in22k_ft_in1k | bert_large | product | retried | kept seed 224, lr 0.0008 holdout 54.70; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 25.06\|25.09 |
| benchmark | convnext_base.fb_in22k_ft_in1k | fasttext | product | retried | kept seed 224, lr 0.0008 holdout 56.46; first tries seed 123, lr 0.0008 holdout 32.08 |
| benchmark | convnext_large.fb_in22k_ft_in1k | bert | product | retried | kept seed 224 holdout 54.61; first tries seed 123 holdout 54.184 |
| benchmark | convnext_large.fb_in22k_ft_in1k | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 55.47; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 54.47\|54.84 |
| benchmark | convnext_small.fb_in22k_ft_in1k | bert | product | retried | kept seed 123 holdout 52.94; first tries seed 224 holdout 24.7283 |
| benchmark | convnext_tiny.fb_in1k | bert | product | retried | kept seed 123, lr 0.0004 holdout 52.60; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 25.20\|24.72 |
| benchmark | convnextv2_base.fcmae_ft_in22k_in1k | bert | product | retried | kept seed 123 holdout 54.19; first tries seed 224 holdout 54.0157 |
| benchmark | convnextv2_base.fcmae_ft_in22k_in1k | fasttext | product | retried | kept seed 224, lr 0.0008 holdout 52.26; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 46.75\|25.17 |
| benchmark | convnextv2_base.fcmae_ft_in22k_in1k | glove | product | retried | kept seed 224 holdout 56.25; first tries seed 123 holdout 51.8147 |
| benchmark | convnextv2_base.fcmae_ft_in22k_in1k | word2vec | product | retried | kept seed 224, lr 0.0008 holdout 56.48; first tries seed 123, lr 0.0008 holdout 53.31 |
| benchmark | dm_nfnet_f0.dm_in1k | bert | product | retried | kept seed 224 holdout 54.05; first tries seed 123 holdout 52.9594 |
| benchmark | pnasnet5large.tf_in1k | bert | product | retried | kept seed 123 holdout 49.59; first tries seed 224 holdout 48.969 |
| benchmark | resnet101_clip.openai | bert | product | retried | kept seed 224 holdout 54.78; first tries seed 123 holdout 54.1569 |
| benchmark | resnet101_clip.openai | fasttext | product | retried | kept seed 123, lr 0.0008 holdout 53.15; first tries seed 224\|123, lr 0.0008\|0.0004 holdout 52.15\|25.09 |
| benchmark | resnet101_clip.openai | glove | product | retried | kept seed 224 holdout 55.24; first tries seed 123 holdout 49.6222 |
| benchmark | resnet18.tv_in1k | bert | product | retried | kept seed 224 holdout 50.15; first tries seed 123 holdout 49.6554 |
| benchmark | resnet18.tv_in1k | glove | product | retried | kept seed 123 holdout 48.78; first tries seed 224 holdout 25.2114 |
| benchmark | resnet34.tv_in1k | bert | product | retried | kept seed 123 holdout 50.56; first tries seed 224 holdout 50.1884 |
| benchmark | resnet34.tv_in1k | glove | product | retried | kept seed 224 holdout 48.67; first tries seed 123 holdout 48.1731 |
| benchmark | resnet50.fb_ssl_yfcc100m_ft_in1k | bert_large | product | retried | kept seed 224, lr 0.0008 holdout 54.22; first tries seed 123, lr 0.0008 holdout 52.97 |
| benchmark | resnet50.fb_swsl_ig1b_ft_in1k | bert | product | retried | kept seed 224 holdout 52.53; first tries seed 123 holdout 52.1416 |
| benchmark | resnet50.fb_swsl_ig1b_ft_in1k | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 53.38; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 52.61\|52.70 |
| benchmark | resnet50_clip.openai | bert | product | retried | kept seed 123 holdout 54.33; first tries seed 224 holdout 54.119 |
| benchmark | resnet50_clip.openai | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 54.63; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 54.17\|53.89 |
| benchmark | resnet50_clip.openai | fasttext | product | retried | kept seed 123, lr 0.0008 holdout 52.33; first tries seed 224\|123, lr 0.0008\|0.0004 holdout 51.61\|25.14 |
| benchmark | resnet50_clip.openai | glove | product | retried | kept seed 224 holdout 52.37; first tries seed 123 holdout 52.064 |
| benchmark | resnet50_clip.openai | word2vec | product | retried | kept seed 224, lr 0.0008 holdout 52.42; first tries seed 123, lr 0.0008 holdout 51.51 |
| benchmark | resnet50x16_clip.openai | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 56.34; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 55.02\|54.98 |
| benchmark | resnet50x16_clip.openai | fasttext | product | retried | kept seed 123, lr 0.0008 holdout 53.18; first tries seed 224\|123, lr 0.0008\|0.0004 holdout 52.32\|24.71 |
| benchmark | resnet50x16_clip.openai | glove | product | retried | kept seed 123 holdout 52.47; first tries seed 224 holdout 51.1377 |
| benchmark | resnet50x16_clip.openai | word2vec | product | retried | kept seed 123, lr 0.0008 holdout 53.22; first tries seed 224\|123, lr 0.0008\|0.0004 holdout 52.43\|51.16 |
| benchmark | resnet50x4_clip.openai | bert | product | retried | kept seed 123 holdout 55.04; first tries seed 224 holdout 54.6348 |
| benchmark | resnet50x4_clip.openai | bert_large | product | retried | kept seed 123 holdout 55.20; first tries seed 224 holdout 54.7756 |
| benchmark | resnet50x4_clip.openai | fasttext | product | retried | kept seed 224 holdout 55.18; first tries seed 123 holdout 52.9753 |
| benchmark | resnet50x4_clip.openai | glove | product | retried | kept seed 224 holdout 55.56; first tries seed 123 holdout 53.0263 |
| benchmark | resnet50x4_clip.openai | word2vec | product | retried | kept seed 224 holdout 56.53; first tries seed 123 holdout 52.7541 |
| benchmark | resnet50x64_clip.openai | bert | product | retried | kept seed 123, lr 0.0004 holdout 56.38; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 55.33\|55.32 |
| benchmark | resnet50x64_clip.openai | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 56.59; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 55.04\|54.81 |
| benchmark | resnet50x64_clip.openai | fasttext | product | retried | kept seed 123, lr 0.0004 holdout 53.53; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 53.04\|52.40 |
| benchmark | resnet50x64_clip.openai | glove | product | retried | kept seed 224, lr 0.0008 holdout 53.23; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 52.48\|25.44 |
| benchmark | resnet50x64_clip.openai | word2vec | product | retried | kept seed 224, lr 0.0008 holdout 53.34; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 52.58\|52.17 |
| benchmark | resnetv2_101x1_bit.goog_in21k_ft_in1k | bert | product | retried | kept seed 123, lr 0.0008 holdout 53.44; first tries seed 224, lr 0.0008 holdout 53.41 |
| benchmark | resnetv2_101x1_bit.goog_in21k_ft_in1k | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 54.46; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 52.90\|52.08 |
| benchmark | resnetv2_101x1_bit.goog_in21k_ft_in1k | fasttext | product | retried | kept seed 123, lr 0.0008 holdout 51.94; first tries seed 224\|123, lr 0.0008\|0.0004 holdout 51.47\|50.03 |
| benchmark | resnetv2_101x1_bit.goog_in21k_ft_in1k | word2vec | product | retried | kept seed 224, lr 0.0008 holdout 55.37; first tries seed 123, lr 0.0008 holdout 49.49 |
| benchmark | resnetv2_152x2_bit.goog_in21k_ft_in1k | bert | product | retried | kept seed 123, lr 0.0004 holdout 54.55; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 53.18\|53.63 |
| benchmark | resnetv2_152x2_bit.goog_in21k_ft_in1k | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 54.67; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 52.88\|52.37 |
| benchmark | resnetv2_50x1_bit.goog_in21k_ft_in1k | bert | product | retried | kept seed 123, lr 0.0004 holdout 54.34; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 53.02\|53.10 |
| benchmark | resnetv2_50x1_bit.goog_in21k_ft_in1k | bert_large | product | retried | kept seed 123, lr 0.0004 holdout 54.09; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 52.92\|52.64 |
| benchmark | resnetv2_50x1_bit.goog_in21k_ft_in1k | fasttext | product | retried | kept seed 224, lr 0.0008 holdout 54.45; first tries seed 123, lr 0.0008 holdout 51.21 |
| benchmark | resnetv2_50x1_bit.goog_in21k_ft_in1k | word2vec | product | retried | kept seed 224, lr 0.0008 holdout 54.39; first tries seed 123, lr 0.0008 holdout 51.45 |
| benchmark | resnext101_32x16d.fb_wsl_ig1b_ft_in1k | glove | product | retried | kept seed 123 holdout 47.35; first tries seed 224 holdout 45.5363 |
| benchmark | resnext101_32x8d.fb_wsl_ig1b_ft_in1k | bert | product | retried | kept seed 123 holdout 51.76; first tries seed 224 holdout 51.515 |
| benchmark | tf_efficientnet_b7.ns_jft_in1k | bert_large | product | retried | kept seed 224, lr 0.0008 holdout 53.19; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 52.52\|52.91 |
| benchmark | tv_alexnet | bert | product | retried | kept seed 123, lr 0.0004 holdout 48.73; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 25.39\|25.51 |
| benchmark | tv_squeezenet1_1 | bert | product | retried | kept seed 224 holdout 49.09; first tries seed 123 holdout 48.9825 |
| benchmark | tv_squeezenet1_1 | glove | product | retried | kept seed 123 holdout 48.30; first tries seed 224 holdout 48.1474 |
| benchmark | vgg16.tv_in1k | bert | product | retried | kept seed 123 holdout 49.85; first tries seed 224 holdout 49.7569 |
| benchmark | vgg16.tv_in1k | glove | product | retried | kept seed 224 holdout 49.96; first tries seed 123 holdout 49.1382 |
| benchmark | vgg19.tv_in1k | bert | product | retried | kept seed 224 holdout 50.53; first tries seed 123 holdout 49.7849 |
| benchmark | vgg19.tv_in1k | glove | product | retried | kept seed 224 holdout 48.74; first tries seed 123 holdout 25.2516 |
| benchmark | convnext_small.fb_in22k_ft_in1k | glove | product | never learned | best try 25.14, other tries 24.40\|24.86 |
| benchmark | convnext_tiny.fb_in1k | glove | product | never learned | best try 24.83, other tries 24.70\|24.57 |
| benchmark | convnextv2_base.fcmae_ft_in22k_in1k | bert_large | product | never learned | best try 25.28, other tries 24.83\|24.97 |
| benchmark | resnext101_32x8d.fb_wsl_ig1b_ft_in1k | glove | product | never learned | best try 25.16, other tries 25.10\|24.40 |
| benchmark | tv_alexnet | glove | product | never learned | best try 25.56, other tries 25.37\|25.22 |
| benchmark | tf_efficientnet_b7.ns_jft_in1k |  |  | failed | CUDA out of memory. Tried to allocate 8.24 GiB. GPU 0 has a total capacity of 23.52 GiB of which 5.6 |
| benchmark | resnetv2_50x1_bit.goog_in21k_ft_in1k |  |  | failed | Got: ConnectError: [Errno -3] Temporary failure in name resolution |
| benchmark | resnetv2_101x1_bit.goog_in21k_ft_in1k |  |  | failed | Got: ConnectError: [Errno -3] Temporary failure in name resolution |
| benchmark | resnetv2_152x2_bit.goog_in21k_ft_in1k |  |  | failed | ResNetV2.load_pretrained() expected an original JAX/Flax NumPy .npz checkpoint, but could not read ' |
| benchmark | nasnetalarge.tf_in1k |  |  | failed | CUDA out of memory. Tried to allocate 436.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 2 |
| benchmark | resnet50x64_clip.openai |  |  | failed | CUDA out of memory. Tried to allocate 3.06 GiB. GPU 0 has a total capacity of 23.52 GiB of which 2.0 |
| modern | clip_l14_336 | bert | product | retried | kept seed 224, lr 0.0008 holdout 55.34; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 53.94\|54.85 |
| modern | pe_lang_l | bert | product | retried | kept seed 123, lr 0.0004 holdout 57.60; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 56.74\|56.92 |
| modern | pe_lang_l | qwen3_emb_0.6b_tokens | product | retried | kept seed 224, lr 0.0008 holdout 56.06; first tries seed 123, lr 0.0008 holdout 55.43 |
| modern | pe_lang_l | qwen3_emb_4b_tokens | product | retried | kept seed 123, lr 0.0004 holdout 56.31; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 56.29\|55.80 |
| modern | qwen3vl_emb_2b | bert | product | retried | kept seed 123, lr 0.0004 holdout 57.03; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 54.89\|54.87 |
| modern | qwen3vl_emb_2b | qwen3_emb_0.6b | product | retried | kept seed 224, lr 0.0008 holdout 55.19; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 52.39\|51.92 |
| modern | qwen3vl_emb_2b | qwen3_emb_0.6b_tokens | product | retried | kept seed 123, lr 0.0004 holdout 54.93; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 53.83\|53.35 |
| modern | qwen3vl_emb_2b | qwen3_emb_4b | product | retried | kept seed 123, lr 0.0004 holdout 54.88; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 54.43\|54.42 |
| modern | qwen3vl_emb_2b | qwen3_emb_4b_tokens | product | retried | kept seed 123, lr 0.0004 holdout 54.67; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 53.29\|53.47 |
| modern | qwen3vl_emb_2b | qwen3vl_emb_2b_text | product | retried | kept seed 224, lr 0.0008 holdout 54.94; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 54.25\|54.02 |
| modern | siglip2_so400m | bert | product | retried | kept seed 123, lr 0.0004 holdout 58.26; first tries seed 123\|224, lr 0.0008\|0.0008 holdout 56.76\|57.10 |
| modern | siglip2_so400m | qwen3_emb_4b | product | retried | kept seed 224, lr 0.0008 holdout 55.95; first tries seed 123\|123, lr 0.0008\|0.0004 holdout 55.15\|55.48 |
| modern | siglip2_so400m |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 6. |
| modern | pe_lang_l |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 6. |
| modern | cradio_v4_so400m |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 5. |
| modern | clip_l14_336 |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 5. |
| modern | qwen3vl_emb_2b |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 5. |
| modern | convnext_base.clip_laion2b |  |  | failed | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 0 has a total capacity of 23.52 GiB of which 5. |
