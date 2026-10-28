# Research-paper methodology template

_Replace every `<...>` placeholder with values taken from your result files. Every number in the paper should be traceable to a `result.json`, `run.json` or `reports/<name>/comparison.csv`._

## 1. Problem formulation
Binary classification of an input image x into REAL (0) or FAKE (1). Two independent tracks, each with its own models, data and evaluation protocol: (i) AI-generated image detection and (ii) facial manipulation detection. The classifier outputs logits z ∈ ℝ²; the fake score is s = softmax(z)₁. The predicted label is FAKE iff s ≥ τ, where τ is selected on the validation split by maximizing Youden's J (equivalently, balanced accuracy).

## 2. Datasets
- **Track 1:** GenImage `<subset description>`, balanced to `<N>` images per class per generator (`ml.data.genimage`, seed `<s>`). GenImage's official val split serves as our test split; its train split is divided `<90/10>` into train/val.
- **Track 2:** FaceForensics++ `<compression>`, `<k>` frames per video, face crops with a margin of 1.3 (`ml.data.faceforensics`). Splits follow `<the official FF++ split | grouped split>`. Groups are connected components of target/source video ids, so no identity appears in two splits.
- **Data validation:** exact (SHA-256) and perceptual (dHash, Hamming ≤ `<h>`) duplicates across splits were `<removed/none found>` (`reports/data_validation/<...>.json`).
- Report class counts per split and source (from `*.summary.json`).

## 3. Models
ResNet-50 and EfficientNet-B0 initialized from torchvision ImageNet weights, with the classifier replaced by Dropout(p) followed by Linear(d, 2). Frequency variant: a 4-layer CNN over the standardized centered log-magnitude 2D FFT of the luminance channel, computed inside the network (so the preprocessing is identical to the baseline). Its 128-dim embedding is concatenated with the backbone features.

## 4. Training
AdamW (lr `<>`, weight decay `<>`), batch size `<>`, `<E>` epochs max, linear warm-up followed by cosine decay, gradient clipping 1.0, mixed precision (float16) on `<GPU>`, early stopping on validation ROC-AUC (patience `<>`). Augmentation: `<...>`. Seeds: `<42, 43, 44>`. Software: PyTorch `<version>`, CUDA `<version>`.

## 5. Evaluation protocol
- Metrics: accuracy, per-class precision/recall, macro-F1, ROC-AUC, confusion matrix; latency at batch size 1.
- 95% CIs: grouped percentile bootstrap (`<1000>` resamples over leakage groups).
- Multiple seeds: report mean ± sample standard deviation.
- Experiment 2: leave-one-generator-out; the held-out generator is absent from train and val.
- Experiment 3: leave-one-manipulation-out on FF++.
- Experiment 4: JPEG quality 90/70/50, downscaling ×0.5/×0.25 (restored to the original size), Gaussian blur σ = 1/2, Gaussian noise σ = 5/15; the threshold is kept fixed.
- Experiments 5/7: RGB vs RGB+FFT under identical splits, seeds and hyperparameters.
- Experiment 6: Grad-CAM (last convolutional block) on randomly sampled TP/TN/FP/FN test images, presented as qualitative only.

## 6. Results (to be filled from files)
Table 1: baseline comparison (`reports/exp1_baseline_aigen__test/comparison.tex`). Table 2: cross-generator. Table 3: cross-manipulation. Figure: robustness curves (`robustness.png`). Table 4: ablation (`reports/exp5_7_frequency_ablation__test/comparison.tex`).

## 7. Threats to validity
Dataset shortcuts (format/size/compression differences between classes), limited generator diversity, frame correlation, threshold transfer across domains, and single-run variance. See docs/LIMITATIONS.md.
