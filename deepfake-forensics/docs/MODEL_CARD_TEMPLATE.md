# Model card: `<model_id>`

_Fill every field from the checkpoint sidecar (`artifacts/checkpoints/<track>/<model_id>.json`), `run.json` and evaluation `result.json` files. Do not enter numbers that are not in those files._

## Model details
- **Track:** ai_generated / face_deepfake
- **Architecture:** ResNet-50 / EfficientNet-B0 (+ Fourier branch: yes/no)
- **Initialization:** e.g. torchvision ResNet50_Weights.IMAGENET1K_V2
- **Version:** `<run_id>@e<epoch>` · **Created:** `<created_at>`
- **Preprocessing:** img_size, resize_mode, ImageNet mean/std
- **Output:** logits [REAL, FAKE]; fake score = softmax[FAKE] (uncalibrated)
- **Decision threshold:** value, method, selected on `<validation split>`

## Intended use
Research and education: estimating whether an image of the relevant kind is likely authentic or generated/manipulated **according to this model**.

**Out of scope:** legal or forensic evidence, decisions about individuals, identifying people, content outside the training distribution (other generators, heavy post-processing, non-face images on the face track).

## Training data
- Dataset ids / manifest / manifest SHA-256 prefix:
- Training sources (generators / methods):
- Excluded (held-out) sources:
- Train class counts (real / fake):
- Known biases:

## Training procedure
Epochs (completed / best), batch size, optimizer (AdamW), learning rate, weight decay, scheduler, AMP, augmentation, seed(s), hardware.

## Evaluation
| Split / condition | n | Accuracy | Macro-F1 | ROC-AUC (95% CI) | Recall FAKE | Recall REAL | Source file |
|---|---|---|---|---|---|---|---|
| test (seen sources) | | | | | | | `artifacts/evaluations/.../result.json` |
| unseen source(s) | | | | | | | |
| JPEG 50 | | | | | | | |

Latency (batch size 1, device, AMP):

## Limitations and risks
Summarize the measured weaknesses (unseen sources, perturbations) and refer to docs/LIMITATIONS.md.

## Ethical considerations
Datasets used under their licenses; no identity recognition; predictions are not proof.
