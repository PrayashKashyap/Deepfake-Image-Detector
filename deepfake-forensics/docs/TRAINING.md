# Training

Training is **only** started from the command line. The API never trains, and never downloads weights at startup.

## Configs

| Config | Track | Model |
|---|---|---|
| `ml/configs/train/aigen_resnet50.yaml` | AI-generated | ResNet-50 (ImageNet init) |
| `ml/configs/train/aigen_efficientnet_b0.yaml` | AI-generated | EfficientNet-B0 |
| `ml/configs/train/face_resnet50.yaml` | Facial deepfake | ResNet-50, plain resize, JPEG/blur augmentation |
| `ml/configs/train/face_efficientnet_b0.yaml` | Facial deepfake | EfficientNet-B0 |
| `ml/configs/train/fixture_smoke.yaml` | — | Software smoke test on the synthetic fixture |

Main keys (defaults in [`ml/training/config.py`](../ml/training/config.py)):
- `dataset.{name | root+manifest, train_split, val_split, include_sources, exclude_sources, max_train_per_class}`
- `model.{arch, pretrained, frequency_branch, dropout}`
- `preprocess.{img_size, resize_mode}`
- `augment.{hflip, random_crop_scale, jpeg_prob, blur_prob}`
- `train.{epochs, batch_size, lr, weight_decay, seed, num_workers, amp, scheduler (cosine|plateau|none), warmup_epochs, early_stopping_patience, monitor, grad_clip, threshold_method}`

## Commands (PowerShell, venv active)

```powershell
# Train and install the best checkpoint for the API
python -m ml.training.train --config ml\configs\train\aigen_efficientnet_b0.yaml --install

# Override values without editing the YAML
python -m ml.training.train --config ml\configs\train\aigen_resnet50.yaml --set train.epochs=5 train.batch_size=32 --seed 43

# Resume an interrupted run
python -m ml.training.train --config ml\configs\train\aigen_resnet50.yaml `
    --resume artifacts\experiments\aigen_resnet50\<run_id>\checkpoints\last.pt

# Install an existing checkpoint manually
Copy-Item artifacts\experiments\<exp>\<run>\checkpoints\best.pt  artifacts\checkpoints\ai_generated\<name>.pt
Copy-Item artifacts\experiments\<exp>\<run>\checkpoints\best.json artifacts\checkpoints\ai_generated\<name>.json
```

ImageNet weights for `pretrained: true` come from the official torchvision model zoo (`download.pytorch.org`) and are cached in `%USERPROFILE%\.cache\torch\hub\checkpoints` on first use.

## What a run produces

`artifacts/experiments/<experiment_id>/<run_id>/` (run id = `YYYYMMDD-HHMMSS_s<seed>`):

- `config.resolved.yaml`: the exact configuration;
- `train.log` and `history.csv`: per-epoch losses and validation metrics;
- `loss_curve.png` and `val_metrics_curve.png`;
- `checkpoints/best.pt` and `best.json`: the best epoch by `monitor`, with the **threshold selected on the validation split**;
- `checkpoints/last.pt`: optimizer, scheduler and scaler state, for resume;
- `run.json`: a machine-readable summary (kind `training`), shown on the Evaluation page.

Each checkpoint embeds its architecture, preprocessing, class order, threshold (value, method and split), dataset ids, manifest hash, training/validation source lists, seed, hyperparameters, config hash and version. The backend loads checkpoints with `torch.load(weights_only=True)`, which refuses arbitrary pickled code.

## Practical settings for an RTX 4060 (8 GB)

| Model | img_size | batch (AMP) | Note |
|---|---|---|---|
| ResNet-50 | 224 | 48 | lower to 32 if other apps use VRAM |
| EfficientNet-B0 | 224 | 64 | |
| + frequency branch | 224 | −10–20% | the FFT runs in float32 |

These are starting points, not measured optima. Set `num_workers` to 4–8 depending on CPU cores.
