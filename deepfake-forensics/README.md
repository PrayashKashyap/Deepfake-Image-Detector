# Deepfake Image Detection and Forensic Analysis Platform

A research-oriented full-stack platform that classifies uploaded images as **REAL** (likely authentic according to the trained model) or **FAKE** (likely AI-generated or manipulated according to the trained model). It also provides Grad-CAM explanations, measured image forensics, robustness experiments and reproducible evaluation tooling.

> **Responsible use.** Predictions are probabilistic model outputs, **not proof** of authenticity or manipulation, and must never be used as legal or forensic evidence. The detector produces false positives and false negatives. It does not identify people. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Status: what works and what needs your data

| Component | Status |
|---|---|
| FastAPI backend, all endpoints, SQLite history, PDF reports, job queue | Implemented and tested (pytest) |
| React dashboard (7 pages) connected to the real API | Implemented, type-checked, Vitest tests |
| ResNet-50 / EfficientNet-B0 (+ optional Fourier branch), training, AMP, early stopping, resume | Implemented and **trained on real data** (RTX 4060) |
| **Facial-deepfake track**: models trained on DeepFakeFace with leakage-safe splits | **Done.** Installed model: EfficientNet-B0 + source normalisation, **95.47% test accuracy, ROC-AUC 0.991** |
| Face robustness (10 conditions) and leave-one-manipulation-out tests | **Done** (see `reports/RESULTS_real_data.md` and the paper) |
| **AI-generated-image track**: GenImage 8-generator subset | **Done** (single seed). Installed model: ResNet-50 + source normalisation, **88.79% test accuracy, ROC-AUC 0.957**, mean accuracy under 9 perturbations 76.6% (the unnormalised model reached 94.12% on clean images but relied on a JPEG-vs-PNG shortcut and fell to 68.4% under perturbations); leave-one-generator-out for Midjourney and BigGAN |
| IEEE paper (facial track) | `paper/` (Overleaf-ready zip `paper/paper_overleaf.zip`) |

All reported numbers come from `artifacts/evaluations/*/result.json`; `python -m ml.experiments.report --prefix real_ --out reports/RESULTS.md` regenerates the summary (current: [reports/RESULTS_real_data.md](reports/RESULTS_real_data.md)).
Datasets are **not** in the repository; see [docs/DATASETS.md](docs/DATASETS.md) for sources and exact preparation commands.

Until a checkpoint is installed, `/api/analyze` returns **HTTP 503 `model_unavailable`** and the UI explains why. No prediction is ever hardcoded.

## Architecture

```
deepfake-forensics/
  frontend/        React 19 + TypeScript + Vite + Tailwind 4 + Recharts (dashboard)
  backend/app/     FastAPI: api/ (routes) core/ (config, errors, security) schemas/ services/ database/ utils/
  backend/tests/   pytest (API integration + ML unit tests)
  ml/              Shared Python package used by BOTH training scripts and the backend
    data/            manifests, GenImage / FF++ preparation, validation, synthetic fixture
    preprocessing/   image decoding + the single eval transform used for evaluation AND inference
    models/          architectures, checkpoint format (weights + embedded metadata)
    training/        config, loops, CLI
    evaluation/      metrics, evaluation CLI, plots
    robustness/      perturbations, robustness CLI
    explainability/  Grad-CAM, Fourier spectra, Grad-CAM gallery
    experiments/     experiment runner + comparison/ablation aggregation
    configs/         train/, eval/, experiments/ YAML + datasets.example.yaml
  scripts/         PowerShell helpers (setup, run, test, smoke pipeline)
  artifacts/       checkpoints/<track>/, evaluations/, experiments/   (git-ignored contents)
  reports/         comparison tables (CSV / Markdown / LaTeX) and figures
  storage/         uploaded images, analysis artifacts, SQLite DB    (git-ignored)
  docs/            documentation
```

Request flow: the browser sends an upload to `POST /api/analyze`. The backend validates the bytes (magic number, full decode, size and pixel limits), then the ModelRegistry supplies the cached checkpoint. The image goes through the eval transform stored in that checkpoint, then softmax gives the FAKE score, which is compared with the threshold chosen on validation data. The backend then runs Grad-CAM, measures forensic properties, and stores the files and the SQLite row.

**Class convention:** `0 = REAL`, `1 = FAKE`. The *fake score* is `softmax(logits)[1]`. It is a model score, not a calibrated probability.

## Quick start (Windows 11, PowerShell)

Prerequisites: Python 3.11+ (verified on 3.14.5), Node.js 20+ (verified on 24.20), Git, and optionally an NVIDIA GPU with a recent driver.

```powershell
cd deepfake-forensics
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1        # add -Cpu for CPU-only PyTorch
# terminal 1
powershell -ExecutionPolicy Bypass -File scripts\run-backend.ps1  # http://127.0.0.1:8000/docs
# terminal 2
powershell -ExecutionPolicy Bypass -File scripts\run-frontend.ps1 # http://127.0.0.1:5173
```

Run all tests: `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`

Details: [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) · backend and frontend setup, CUDA troubleshooting.

## From zero to a working detector

1. Obtain a dataset legally ([docs/DATASETS.md](docs/DATASETS.md)) and build a manifest:
   `python -m ml.data.genimage --root D:\datasets\GenImage --out data\manifests\genimage_v1.csv --max-per-class-per-source 2000`
2. Validate it: `python -m ml.data.validate --manifest data\manifests\genimage_v1.csv --root D:\datasets\GenImage --near-duplicates`
3. Register it in `ml\configs\datasets.local.yaml`.
4. Train and install: `python -m ml.training.train --config ml\configs\train\aigen_efficientnet_b0.yaml --install`
5. Evaluate: `python -m ml.evaluation.evaluate --checkpoint artifacts\checkpoints\ai_generated\<model>.pt --dataset genimage_v1 --split test`
6. Open the UI. Analysis is now enabled for that track, and results appear on the Model Evaluation page.

## Features

- **Dashboard**: real counts only, with an honest empty state.
- **Image analysis**: drag-and-drop, track selection, model and version display, score, threshold, latency, a side-by-side Grad-CAM view, and a disclaimer.
- **Forensic analysis**: dimensions, format, size, color mode, EXIF with sensitive fields redacted, a JPEG quantization-table quality estimate, PNG text chunks, and a Fourier spectrum with its radial profile. Measurements are kept separate from model inferences.
- **Robustness laboratory**: JPEG, downscale, blur and noise sliders. The real model runs on each variant, clearly labeled as a single-image demonstration.
- **Model evaluation**: accuracy, precision and recall per class, macro-F1, ROC-AUC with grouped-bootstrap 95% CIs, confusion matrix, ROC, per-source (cross-generator / cross-manipulation) tables, latency, robustness deltas, ablation tables, training curves, artifact downloads and background evaluation jobs.
- **History and reports**: search and filters, pagination, detail view, deletion of records and files, PDF reports, metrics CSV export.
- **Settings**: health, CUDA and device info, checkpoint selection, per-track threshold overrides, architecture status (trained / pretrained / unavailable), and an optional API key.

## Documentation

| Doc | Contents |
|---|---|
| [INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) | Installation, backend/frontend setup, CUDA & GPU troubleshooting |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Environment-variable reference |
| [DATASETS.md](docs/DATASETS.md) | Dataset acquisition, licensing, preparation, leakage control |
| [TRAINING.md](docs/TRAINING.md) | Training commands, configs, checkpoints |
| [EVALUATION.md](docs/EVALUATION.md) | Evaluation, robustness, experiments 1–7, result schemas |
| [API.md](docs/API.md) | REST API reference |
| [MODEL_CARD_TEMPLATE.md](docs/MODEL_CARD_TEMPLATE.md) | Model card to fill in per trained model |
| [METHODOLOGY_TEMPLATE.md](docs/METHODOLOGY_TEMPLATE.md) | Research-paper methodology template |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Seeds, manifests, versions, what to archive |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Limitations and responsible use |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems |

## Screenshots

_Placeholders: add screenshots of the Dashboard, Image analysis, Forensics, Robustness lab and Evaluation pages after training a model._

## License and data

Code license: choose one before publishing (e.g. MIT). Datasets are **not** included and are governed by their own licenses. Large files (datasets, checkpoints, uploads) are excluded by `.gitignore`.
