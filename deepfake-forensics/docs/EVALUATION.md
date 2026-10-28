# Evaluation, robustness and research experiments

## Principles

- Thresholds and hyperparameters are chosen on **validation** data only. Evaluation reuses the threshold stored in the checkpoint; the test split is never used for tuning.
- Results are written as JSON/CSV with PNG figures. The UI and the comparison tables only **read** these files; nothing is estimated or invented.
- Every result records dataset ids, the manifest file and its SHA-256 prefix, splits, source filters, the model id/version, the training seed, the evaluation seed, the threshold provenance, the environment and a timestamp.
- Results derived from the synthetic fixture carry `is_synthetic_fixture: true` and must not be reported.

## Single evaluation

```powershell
python -m ml.evaluation.evaluate --checkpoint artifacts\checkpoints\ai_generated\<model>.pt --dataset genimage_v1 --split test
# cross-generator: only the held-out generator's test images
python -m ml.evaluation.evaluate --checkpoint <ckpt> --dataset genimage_v1 --split test --include-sources Midjourney
```

Metrics (scikit-learn, FAKE = positive class):
- accuracy and balanced accuracy;
- precision, recall and F1 per class, plus macro-F1;
- ROC-AUC (reported as `null` when undefined), ROC curve and confusion matrix;
- per-source metrics (detection rate for FAKE sources, specificity for REAL sources);
- 95% **grouped percentile-bootstrap** confidence intervals (groups resampled, so correlated frames of one video count together);
- latency: batch size 1, warm-up excluded, CUDA synchronised.

Output: `artifacts/evaluations/<eval_id>/` contains `result.json`, `metrics.csv`, `predictions.csv`, `roc.png` and `confusion_matrix.png`.

Evaluations can also be started from the UI (Model Evaluation → Run an evaluation job) or with `POST /api/evaluation/run`. They run as background jobs with progress reporting.

## Robustness (Experiment 4)

```powershell
python -m ml.robustness.evaluate_robustness --checkpoint <ckpt> --dataset genimage_v1 --split test
python -m ml.evaluation.evaluate --config ml\configs\eval\exp4_robustness.yaml --checkpoint <ckpt> --dataset genimage_v1
```

Default suite: clean, JPEG 90/70/50, downscale ×0.5/×0.25 (restored to the original size), Gaussian blur σ = 1/2, and Gaussian noise σ = 5/15 (0–255 scale, per-image seeded). Each condition reports the same metrics at the **unchanged** threshold, plus the deltas against clean (`robustness.csv`, `robustness.png`, `roc_robustness.png`).

## Experiments

Run any experiment with `python -m ml.experiments.run --experiment <yaml>` (`--dry-run` prints the plan; `--skip-training` reuses finished runs). The runner trains every variant and seed, evaluates each best checkpoint, aggregates the results with `ml.experiments.compare`, and records everything in `artifacts/experiments/<exp>/experiment_runs.json`.

| # | Config | Design |
|---|---|---|
| 1 | `exp1_baseline_aigen.yaml`, `exp1_baseline_face.yaml` | ResNet-50 vs EfficientNet-B0, identical manifest, splits and preprocessing, seeds 42/43/44 |
| 2 | `exp2_cross_generator.yaml` | Leave-one-generator-out: the generator is excluded from train **and** val, then tested alone. Edit the source names to match your manifest summary |
| 3 | `exp3_cross_manipulation.yaml` | Train without one FF++ method (plus FaceShifter); test on `original` + the unseen method (video-grouped splits) |
| 4 | `exp4_robustness.yaml` | Robustness suite for both architectures |
| 5 & 7 | `exp5_7_frequency_ablation.yaml` | RGB-only vs RGB + Fourier branch. The only difference is `model.frequency_branch`; 3 seeds |
| 6 | Grad-CAM gallery | `python -m ml.explainability.gallery --eval-dir artifacts\evaluations\<id> --checkpoint <ckpt> --root <dataset root> --k 6` renders TP/TN/FP/FN examples (`gradcam_gallery/*.png`) |

Manual comparison (any set of `result.json` files; comma-separated paths are seeds):

```powershell
python -m ml.experiments.compare --name my_ablation --type ablation --baseline "RGB" `
  --variant "RGB=artifacts\evaluations\a_s42\result.json,artifacts\evaluations\a_s43\result.json" `
  --variant "RGB+FFT=artifacts\evaluations\b_s42\result.json,artifacts\evaluations\b_s43\result.json"
```

This writes `reports/<name>/comparison.{csv,md,tex,png}` (mean ± sample std across seeds) and `artifacts/evaluations/<name>/result.json` (kind `comparison`, shown in the dashboard).

Interpretation guidance: differences smaller than the spread across seeds, or inside overlapping CIs, are not evidence of improvement. Do not call results state-of-the-art without a matched comparison on the same benchmark protocol.

## Result schemas (schema_version 1.0)

`kind: "evaluation" | "robustness"`

```jsonc
{
  "schema_version": "1.0", "kind": "evaluation", "eval_id": "...", "experiment_id": "...", "created_at": "ISO-8601",
  "track": "ai_generated", "description": "",
  "model": {"model_id", "version", "arch", "frequency_branch", "checkpoint_file", "training_run", "training_seed"},
  "dataset": {"dataset_ids", "manifest", "manifest_sha256_16", "splits", "include_sources", "exclude_sources",
              "eval_sources", "train_sources", "unseen_sources", "n_samples", "class_counts", "is_synthetic_fixture"},
  "seed": 0, "threshold": {"value", "selected_on", "method"},
  "metrics": {"n", "threshold", "accuracy", "balanced_accuracy", "precision_real", "recall_real", "f1_real",
              "precision_fake", "recall_fake", "f1_fake", "f1_macro", "roc_auc", "support_real", "support_fake", "confusion_matrix"},
  "per_class": [{"class", "precision", "recall", "f1", "support"}],
  "confusion_matrix": [[TN, FP], [FN, TP]],          // rows = true [REAL, FAKE]
  "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
  "per_source": [{"source", "n", "labels_present", "accuracy", "detection_rate", "specificity", "roc_auc", "mean_fake_score"}],
  "confidence_intervals": {"method": "grouped_percentile_bootstrap", "alpha", "n_groups", "metrics": {"accuracy": {"low", "high", "n_boot"}}},
  "latency": {"batch_size": 1, "n", "mean_ms", "p50_ms", "p95_ms", "device", "amp"},
  "environment": {"torch_version", "cuda_available", "cuda_version", "cudnn_version"},
  "artifacts": {"predictions_csv", "confusion_matrix_png", "roc_png", "metrics_csv"},
  "robustness": {"reference": "clean", "conditions": [{"name", "perturbation", "accuracy", "f1_macro", "roc_auc",
                 "recall_fake", "recall_real", "delta_accuracy", "delta_f1_macro", "delta_roc_auc"}], "detail": []}  // robustness only
}
```

`kind: "training"` (`run.json`): experiment_id, run_id, track, model, dataset (with train/val summaries), seed, threshold, best_epoch, monitor, best_value, validation_metrics, history[], environment, artifacts.

`kind: "comparison"`: comparison_type (`comparison|ablation|cross_source`), baseline, variants{label: {n_runs, eval_ids, models, seeds, dataset, metrics{m: {mean, std, n}}, latency_mean_ms}}, table[].
