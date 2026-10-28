// Test-only mock payloads (shapes mirror the backend schemas). Never used by the application.
import type { AnalysisResult, EvaluationSummary, ExperimentDetail, ExperimentSummary, ModelsResponse, StatsResponse } from '../api/types'

export const modelsReady: ModelsResponse = {
  tracks: [
    { track: 'ai_generated', label: 'AI-generated image detection', status: 'ready', active_model_id: 'm1', threshold_override: null, message: null },
    { track: 'face_deepfake', label: 'Facial deepfake detection', status: 'unavailable', active_model_id: null, threshold_override: null, message: 'No trained checkpoint installed for this track.' },
  ],
  models: [{ model_id: 'm1', track: 'ai_generated', status: 'trained', is_active: true, loaded: false, display_name: 'resnet50 (ai_generated)', arch: 'resnet50', version: 'r1@e3', is_synthetic_fixture: false }],
  architectures: [],
  inference_device: 'cpu',
}

export const modelsNone: ModelsResponse = {
  ...modelsReady,
  tracks: modelsReady.tracks.map((t) => ({ ...t, status: 'unavailable' as const, active_model_id: null, message: 'No trained checkpoint installed for this track.' })),
  models: [],
}

export const analysis: AnalysisResult = {
  id: 'abc123',
  created_at: '2026-10-26T12:00:00+00:00',
  track: 'ai_generated',
  track_label: 'AI-generated image detection',
  prediction: 'FAKE',
  fake_score: 0.8731,
  real_score: 0.1269,
  threshold: 0.42,
  threshold_source: 'checkpoint (youden, selected on validation)',
  inference_ms: 12.3,
  gradcam_ms: 30.1,
  device: 'cpu',
  model: { model_id: 'm1', display_name: 'resnet50 (ai_generated)', arch: 'resnet50', version: 'r1@e3', frequency_branch: false, is_synthetic_fixture: false, trained_on: ['genimage'] },
  image: { file_name: 'cat.png', format: 'PNG', file_size_bytes: 1000, width: 100, height: 80, color_mode: 'RGB', sha256: 'x' },
  explanation: { available: true, explained_class: 'FAKE', target_layer: 'layer4[-1]', note: 'qualitative' },
  forensics: { properties: {}, jpeg: null, png: null, pixel_statistics: {}, exif: { present: false, tags: {}, redacted: [] }, frequency: {}, observations: [] },
  artifacts: { original: '/api/analyses/abc123/artifacts/original', gradcam: '/api/analyses/abc123/artifacts/gradcam' },
  warnings: [],
  score_note: 'model score note',
  disclaimer: 'This prediction is not proof.',
}

export const statsEmpty: StatsResponse = { total: 0, real: 0, fake: 0, by_track: [], per_day: [], fixture_model_analyses: 0 }
export const evalEmpty: EvaluationSummary = { available: false, counts: {}, latest_evaluation_by_track: {}, experiments: [], message: 'No evaluation results available.' }

const summary: ExperimentSummary = {
  id: 'eval--e1',
  kind: 'evaluation',
  experiment_id: 'exp1',
  track: 'ai_generated',
  created_at: '2026-10-26T12:00:00+00:00',
  description: '',
  model_id: 'm1',
  arch: 'resnet50',
  frequency_branch: false,
  splits: ['test'],
  dataset_ids: ['genimage'],
  n_samples: 4,
  unseen_sources: null,
  is_synthetic_fixture: false,
  metrics: { accuracy: 0.75, f1_macro: 0.7333, roc_auc: 0.875 },
  metrics_scope: 'evaluation split',
  comparison_type: null,
  latency_mean_ms: 5,
}
export const evalSummary: EvaluationSummary = { available: true, counts: { evaluation: 1, training: 0, robustness: 0, comparison: 0 }, latest_evaluation_by_track: { ai_generated: summary }, experiments: [summary], message: null }
export const evalDetail: ExperimentDetail = {
  summary,
  artifacts: [{ name: 'roc.png', url: '/api/evaluation/experiments/eval--e1/artifacts/roc.png' }],
  result: {
    kind: 'evaluation',
    experiment_id: 'exp1',
    track: 'ai_generated',
    created_at: summary.created_at,
    model: { model_id: 'm1', arch: 'resnet50' },
    dataset: { dataset_ids: ['genimage'], splits: ['test'], n_samples: 4, class_counts: { real: 2, fake: 2 } },
    threshold: { value: 0.42, method: 'youden', selected_on: 'validation' },
    seed: 0,
    metrics: { accuracy: 0.75, f1_macro: 0.7333, roc_auc: 0.875 },
    per_class: [
      { class: 'REAL', precision: 1, recall: 0.5, f1: 0.667, support: 2 },
      { class: 'FAKE', precision: 0.667, recall: 1, f1: 0.8, support: 2 },
    ],
    confusion_matrix: [
      [1, 1],
      [0, 2],
    ],
    roc_curve: { fpr: [0, 0, 0.5, 1], tpr: [0, 0.5, 1, 1], thresholds: [1, 0.9, 0.4, 0.1] },
    per_source: [{ source: 'sd', n: 2, accuracy: 1, detection_rate: 1, specificity: null, roc_auc: null, mean_fake_score: 0.8 }],
    latency: { mean_ms: 5, p95_ms: 7, device: 'cpu', amp: false },
  },
}
