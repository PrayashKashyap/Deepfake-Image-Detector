// Types mirroring backend/app/schemas/api.py
export type Track = 'ai_generated' | 'face_deepfake'
export type PredictionLabel = 'REAL' | 'FAKE'

export const TRACK_LABELS: Record<Track, string> = {
  ai_generated: 'AI-generated images',
  face_deepfake: 'Facial deepfakes',
}

export interface ApiErrorBody {
  code: string
  message: string
  details?: unknown
}

export interface GPUDevice {
  index: number
  name: string
  total_memory_mb: number
  capability: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  version: string
  time: string
  device: {
    inference_device: string
    torch_version: string
    cuda_available: boolean
    cuda_version: string | null
    cudnn_version: number | null
    devices: GPUDevice[]
  }
  services: Record<string, string>
  auth_required: boolean
}

export interface ModelInfo {
  model_id: string
  track: Track
  status: 'trained' | 'invalid'
  is_active: boolean
  loaded: boolean
  display_name?: string | null
  arch?: string | null
  frequency_branch?: boolean | null
  version?: string | null
  created_at?: string | null
  pretrained_init?: string | null
  threshold?: { value: number; selected_on: string; method: string } | null
  validation_metrics?: Record<string, number | null> | null
  dataset?: Record<string, unknown> | null
  training?: Record<string, unknown> | null
  is_synthetic_fixture: boolean
  error?: string | null
}

export interface TrackStatus {
  track: Track
  label: string
  status: 'ready' | 'unavailable'
  active_model_id: string | null
  threshold_override: number | null
  message: string | null
}

export interface ArchitectureStatus {
  arch: string
  status: 'trained' | 'pretrained' | 'unavailable'
  trained_checkpoints: number
  pretrained_backbone_cached: boolean
  pretrained_source: string
  note: string
}

export interface ModelsResponse {
  tracks: TrackStatus[]
  models: ModelInfo[]
  architectures: ArchitectureStatus[]
  inference_device: string
}

export interface SettingsResponse {
  thresholds: Record<Track, number | null>
  active_models: Record<Track, string | null>
  gradcam_enabled: boolean
  max_upload_mb: number
  max_image_pixels: number
  inference_amp: boolean
}

export interface ModelRef {
  model_id: string
  display_name?: string | null
  arch?: string | null
  version?: string | null
  frequency_branch: boolean
  is_synthetic_fixture: boolean
  trained_on: string[]
}

export interface ImageSummary {
  file_name: string
  format: string
  file_size_bytes: number
  width: number
  height: number
  color_mode: string
  sha256: string
}

export interface Observation {
  kind: string
  text: string
  caveat: string
}

export interface ForensicsData {
  properties: Record<string, unknown>
  jpeg: Record<string, unknown> | null
  png: Record<string, unknown> | null
  pixel_statistics: Record<string, unknown>
  exif: { present: boolean; tags: Record<string, unknown>; redacted: string[] }
  frequency: { radial_profile?: { radius: number[]; log_magnitude: number[] } }
  observations: Observation[]
}

export interface AnalysisResult {
  id: string
  created_at: string
  track: Track
  track_label: string
  prediction: PredictionLabel
  fake_score: number
  real_score: number
  threshold: number
  threshold_source: string
  inference_ms: number
  gradcam_ms: number | null
  device: string
  model: ModelRef
  image: ImageSummary
  explanation: {
    available: boolean
    explained_class?: string
    target_layer?: string
    resolution_note?: string
    note?: string
    reason?: string
  }
  forensics: ForensicsData
  artifacts: Partial<Record<'original' | 'input' | 'gradcam' | 'heatmap' | 'spectrum', string>>
  warnings: string[]
  score_note: string
  disclaimer: string
}

export interface AnalysisSummary {
  id: string
  created_at: string
  track: Track
  model_id: string
  model_version: string | null
  arch: string | null
  prediction: PredictionLabel
  fake_score: number
  threshold: number
  threshold_source: string | null
  inference_ms: number | null
  file_name: string | null
  file_format: string | null
  file_size: number | null
  width: number | null
  height: number | null
  is_fixture_model: boolean
}

export interface AnalysisList {
  items: AnalysisSummary[]
  total: number
  page: number
  page_size: number
}

export interface StatsResponse {
  total: number
  real: number
  fake: number
  by_track: { track: Track; prediction: PredictionLabel; n: number }[]
  per_day: { day: string; prediction: PredictionLabel; n: number }[]
  fixture_model_analyses: number
}

export interface ForensicsResponse extends ForensicsData {
  analysis_id: string
  image: ImageSummary
  spectrum_url: string | null
  note: string
}

export interface RobustnessVariant {
  name: string
  perturbations: Record<string, unknown>[]
  prediction: PredictionLabel
  fake_score: number
  delta_fake_score: number
  prediction_changed: boolean
  inference_ms: number
  image_data_url: string
}

export interface RobustnessResponse {
  track: Track
  track_label: string
  model: ModelRef
  threshold: number
  threshold_source: string
  original: RobustnessVariant
  variants: RobustnessVariant[]
  warnings: string[]
  note: string
}

export interface RobustnessParams {
  jpeg_quality?: number | null
  downscale_max_side?: number | null
  blur_sigma?: number | null
  noise_std?: number | null
}

export interface ExperimentSummary {
  id: string
  kind: 'training' | 'evaluation' | 'robustness' | 'comparison' | string
  experiment_id: string | null
  track: string | null
  created_at: string | null
  description: string | null
  model_id: string | null
  arch: string | null
  frequency_branch: boolean | null
  splits: (string | null)[] | null
  dataset_ids: string[] | null
  n_samples: number | null
  unseen_sources: string[] | null
  is_synthetic_fixture: boolean
  metrics: Record<string, number | null>
  metrics_scope: string
  comparison_type: string | null
  latency_mean_ms: number | null
}

export interface EvaluationSummary {
  available: boolean
  counts: Record<string, number>
  latest_evaluation_by_track: Record<string, ExperimentSummary>
  experiments: ExperimentSummary[]
  message: string | null
}

export interface ExperimentDetail {
  summary: ExperimentSummary
  // Raw result.json / run.json as written by the ml package
  result: Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
  artifacts: { name: string; url: string }[]
}

export interface EvaluationRunRequest {
  model_id: string
  dataset_name?: string | null
  dataset_root?: string | null
  manifest?: string | null
  splits: ('train' | 'val' | 'test')[]
  include_sources?: string[]
  exclude_sources?: string[]
  max_per_class?: number | null
  batch_size?: number
  robustness?: boolean
}

export interface JobStatus {
  id: string
  kind: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  progress: number
  message: string | null
  result: Record<string, string> | null
  error: string | null
  logs: string[]
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface DatasetsResponse {
  datasets: { name: string; track: string | null; description: string }[]
  config_file_present: boolean
}
