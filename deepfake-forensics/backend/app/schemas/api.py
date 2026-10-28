"""Pydantic request/response schemas (documented in OpenAPI at /docs)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Track = Literal["ai_generated", "face_deepfake"]
PredictionLabel = Literal["REAL", "FAKE"]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------------------------------------------------------- health / models
class GPUDevice(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    capability: str


class DeviceStatus(BaseModel):
    inference_device: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: int | None
    devices: list[GPUDevice]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    time: str
    device: DeviceStatus
    services: dict[str, str]
    auth_required: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    track: str
    status: Literal["trained", "invalid"]
    is_active: bool
    loaded: bool
    display_name: str | None = None
    arch: str | None = None
    frequency_branch: bool | None = None
    version: str | None = None
    created_at: str | None = None
    pretrained_init: str | None = None
    threshold: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    preprocess: dict[str, Any] | None = None
    is_synthetic_fixture: bool = False
    error: str | None = None


class TrackStatus(BaseModel):
    track: Track
    label: str
    status: Literal["ready", "unavailable"]
    active_model_id: str | None
    threshold_override: float | None
    message: str | None


class ArchitectureStatus(BaseModel):
    arch: str
    status: Literal["trained", "pretrained", "unavailable"]
    trained_checkpoints: int
    pretrained_backbone_cached: bool
    pretrained_source: str
    note: str


class ModelsResponse(BaseModel):
    tracks: list[TrackStatus]
    models: list[ModelInfo]
    architectures: list[ArchitectureStatus]
    inference_device: str


class ActiveModelUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str | None = Field(default=None, description="Installed model id, or null to use the newest checkpoint")


# ----------------------------------------------------------------------------- settings
class SettingsResponse(BaseModel):
    thresholds: dict[str, float | None]
    active_models: dict[str, str | None]
    gradcam_enabled: bool
    max_upload_mb: float
    max_image_pixels: int
    inference_amp: bool


class SettingsUpdate(BaseModel):
    thresholds: dict[Track, float | None] | None = Field(default=None, description="Per-track threshold override in (0, 1); null resets to the checkpoint's validation-selected threshold")

    @field_validator("thresholds")
    @classmethod
    def _range(cls, v):
        for k, t in (v or {}).items():
            if t is not None and not (0.0 < float(t) < 1.0):
                raise ValueError(f"threshold for {k} must be strictly between 0 and 1")
        return v


# ----------------------------------------------------------------------------- analyses
class ModelRef(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    display_name: str | None = None
    arch: str | None = None
    version: str | None = None
    frequency_branch: bool = False
    is_synthetic_fixture: bool = False
    trained_on: list[str] = []


class ImageSummary(BaseModel):
    file_name: str
    format: str
    file_size_bytes: int
    width: int
    height: int
    color_mode: str
    sha256: str


class AnalysisResult(BaseModel):
    id: str
    created_at: str
    track: Track
    track_label: str
    prediction: PredictionLabel
    fake_score: float = Field(description="Softmax output of the model for class FAKE (index 1)")
    real_score: float
    threshold: float
    threshold_source: str
    inference_ms: float
    gradcam_ms: float | None
    device: str
    model: ModelRef
    image: ImageSummary
    explanation: dict[str, Any]
    forensics: dict[str, Any]
    artifacts: dict[str, str]
    warnings: list[str]
    score_note: str
    disclaimer: str


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    created_at: str
    track: str
    model_id: str
    model_version: str | None
    arch: str | None
    prediction: PredictionLabel
    fake_score: float
    threshold: float
    threshold_source: str | None
    inference_ms: float | None
    file_name: str | None
    file_format: str | None
    file_size: int | None
    width: int | None
    height: int | None
    is_fixture_model: bool


class AnalysisList(BaseModel):
    items: list[AnalysisSummary]
    total: int
    page: int
    page_size: int


class StatsResponse(BaseModel):
    total: int
    real: int
    fake: int
    by_track: list[dict[str, Any]]
    per_day: list[dict[str, Any]]
    fixture_model_analyses: int


class ForensicsResponse(BaseModel):
    analysis_id: str
    image: ImageSummary
    properties: dict[str, Any]
    jpeg: dict[str, Any] | None
    png: dict[str, Any] | None
    pixel_statistics: dict[str, Any]
    exif: dict[str, Any]
    frequency: dict[str, Any]
    observations: list[dict[str, str]]
    spectrum_url: str | None
    note: str


# ----------------------------------------------------------------------------- robustness
class RobustnessParams(BaseModel):
    jpeg_quality: int | None = Field(default=None, ge=5, le=100)
    downscale_max_side: int | None = Field(default=None, ge=16, le=4096)
    blur_sigma: float | None = Field(default=None, ge=0.0, le=10.0)
    noise_std: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _at_least_one(self):
        if all(getattr(self, f) is None for f in ("jpeg_quality", "downscale_max_side", "blur_sigma", "noise_std")):
            raise ValueError("Select at least one perturbation")
        return self


class RobustnessVariant(BaseModel):
    name: str
    perturbations: list[dict[str, Any]]
    prediction: PredictionLabel
    fake_score: float
    delta_fake_score: float
    prediction_changed: bool
    inference_ms: float
    image_data_url: str


class RobustnessResponse(BaseModel):
    track: Track
    track_label: str
    model: ModelRef
    threshold: float
    threshold_source: str
    original: RobustnessVariant
    variants: list[RobustnessVariant]
    warnings: list[str]
    note: str


# ----------------------------------------------------------------------------- evaluation / jobs
class ExperimentSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    kind: str
    experiment_id: str | None
    track: str | None
    created_at: str | None
    description: str | None
    model_id: str | None
    arch: str | None
    frequency_branch: bool | None
    splits: list[str | None] | None
    dataset_ids: list[str] | None
    n_samples: int | None
    unseen_sources: list[str] | None
    is_synthetic_fixture: bool
    metrics: dict[str, float | None]
    metrics_scope: str
    comparison_type: str | None
    latency_mean_ms: float | None


class EvaluationSummary(BaseModel):
    available: bool
    counts: dict[str, int]
    latest_evaluation_by_track: dict[str, ExperimentSummary]
    experiments: list[ExperimentSummary]
    message: str | None


class ExperimentDetail(BaseModel):
    summary: ExperimentSummary
    result: dict[str, Any]
    artifacts: list[dict[str, str]]


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(description="Installed checkpoint id (see GET /api/models)")
    dataset_name: str | None = Field(default=None, description="Named dataset from datasets.local.yaml")
    dataset_root: str | None = Field(default=None, description="Dataset root (must be inside DFF_DATA_ROOTS)")
    manifest: str | None = Field(default=None, description="Manifest CSV (must be inside DFF_DATA_ROOTS)")
    splits: list[Literal["train", "val", "test"]] = ["test"]
    include_sources: list[str] = []
    exclude_sources: list[str] = []
    max_per_class: int | None = Field(default=None, ge=1, le=1_000_000)
    batch_size: int = Field(default=32, ge=1, le=512)
    num_workers: int = Field(default=0, ge=0, le=16)
    robustness: bool = False
    bootstrap: bool = True

    @model_validator(mode="after")
    def _dataset(self):
        if not self.dataset_name and not (self.dataset_root and self.manifest):
            raise ValueError("Provide dataset_name, or both dataset_root and manifest")
        if not self.splits:
            raise ValueError("splits must not be empty")
        return self


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: float
    message: str | None
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None
    logs: list[str] = []
    created_at: str
    started_at: str | None
    finished_at: str | None


class DatasetsResponse(BaseModel):
    datasets: list[dict[str, Any]]
    config_file_present: bool
