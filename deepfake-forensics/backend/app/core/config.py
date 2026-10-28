"""Application settings, loaded from environment variables (prefix DFF_) and an optional .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.common import REPO_ROOT, resolve_repo_path

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DFF_",
        env_file=(str(REPO_ROOT / ".env"), str(REPO_ROOT / "backend" / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Deepfake Image Detection and Forensic Analysis Platform"
    environment: str = "development"
    log_level: str = "INFO"

    # Storage (relative paths are resolved against the repository root)
    storage_dir: Path = Path("storage")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    database_path: Path = Path("storage/app.db")

    # HTTP
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    api_key: str = Field(default="", description="If set, every /api request must send X-API-Key")

    # Upload validation
    max_upload_mb: float = 15.0
    max_image_pixels: int = 40_000_000
    min_image_side: int = 32

    # Inference
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    inference_amp: bool = True
    model_cache_size: int = 2
    enable_gradcam: bool = True

    # Evaluation jobs: dataset roots/manifests given directly must live under one of these
    # directories (comma separated). Named datasets from datasets.local.yaml are always allowed.
    data_roots: str = "data"
    job_workers: int = 1

    @field_validator("storage_dir", "artifacts_dir", "reports_dir", "database_path", mode="after")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return resolve_repo_path(v)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_data_roots(self) -> list[Path]:
        return [resolve_repo_path(p.strip()) for p in self.data_roots.split(",") if p.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def evaluations_dir(self) -> Path:
        return self.artifacts_dir / "evaluations"

    @property
    def experiments_dir(self) -> Path:
        return self.artifacts_dir / "experiments"

    @property
    def analyses_dir(self) -> Path:
        return self.storage_dir / "analyses"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.analyses_dir, self.checkpoints_dir, self.evaluations_dir, self.experiments_dir, self.reports_dir, self.database_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        for track in ("ai_generated", "face_deepfake"):
            (self.checkpoints_dir / track).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
