"""Health, models and settings endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ml.common import TRACKS, device_info, utc_now_iso
from ml.data.registry import datasets_config_path, list_datasets

from app.core.config import APP_VERSION
from app.core.container import Container, get_container
from app.core.errors import NotFoundError
from app.schemas.api import (
    ActiveModelUpdate,
    DatasetsResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SettingsResponse,
    SettingsUpdate,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Application health, device availability and service status")
def health(c: Container = Depends(get_container)) -> HealthResponse:
    db_ok = c.db.ping()
    storage_ok = c.settings.storage_dir.is_dir()
    tracks = c.registry.track_status()
    services = {
        "database": "ok" if db_ok else "error",
        "storage": "ok" if storage_ok else "error",
        **{f"model:{t['track']}": ("ready" if t["status"] == "ready" else "no checkpoint installed") for t in tracks},
    }
    return HealthResponse(
        status="ok" if db_ok and storage_ok else "degraded",
        version=APP_VERSION,
        time=utc_now_iso(),
        device={"inference_device": str(c.registry.device), **device_info()},
        services=services,
        auth_required=bool(c.settings.api_key),
    )


@router.get("/models", response_model=ModelsResponse, summary="Installed models, supported tracks and checkpoint status")
def models(c: Container = Depends(get_container)) -> ModelsResponse:
    loaded = set(c.registry.loaded_ids())
    active = {t: c.registry.active_model_id(t) for t in TRACKS}
    items = []
    for e in c.registry.scan():
        m = e.meta
        items.append(ModelInfo(
            model_id=e.model_id,
            track=e.track,
            status="trained" if e.valid else "invalid",
            is_active=active.get(e.track) == e.model_id,
            loaded=e.model_id in loaded,
            display_name=m.get("display_name"),
            arch=m.get("arch"),
            frequency_branch=m.get("frequency_branch"),
            version=m.get("version"),
            created_at=m.get("created_at"),
            pretrained_init=m.get("pretrained_init"),
            threshold=m.get("threshold"),
            validation_metrics=m.get("validation_metrics"),
            dataset=m.get("dataset"),
            training=m.get("training"),
            preprocess=m.get("preprocess"),
            is_synthetic_fixture=e.is_fixture,
            error=e.error,
        ))
    return ModelsResponse(tracks=c.registry.track_status(), models=items, architectures=c.registry.pretrained_status(), inference_device=str(c.registry.device))


@router.put("/models/{track}/active", response_model=ModelsResponse, summary="Select the active checkpoint for a track")
def set_active_model(track: str, body: ActiveModelUpdate, c: Container = Depends(get_container)) -> ModelsResponse:
    if track not in TRACKS:
        raise NotFoundError(f"Unknown track '{track}'")
    c.registry.set_active(track, body.model_id)
    return models(c)


@router.get("/settings", response_model=SettingsResponse, summary="Current runtime settings")
def get_settings_route(c: Container = Depends(get_container)) -> SettingsResponse:
    return SettingsResponse(
        thresholds={t: c.registry.threshold_override(t) for t in TRACKS},
        active_models={t: c.registry.active_model_id(t) for t in TRACKS},
        gradcam_enabled=c.settings.enable_gradcam,
        max_upload_mb=c.settings.max_upload_mb,
        max_image_pixels=c.settings.max_image_pixels,
        inference_amp=c.settings.inference_amp,
    )


@router.put("/settings", response_model=SettingsResponse, summary="Update per-track threshold overrides")
def update_settings(body: SettingsUpdate, c: Container = Depends(get_container)) -> SettingsResponse:
    for track, value in (body.thresholds or {}).items():
        c.registry.set_threshold_override(track, value)
    return get_settings_route(c)


@router.get("/datasets", response_model=DatasetsResponse, summary="Named datasets configured in datasets.local.yaml (paths are not exposed)")
def datasets(c: Container = Depends(get_container)) -> DatasetsResponse:
    items = []
    for name, entry in list_datasets().items():
        items.append({"name": name, "track": entry.get("track"), "description": entry.get("description", "")})
    return DatasetsResponse(datasets=items, config_file_present=datasets_config_path().exists())
