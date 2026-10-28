"""Discovers installed checkpoints, tracks the active model per track, and caches loaded models.

Checkpoints are installed under  <artifacts>/checkpoints/<track>/<model_id>.pt  (+ .json sidecar),
normally by `python -m ml.training.train ... --install`. Models are loaded lazily on first
use and kept in a small LRU cache; they are never reloaded per request.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ml.common import TRACK_LABELS, TRACKS, resolve_device
from ml.models.checkpoint import load_model_for_inference, read_checkpoint_meta, validate_meta
from ml.models.factory import ARCHITECTURES, PRETRAINED_SOURCES, DetectorNet
from ml.preprocessing.transforms import PreprocessConfig, build_eval_transform

from app.core.config import Settings
from app.core.errors import ModelUnavailableError, NotFoundError
from app.database.repositories import SettingsRepository

log = logging.getLogger("dff.models")

# File names torchvision uses for the ImageNet weights referenced in ml.models.factory.
_PRETRAINED_FILES = {"resnet50": "resnet50-11ad3fa6.pth", "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth"}


@dataclass
class ModelEntry:
    model_id: str
    track: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    error: str | None = None

    @property
    def is_fixture(self) -> bool:
        return bool(self.meta.get("dataset", {}).get("is_synthetic_fixture"))


@dataclass
class LoadedModel:
    entry: ModelEntry
    model: DetectorNet
    meta: dict[str, Any]
    preprocess: PreprocessConfig
    transform: Any
    device: torch.device
    lock: threading.Lock = field(default_factory=threading.Lock)  # Grad-CAM hooks are not re-entrant


class ModelRegistry:
    def __init__(self, settings: Settings, settings_repo: SettingsRepository) -> None:
        self.settings = settings
        self.repo = settings_repo
        self.device = resolve_device(settings.device)
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def scan(self) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        for track in TRACKS:
            d = self.settings.checkpoints_dir / track
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.pt")):
                try:
                    meta = read_checkpoint_meta(p)
                    validate_meta(meta)
                    if meta["track"] != track:
                        raise ValueError(f"checkpoint track {meta['track']!r} does not match directory {track!r}")
                    entries.append(ModelEntry(p.stem, track, p, meta))
                except Exception as exc:  # noqa: BLE001 - surface any unreadable checkpoint as invalid
                    entries.append(ModelEntry(p.stem, track, p, {}, valid=False, error=f"{type(exc).__name__}: {exc}"))
        return entries

    def get_entry(self, model_id: str) -> ModelEntry:
        for e in self.scan():
            if e.model_id == model_id:
                return e
        raise NotFoundError(f"Model '{model_id}' is not installed")

    def active_model_id(self, track: str) -> str | None:
        valid = [e for e in self.scan() if e.track == track and e.valid]
        chosen = self.repo.get(f"active_model.{track}")
        if chosen and any(e.model_id == chosen for e in valid):
            return chosen
        if not valid:
            return None
        # Default: most recently created checkpoint.
        return max(valid, key=lambda e: (e.meta.get("created_at") or "", e.path.stat().st_mtime)).model_id

    def set_active(self, track: str, model_id: str | None) -> None:
        if model_id is not None:
            e = self.get_entry(model_id)
            if e.track != track:
                raise NotFoundError(f"Model '{model_id}' does not belong to track '{track}'")
            if not e.valid:
                raise ModelUnavailableError(f"Model '{model_id}' is invalid: {e.error}")
        self.repo.set(f"active_model.{track}", model_id)

    def threshold_override(self, track: str) -> float | None:
        v = self.repo.get(f"threshold.{track}")
        return float(v) if v is not None else None

    def set_threshold_override(self, track: str, value: float | None) -> None:
        self.repo.set(f"threshold.{track}", value)

    def effective_threshold(self, track: str, meta: dict[str, Any]) -> tuple[float, str]:
        ov = self.threshold_override(track)
        if ov is not None:
            return ov, "user setting"
        t = meta.get("threshold", {})
        return float(t.get("value", 0.5)), f"checkpoint ({t.get('method', 'fixed')}, selected on {t.get('selected_on', 'n/a')})"

    def pretrained_status(self) -> list[dict[str, Any]]:
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        entries = self.scan()
        out = []
        for arch in ARCHITECTURES:
            cached = (hub / _PRETRAINED_FILES[arch]).exists()
            n_trained = sum(1 for e in entries if e.valid and e.meta.get("arch") == arch)
            status = "trained" if n_trained else ("pretrained" if cached else "unavailable")
            out.append({
                "arch": arch,
                "status": status,
                "trained_checkpoints": n_trained,
                "pretrained_backbone_cached": cached,
                "pretrained_source": PRETRAINED_SOURCES[arch],
                "note": {
                    "trained": "A fine-tuned detection checkpoint is installed.",
                    "pretrained": "Only ImageNet backbone weights are cached. They are a training starting point and cannot detect fakes until fine-tuned.",
                    "unavailable": "No detection checkpoint installed and no pretrained weights cached yet (downloaded automatically when training starts).",
                }[status],
            })
        return out

    def track_status(self) -> list[dict[str, Any]]:
        out = []
        for t in TRACKS:
            mid = self.active_model_id(t)
            out.append({
                "track": t,
                "label": TRACK_LABELS[t],
                "status": "ready" if mid else "unavailable",
                "active_model_id": mid,
                "threshold_override": self.threshold_override(t),
                "message": None if mid else (
                    f"No trained checkpoint installed for this track. Train one with the documented CLI and install it into "
                    f"artifacts/checkpoints/{t}/ (python -m ml.training.train --config ml/configs/train/... --install)."
                ),
            })
        return out

    # ------------------------------------------------------------------ loading
    def require_active(self, track: str) -> LoadedModel:
        if track not in TRACKS:
            raise NotFoundError(f"Unknown track '{track}'")
        mid = self.active_model_id(track)
        if mid is None:
            raise ModelUnavailableError(
                f"No trained model is available for '{TRACK_LABELS[track]}'. No prediction was made.",
                details={"track": track, "how_to_fix": "Train and install a checkpoint: see docs/TRAINING.md"},
            )
        return self.load(mid)

    def load(self, model_id: str) -> LoadedModel:
        with self._cache_lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]
        entry = self.get_entry(model_id)
        if not entry.valid:
            raise ModelUnavailableError(f"Checkpoint '{model_id}' is invalid: {entry.error}")
        try:
            model, meta = load_model_for_inference(entry.path, self.device)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load checkpoint %s", model_id)
            raise ModelUnavailableError(f"Checkpoint '{model_id}' could not be loaded: {type(exc).__name__}") from exc
        pcfg = PreprocessConfig.from_dict(meta["preprocess"])
        lm = LoadedModel(entry, model, meta, pcfg, build_eval_transform(pcfg), self.device)
        with self._cache_lock:
            self._cache[model_id] = lm
            while len(self._cache) > max(1, self.settings.model_cache_size):
                evicted, _ = self._cache.popitem(last=False)
                log.info("Evicted model %s from cache", evicted)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        log.info("Loaded model %s on %s", model_id, self.device)
        return lm

    def loaded_ids(self) -> list[str]:
        with self._cache_lock:
            return list(self._cache)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
