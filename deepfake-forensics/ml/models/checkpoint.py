"""Checkpoint save/load with embedded metadata.

A checkpoint file `<name>.pt` contains:
    {"format_version", "meta", "model_state", optional "optimizer_state", "scheduler_state",
     "scaler_state", "epoch", "best_metric", "history"}
A JSON sidecar `<name>.json` with the same `meta` is written next to it so that model
registries can list checkpoints without deserializing the weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ml.common import CLASS_NAMES, load_json, save_json
from ml.models.factory import DetectorNet, ModelConfig, build_model
from ml.preprocessing.transforms import PreprocessConfig

CHECKPOINT_FORMAT_VERSION = 1
REQUIRED_META_KEYS = ("track", "model_config", "preprocess", "class_names", "threshold")


def sidecar_path(ckpt_path: str | Path) -> Path:
    return Path(ckpt_path).with_suffix(".json")


def save_checkpoint(path: str | Path, model: torch.nn.Module, meta: dict[str, Any], **extra: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"Checkpoint meta is missing keys: {missing}")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "meta": meta,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    save_json(meta, sidecar_path(path))
    return path


def read_checkpoint_meta(path: str | Path) -> dict[str, Any]:
    """Read metadata, preferring the JSON sidecar (fast) over loading the weights."""
    side = sidecar_path(path)
    if side.exists():
        return load_json(side)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["meta"]


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=True: refuses arbitrary pickled objects (safe loading).
    return torch.load(path, map_location=map_location, weights_only=True)


def load_model_for_inference(path: str | Path, device: torch.device) -> tuple[DetectorNet, dict[str, Any]]:
    payload = load_checkpoint_payload(path, map_location="cpu")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    meta = payload["meta"]
    validate_meta(meta)
    mcfg = ModelConfig.from_dict(meta["model_config"])
    mcfg.pretrained = False  # weights come from the checkpoint; never download at inference time
    model = build_model(mcfg)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, meta


def validate_meta(meta: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"Checkpoint meta is missing keys: {missing}")
    if list(meta["class_names"]) != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order {meta['class_names']} != expected {list(CLASS_NAMES)}")
    ModelConfig.from_dict(meta["model_config"])
    PreprocessConfig.from_dict(meta["preprocess"])
